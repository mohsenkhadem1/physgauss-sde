"""Sparse-depth joint point-support/material reference, with a truth-read gate.

Only A frames10/20 enter inference. Every support stratum is integrated using
its exact prior mass. Independent seeds/count refinements diagnose numerical
stability; sensor noise is a stipulated working model, not calibrated model
discrepancy. Held trajectories and physical labels are read after freezing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

try:
    from campaign_point_material_reference import draw_stratified, mask_probabilities, prefix_indices, summarize
    from campaign_material_marginal import MaterialMarginalPrior
    from campaign_hypothesis_inputs import make_prior
    from campaign_exciting_commands import make_exciting_commands, PROGRAM_IDS
    from campaign_sparse_depth_queries import predict_ranges
    from censored_range_observation import CensoredRangeModel, CensoredRanges
except ImportError:
    from experiments.campaign_point_material_reference import draw_stratified, mask_probabilities, prefix_indices, summarize
    from experiments.campaign_material_marginal import MaterialMarginalPrior
    from experiments.campaign_hypothesis_inputs import make_prior
    from experiments.campaign_exciting_commands import make_exciting_commands, PROGRAM_IDS
    from experiments.campaign_sparse_depth_queries import predict_ranges
    from experiments.censored_range_observation import CensoredRangeModel, CensoredRanges

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def checked(record):
    path = (ROOT/record['path']).resolve()
    if not path.is_relative_to(ROOT) or sha(path) != record['sha256']:
        raise ValueError('Pinned artifact changed: '+record['path'])
    return path


def stability(references, seeds, counts):
    largest = [r for r in references if r['count_per_mask'] == max(counts)]
    supports = np.array([[s['posterior_probability'] for s in r['support_categories']] for r in largest])
    tv = max(.5*np.abs(a-b).sum() for a in supports for b in supports)
    result = dict(minimum_largest_run_ESS=min(r['ESS'] for r in largest),
        largest_run_log_evidence_range=float(np.ptp([r['log_evidence'] for r in largest])),
        maximum_independent_seed_support_total_variation=float(tv), material=[],
        qualification='Monte Carlo integration diagnostics, not a physical-calibration certificate')
    refinements = []
    for seed in seeds:
        rows = sorted((r for r in references if r['seed']==seed),key=lambda r:r['count_per_mask'])
        a,b = rows[0],rows[-1]
        pa = np.array([s['posterior_probability'] for s in a['support_categories']])
        pb = np.array([s['posterior_probability'] for s in b['support_categories']])
        refinements.append(dict(seed=seed,support_total_variation=float(.5*np.abs(pa-pb).sum()),
            absolute_log_evidence_shift=float(abs(a['log_evidence']-b['log_evidence']))))
    result['particle_refinement_by_seed'] = refinements
    for j in range(4):
        sd = float(np.mean([r['material'][j]['std'] for r in largest]))
        means = [r['material'][j]['mean'] for r in largest]
        q = np.array([[r['material'][j][key] for key in ['quantile05','median','quantile95']] for r in largest])
        shifts = []
        for seed in seeds:
            rows = [r for r in references if r['seed'] == seed]
            shifts.append(abs(rows[0]['material'][j]['mean']-rows[-1]['material'][j]['mean']))
        result['material'].append(dict(name=largest[0]['material'][j]['name'],
            independent_seed_mean_range_in_posterior_std=float(np.ptp(means)/sd) if sd else None,
            independent_seed_maximum_quantile_range_in_posterior_std=float(np.ptp(q,axis=0).max()/sd) if sd else None,
            maximum_particle_refinement_mean_shift_in_posterior_std=float(max(shifts)/sd) if sd else None))
    return result


def resample(weights, count, seed):
    cdf = np.cumsum(weights)
    cdf[-1] = 1.
    return np.searchsorted(cdf, (np.random.default_rng(seed).random()+np.arange(count))/count, side='right')


def run(design_path, cache_path, cache_sha256, output, *, device, batch_size):
    # Import after CLI parsing: tests of the numerical measure need no cache.
    try:
        from campaign_support_basis_cache import load_support_basis_cache
    except ImportError:
        from experiments.campaign_support_basis_cache import load_support_basis_cache
    torch.set_num_threads(2)
    output = Path(output)
    if output.exists() or batch_size not in (256, 512):
        raise ValueError('Fresh output and prospectively benchmarked batch256/512 required')
    design = read(design_path)
    for path, digest in design['source_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Frozen design source changed: '+path)
    forward, public, ids, distribution = load_support_basis_cache(cache_path, cache_sha256)
    probabilities = mask_probabilities(distribution)
    if set(ids) != set(probabilities):
        raise ValueError('Incomplete support cache')
    if public != read(checked(design['public_hardware'])):
        raise ValueError('Cache and observations must share public hardware')
    if distribution != read(checked(design['support_prior']['distribution'])):
        raise ValueError('Cache and truth design must share the complete normalized support prior')
    prior = MaterialMarginalPrior(make_prior(public))
    if prior.metadata() != design['material_prior']:
        raise ValueError('Truth and reference material sampling laws differ')
    queries = read(checked(design['public_acquired_queries']))
    if sorted({q['frame_index'] for q in queries}) != [10,20] or sum(len(q['xs']) for q in queries) != 72:
        raise ValueError('Exactly the frozen72 prefix depth rays required')
    protocols = [read(checked(r)) for r in design['physics_cases']]
    directories = [ROOT/p['output'] for p in protocols]
    for directory in directories:
        e, s = read(directory/'execution.json'), read(directory/'summary.json')
        if not e['terminal'] or not e['completed'] or not s['completed'] or sha(directory/'trajectory.npz') != s['trajectory_sha256']:
            raise ValueError('Both finite captures must finish without retry')
    source_names = ['campaign_point_material_reference.py', 'run_campaign_point_material_reference.py',
        'campaign_support_basis_cache.py', 'campaign_packed_boundary_forward.py',
        'campaign_support_mask_prior.py', 'campaign_material_marginal.py',
        'campaign_sparse_depth_queries.py', 'censored_range_observation.py',
        'evaluate_campaign_point_material_reference.py']
    sources = {**design['source_sha256'], **read(cache_path)['source_sha256'],
               **{'experiments/'+n:sha(ROOT/'experiments'/n) for n in source_names}}
    plan = design['reference_plan']
    counts, seeds = plan['counts_per_mask'], plan['seeds']
    if counts != [256,1024] or seeds != [20261112,20261113,20261114]:
        raise ValueError('Prospective material-per-mask counts and seeds required')
    output.mkdir(parents=True, exist_ok=False)
    write(output/'protocol.json', dict(design_path=str(design_path),design_sha256=sha(design_path),
        cache_path=str(cache_path),cache_sha256=cache_sha256,source_sha256=sources,
        counts_per_mask=counts,seeds=seeds,device=device,batch_size=batch_size,
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        method='Independent material prior draws stratified across ALL55 exact-mass point-pin masks',
        integration_weights='p(mask)/N_mask times likelihood',
        support_type_known=True, support_location_known=False, subnode_center_identifiable=False,
        noise_model_calibrated=False, known_forces_inferred=False, raw_outputs_LOCAL=True))
    started = time.monotonic()
    status = dict(pid=os.getpid(),terminal=False,completed=False,processed_particles=0)
    write(output/'execution.json',status)
    noise = design['likelihood']
    likelihood = CensoredRangeModel(np.eye(72)*noise['range_std_m']**2,
        noise['visibility_error_probability'],noise['background_std_m'],noise['outlier_probability'])
    rest = np.asarray(public['rest_points_m'])
    faces = np.asarray(public['surface_faces'])
    clocks = [make_exciting_commands(p) for p in PROGRAM_IDS]
    try:
        # No unacquired trajectory frame is supplied to the sensor adapter.
        acquired = np.broadcast_to(rest,(1,61,len(rest),3)).copy()
        with np.load(directories[0]/'trajectory.npz',allow_pickle=False) as f:
            acquired[0,[10,20]] = f['positions_m'][[10,20]]
        with torch.no_grad():
            predictions = predict_ranges(acquired,faces,queries,device=device,particle_chunk=64)
        noisy,_ = likelihood.fantasies(predictions,count=1,seed=noise['observation_seed'])
        observation = CensoredRanges(noisy.visible[0],noisy.residual_m[0])
        np.savez_compressed(output/'acquired-observation.npz',visible=observation.visible,residual_m=observation.residual_m)
        del acquired, predictions, noisy
        references = []
        for seed in seeds:
            geometry, physical = draw_stratified(prior,distribution,count_per_mask=max(counts),seed=seed)
            ll = np.empty(len(geometry))
            # Keep the observation-independent sensor library LOCAL. Independent
            # calibration cases can reuse these forward queries without reruns.
            predicted_visible = np.empty((len(geometry),72),dtype=bool)
            predicted_ranges = np.empty((len(geometry),72),dtype=np.float64)
            for lo in range(0,len(geometry),batch_size):
                with torch.no_grad():
                    q = forward.predict(clocks[0],torch.tensor(geometry[lo:lo+batch_size]),
                        torch.tensor(physical[lo:lo+batch_size]),device=device)['positions_m']
                    rays = predict_ranges(q,faces,queries,device=device,particle_chunk=64)
                ll[lo:lo+batch_size] = likelihood.log_prob(rays,observation)
                predicted_visible[lo:lo+batch_size] = rays.visible
                predicted_ranges[lo:lo+batch_size] = rays.residual_m
                del q, rays
                status['processed_particles'] += min(batch_size,len(geometry)-lo)
                if lo % (batch_size*16) == 0:
                    status.update(seed=seed,elapsed_s=time.monotonic()-started)
                    write(output/'execution.json',status)
                    print(json.dumps(status),flush=True)
                    if any(sha(ROOT/p)!=digest for p,digest in sources.items()):
                        raise ValueError('Frozen reference source changed')
            for count in counts:
                rows = prefix_indices(geometry,count,probabilities)
                report, _ = summarize(geometry[rows],physical[rows],ll[rows],probabilities)
                references.append(dict(seed=seed,count_per_mask=count,**report))
            _, weights = summarize(geometry,physical,ll,probabilities)
            np.savez_compressed(output/f'reference-{seed}.npz',geometry_ids=geometry,
                physical_coordinates=physical,log_likelihood=ll,weights=weights,
                predicted_visible=predicted_visible,predicted_residual_m=predicted_ranges)
            write(output/'reference-summaries.json',references)
        diagnostics = stability(references,seeds,counts)
        write(output/'reference-stability.json',diagnostics)
        # This gate binds every complete reference BEFORE any labels/held frames.
        write(output/'posterior-frozen.json',dict(reference_summaries_sha256=sha(output/'reference-summaries.json'),
            references=[dict(path=str(output/f'reference-{s}.npz'),sha256=sha(output/f'reference-{s}.npz')) for s in seeds],
            source_sha256=sources,physical_truth_opened=False,held_motion_used_for_inference=False,
            held_motion_evaluated=False,
            acquisition_scope='NPZ positions array decompressed; only frames10/20 selected and supplied to sensor/likelihood'))
        status.update(terminal=True,completed=True,elapsed_s=time.monotonic()-started,
            posterior_frozen=True,evaluation_pending=True)
        write(output/'execution.json',status)
        return diagnostics
    except BaseException as exc:
        status.update(terminal=True,completed=False,elapsed_s=time.monotonic()-started,error=repr(exc))
        write(output/'execution.json',status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',required=True)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--cache-sha256',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.design,args.cache,args.cache_sha256,args.output,device=args.device,batch_size=args.batch_size)))
