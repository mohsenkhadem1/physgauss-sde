"""Evaluate a frozen unknown-point/material posterior against held finite motion.

Equal 2048-particle prior/posterior approximations receive identical held noisy
depth observations. Proper mixture predictive log scores complement mean
motion error. One object and stipulated sensor noise establish no coverage or
forward-discrepancy calibration. Reset program B is not a continuing poke.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from run_campaign_point_material_reference import ROOT, sha, read, write, checked, resample
    from campaign_point_material_reference import mask_probabilities
    from campaign_exciting_commands import PROGRAM_IDS, make_exciting_commands
    from campaign_sparse_depth_queries import predict_ranges
    from censored_range_observation import CensoredRangeModel, CensoredRanges, _logsumexp
except ImportError:
    from experiments.run_campaign_point_material_reference import ROOT, sha, read, write, checked, resample
    from experiments.campaign_point_material_reference import mask_probabilities
    from experiments.campaign_exciting_commands import PROGRAM_IDS, make_exciting_commands
    from experiments.campaign_sparse_depth_queries import predict_ranges
    from experiments.censored_range_observation import CensoredRangeModel, CensoredRanges, _logsumexp


def run(reference_path):
    try:
        from campaign_support_basis_cache import load_support_basis_cache
    except ImportError:
        from experiments.campaign_support_basis_cache import load_support_basis_cache
    torch.set_num_threads(2)
    output = Path(reference_path)
    protocol = read(output/'protocol.json')
    gate = read(output/'posterior-frozen.json')
    if gate['reference_summaries_sha256'] != sha(output/'reference-summaries.json'):
        raise ValueError('Posterior summaries changed before truth gate')
    for record in gate['references']:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Frozen reference particles changed')
    for path,digest in gate['source_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Frozen inference/evaluation dependency changed: '+path)
    if (output/'evaluation.json').exists():
        raise ValueError('Do not overwrite a completed evaluation')
    design = read(protocol['design_path'])
    if sha(protocol['design_path']) != protocol['design_sha256']:
        raise ValueError('Prospective design changed')
    forward, public, _, distribution = load_support_basis_cache(protocol['cache_path'],protocol['cache_sha256'])
    probabilities = mask_probabilities(distribution)
    queries = read(checked(design['public_held_queries']))
    if sorted({q['frame_index'] for q in queries}) != [30,50]:
        raise ValueError('Only prospectively held frames required')
    first_seed = protocol['seeds'][0]
    with np.load(output/f'reference-{first_seed}.npz',allow_pickle=False) as f:
        geometry,physical,weights = f['geometry_ids'],f['physical_coordinates'],f['weights']
    prior_weights = np.empty(len(geometry))
    for node,mass in probabilities.items():
        rows = geometry == node
        prior_weights[rows] = mass/rows.sum()
    selected = dict(prior=resample(prior_weights,2048,20261118),
                    posterior=resample(weights,2048,20261115))
    # Only now are held trajectories/physical truth read. Inference never calls
    # this module; its source is pinned by the prospective run protocol.
    truth_spec = read(checked(design['truth_spec']))
    support_truth = int(truth_spec['fixture_nodes'][0])
    summaries = read(output/'reference-summaries.json')
    summary = next(r for r in summaries if r['seed']==first_seed and r['count_per_mask']==1024)
    support_rows = sorted(summary['support_categories'],key=lambda r:-r['posterior_probability'])
    actual = next(r for r in support_rows if r['node_id']==support_truth)
    material_truth = [truth_spec['material'][k] for k in
        ['youngs_modulus_Pa','density_kg_m3','poissons_ratio','body_linear_drag_s_inverse']]
    material_checks = [dict(name=r['name'],truth=float(x),in_90_percent_interval=r['quantile05']<=x<=r['quantile95'])
        for r,x in zip(summary['material'],material_truth)]
    noise = design['likelihood']
    likelihood = CensoredRangeModel(np.eye(72)*noise['range_std_m']**2,
        noise['visibility_error_probability'],noise['background_std_m'],noise['outlier_probability'])
    device,batch_size = protocol['device'],protocol['batch_size']
    rest,faces = np.asarray(public['rest_points_m']),np.asarray(public['surface_faces'])
    evaluations = []
    for program,case,observation_seed in zip(PROGRAM_IDS,design['physics_cases'],[20261116,20261117]):
        physics = read(checked(case))
        directory = ROOT/physics['output']
        physics_summary = read(directory/'summary.json')
        if sha(directory/'trajectory.npz') != physics_summary['trajectory_sha256']:
            raise ValueError('Held physical trajectory changed')
        with np.load(directory/'trajectory.npz',allow_pickle=False) as f:
            truth = f['positions_m'].copy()
        with torch.no_grad():
            truth_ranges = predict_ranges(truth[None],faces,queries,device=device,particle_chunk=64)
        noisy,_ = likelihood.fantasies(truth_ranges,count=1,seed=observation_seed)
        observation = CensoredRanges(noisy.visible[0],noisy.residual_m[0])
        np.savez_compressed(output/f'held-observation-{program}.npz',visible=observation.visible,residual_m=observation.residual_m)
        row = dict(program=program,observation_seed=observation_seed,
            interpretation='A future beyond acquired1s prefix' if program==PROGRAM_IDS[0] else 'B reset-force prediction, not a continuing poke')
        for label,rows in selected.items():
            total = np.zeros((61,len(rest),3))
            second = np.zeros_like(total)
            held_log_likelihood = []
            clock = make_exciting_commands(program)
            for lo in range(0,len(rows),batch_size):
                part = rows[lo:lo+batch_size]
                with torch.no_grad():
                    q = forward.predict(clock,torch.tensor(geometry[part]),torch.tensor(physical[part]),device=device)['positions_m']
                    rays = predict_ranges(q,faces,queries,device=device,particle_chunk=64)
                held_log_likelihood.append(likelihood.log_prob(rays,observation))
                u = q.cpu().numpy()-rest
                total += u.sum(0)/len(rows)
                second += np.square(u).sum(0)/len(rows)
            variance = second-np.square(total)
            tolerance = 64*np.finfo(float).eps*np.maximum(second,np.square(total))
            if np.any(variance < -tolerance):
                raise ValueError('Moment accumulation lost precision')
            variance = np.maximum(variance,0.)
            mean = rest+total
            error = mean-truth
            ll = np.concatenate(held_log_likelihood)
            row[label] = dict(particles=2048,
                position_mean_RMSE_m=float(np.sqrt(np.mean(error**2))),
                after_1s_mean_RMSE_m=float(np.sqrt(np.mean(error[21:]**2))),
                latent_standard_deviation_RMS_m=float(np.sqrt(np.mean(variance))),
                held_noisy_depth_mixture_predictive_log_density=float(_logsumexp(ll)-np.log(len(ll))))
            np.savez_compressed(output/f'{label}-predictive-{program}.npz',mean_positions_m=mean,
                latent_std_m=np.sqrt(variance),times_s=np.arange(61)/20)
        row['static_RMSE_m'] = float(np.sqrt(np.mean((truth-rest)**2)))
        evaluations.append(row)
    result = dict(schema='unknown-point-support-material-depth-oracle-evaluation-v0.1',
        first_seed_fixed_before_evaluation=first_seed,held_motion=evaluations,
        true_support_node=support_truth,true_support_prior_probability=actual['prior_probability'],
        true_support_posterior_probability=actual['posterior_probability'],
        true_support_posterior_rank=1+next(i for i,r in enumerate(support_rows) if r['node_id']==support_truth),
        one_truth_material_interval_checks=material_checks,
        scope='One fresh proper-prior object; point-pin type known; unknown node/material; working approximate linear mechanics and stipulated sensor noise.',
        limitations=['No subnode support-center localization','No image encoder or RGB Gaussian rendering',
            'No calibrated forward-model discrepancy','No population coverage claim','No sequential active policy result'])
    write(output/'evaluation.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.reference)))
