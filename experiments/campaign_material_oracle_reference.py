"""Independent-prior numerical material reference from sparse prefix depth.

One fresh material draw, known support geometry/hardware. A modal linear
approximation scores finite-FEM depth observations; this isolates a numerical
inference diagnostic, not unknown-boundary inference or a calibration study.
All held motion and material truth are opened after posterior files freeze.
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
    from campaign_material_marginal import MaterialMarginalPrior
    from campaign_hypothesis_inputs import make_prior, known_hardware_from_public
    from campaign_exciting_commands import make_exciting_commands, PROGRAM_IDS
    from campaign_linear_hypothesis import LinearHypothesisDynamics
    from campaign_modal_hypothesis import FixedModalLinearDynamics
    from campaign_modal_static_lift import complete_static_lift
    from campaign_modal_material_forward import BatchedModalMaterialForward
    from campaign_sparse_depth_queries import predict_ranges
    from censored_range_observation import CensoredRangeModel, CensoredRanges
    from campaign_tempered_reference import normalized_log_weights
except ImportError:
    from experiments.campaign_material_marginal import MaterialMarginalPrior
    from experiments.campaign_hypothesis_inputs import make_prior, known_hardware_from_public
    from experiments.campaign_exciting_commands import make_exciting_commands, PROGRAM_IDS
    from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
    from experiments.campaign_modal_hypothesis import FixedModalLinearDynamics
    from experiments.campaign_modal_static_lift import complete_static_lift
    from experiments.campaign_modal_material_forward import BatchedModalMaterialForward
    from experiments.campaign_sparse_depth_queries import predict_ranges
    from experiments.censored_range_observation import CensoredRangeModel, CensoredRanges
    from experiments.campaign_tempered_reference import normalized_log_weights

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def checked(record):
    path = (ROOT/record['path']).resolve()
    if not path.is_relative_to(ROOT) or sha(path)!=record['sha256']:
        raise ValueError('Pinned artifact changed: '+record['path'])
    return path


def importance_summary(physical,log_likelihood):
    weights,normalizer = normalized_log_weights(log_likelihood)
    fields = []
    for j,name in enumerate(['E_Pa','rho_kg_m3','nu','body_drag_s_inverse']):
        x = physical[:,j]
        mean = float(weights@x)
        sd = float(np.sqrt(weights@np.square(x-mean)))
        order = np.argsort(x,kind='stable')
        bounds = np.interp([.05,.5,.95],np.cumsum(weights[order]),x[order])
        fields.append(dict(name=name,mean=mean,std=sd,quantile05=float(bounds[0]),
                           median=float(bounds[1]),quantile95=float(bounds[2])))
    centered = physical-weights@physical
    covariance = centered.T@(weights[:,None]*centered)
    return dict(particles=len(weights),ESS=float(1/np.square(weights).sum()),
        maximum_weight=float(weights.max()),log_evidence=float(normalizer-np.log(len(weights))),
        weighted_physical_material_covariance=covariance.tolist(),material=fields),weights


def run(design_path,output,*,device='cuda:0',batch_size=512):
    torch.set_num_threads(2)
    design_path = Path(design_path)
    design = read(design_path)
    output = Path(output)
    if output.exists() or device!='cuda:0' or batch_size!=512:
        raise ValueError('Fresh output and benchmarked physicalGPU0/batch512 required')
    for name,digest in design['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Frozen oracle dependency changed: '+name)
    protocols = [read(checked(r)) for r in design['physics_cases']]
    directories = [ROOT/p['output'] for p in protocols]
    for directory in directories:
        execution,summary = read(directory/'execution.json'),read(directory/'summary.json')
        if not execution['terminal'] or not execution['completed'] or not summary['completed']:
            raise ValueError('Both fresh finite captures must complete without retries')
        if sha(directory/'trajectory.npz')!=summary['trajectory_sha256']:
            raise ValueError('Fresh finite trajectory changed')
    public = read(checked(design['public_hardware']))
    template = read(checked(design['nominal_support_template']))
    prior = MaterialMarginalPrior(make_prior(public))
    if prior.metadata()!=design['material_prior']:
        raise ValueError('Numerical reference prior differs from truth sampling law')
    queries = read(checked(design['public_acquired_queries']))
    held_queries = read(checked(design['public_held_queries']))
    sources = {**design['source_sha256'],str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    # Bind code and scheduling before observations or likelihood evaluations.
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',dict(design_path=str(design_path),design_sha256=sha(design_path),
        source_sha256=sources,device=device,batch_size=batch_size,
        method='Independent prior importance sampling; all particles retained; no proposal fit',
        posterior_predictive_particles=2048,predictive_seed=20261104,
        known_support_oracle=True,unknown_boundary_inference=False,
        calibrated_uncertainty_established=False,raw_outputs_LOCAL=True))
    status = dict(pid=os.getpid(),terminal=False,completed=False,processed_particles=0)
    write(output/'execution.json',status)
    started = time.monotonic()
    try:
        known = known_hardware_from_public(public)
        full = LinearHypothesisDynamics.from_hypothesis(known,template)
        modal = complete_static_lift(FixedModalLinearDynamics(full,mode_count=96))
        forward = BatchedModalMaterialForward(modal)
        clocks = [make_exciting_commands(p) for p in PROGRAM_IDS]
        noise = design['likelihood']
        dimension = sum(len(q['xs']) for q in queries)
        likelihood = CensoredRangeModel(np.eye(dimension)*noise['range_std_m']**2,
            noise['visibility_error_probability'],noise['background_std_m'],noise['outlier_probability'])
        # Only acquired frames enter the observation; future frames are never
        # passed to the likelihood. The sensor helper has frame-isolation tests.
        with np.load(directories[0]/'trajectory.npz',allow_pickle=False) as f:
            selected = sorted({q['frame_index'] for q in queries})
            acquired_truth = np.broadcast_to(known.points,(1,61,len(known.points),3)).copy()
            acquired_truth[0,selected] = f['positions_m'][selected]
        with torch.no_grad():
            acquired = predict_ranges(acquired_truth,known.faces,queries,device=device,particle_chunk=64)
        noisy,_ = likelihood.fantasies(acquired,count=1,seed=noise['observation_seed'])
        observation = CensoredRanges(noisy.visible[0],noisy.residual_m[0])
        np.savez_compressed(output/'acquired-observation.npz',visible=observation.visible,
                             residual_m=observation.residual_m)
        del acquired_truth,acquired,noisy
        references = []
        first_physical = first_weights = None
        counts = design['reference_plan']['particles_per_seed']
        for seed in design['reference_plan']['seeds']:
            state = prior.sample(max(counts),np.random.default_rng(seed))
            physical = prior.physical_coordinates(state)
            ll = np.empty(len(state))
            for lo in range(0,len(state),batch_size):
                with torch.no_grad():
                    result = forward.predict(clocks[0],torch.tensor(physical[lo:lo+batch_size]),device=device)
                    ranges = predict_ranges(result['positions_m'],known.faces,queries,device=device,particle_chunk=64)
                ll[lo:lo+batch_size] = likelihood.log_prob(ranges,observation)
                status['processed_particles'] += len(ranges.visible)
                if lo%(8*batch_size)==0:
                    status.update(seed=seed,elapsed_s=time.monotonic()-started)
                    write(output/'execution.json',status)
                    print(json.dumps(status),flush=True)
                    if any(sha(ROOT/p)!=digest for p,digest in sources.items()):
                        raise ValueError('Frozen reference source changed')
            for count in counts:
                report,weights = importance_summary(physical[:count],ll[:count])
                references.append(dict(seed=seed,**report))
            _,weights = importance_summary(physical,ll)
            np.savez_compressed(output/('reference-'+str(seed)+'.npz'),
                intrinsic_coordinates=state.coordinates,physical_coordinates=physical,
                log_likelihood=ll,weights=weights)
            if first_physical is None:
                first_physical,first_weights = physical,weights
            write(output/'reference-summaries.json',references)
        largest = [r for r in references if r['particles']==max(counts)]
        stability = dict(minimum_largest_run_ESS=min(r['ESS'] for r in largest),
            largest_run_log_evidence_range=max(r['log_evidence'] for r in largest)-min(r['log_evidence'] for r in largest),
            variables=[],scope='Independent-prior Monte Carlo checks, not a calibrated-posterior certificate')
        stability['minimum_ESS_criterion_pass'] = stability['minimum_largest_run_ESS']>=design['reference_plan']['minimum_ESS_for_stability_assessment']
        for j,name in enumerate(['E_Pa','rho_kg_m3','nu','body_drag_s_inverse']):
            sd = np.mean([r['material'][j]['std'] for r in largest])
            means = [r['material'][j]['mean'] for r in largest]
            quantiles = np.array([[r['material'][j][k] for k in ['quantile05','median','quantile95']] for r in largest])
            shifts = []
            for seed in design['reference_plan']['seeds']:
                a,b = [r for r in references if r['seed']==seed]
                shifts.append(abs(a['material'][j]['mean']-b['material'][j]['mean']))
            stability['variables'].append(dict(name=name,
                independent_seed_mean_range_in_posterior_std=float(np.ptp(means)/sd) if sd>0 else None,
                independent_seed_maximum_quantile_range_in_posterior_std=float(np.ptp(quantiles,axis=0).max()/sd) if sd>0 else None,
                maximum_particle_refinement_mean_shift_in_posterior_std=float(max(shifts)/sd) if sd>0 else None))
        write(output/'reference-stability.json',stability)
        write(output/'posterior-frozen.json',dict(reference_summaries_sha256=sha(output/'reference-summaries.json'),
            source_sha256=sources,held_motion_opened=False,material_truth_opened=False,
            inference_completed_without_truth_labels=True))
        # Fixed first-seed posterior, not a seed selected using held scores.
        cdf = np.cumsum(first_weights)
        cdf[-1] = 1.
        positions = (np.random.default_rng(20261104).random()+np.arange(2048))/2048
        ids = np.searchsorted(cdf,positions,side='right')
        posterior = first_physical[ids]
        evaluation = []
        for program,directory,clock in zip(PROGRAM_IDS,directories,clocks):
            mean_displacement = np.zeros((61,len(known.points),3))
            second = np.zeros_like(mean_displacement)
            for lo in range(0,len(posterior),batch_size):
                with torch.no_grad():
                    q = forward.predict(clock,torch.tensor(posterior[lo:lo+batch_size]),device=device)['positions_m'].cpu().numpy()
                displacement = q-known.points
                mean_displacement += displacement.sum(0)/len(posterior)
                second += np.square(displacement).sum(0)/len(posterior)
            mean = known.points+mean_displacement
            variance = second-np.square(mean_displacement)
            # Roundoff in E[x²]-E[x]² is bounded; the stored variance must not
            # silently hide a substantial negative numerical result.
            tolerance = 64*np.finfo(float).eps*np.maximum(second,np.square(mean_displacement))
            if np.any(variance < -tolerance):
                raise ValueError('Posterior moment accumulation lost precision')
            std = np.sqrt(np.maximum(variance,0.))
            with np.load(directory/'trajectory.npz',allow_pickle=False) as f:
                truth = f['positions_m'].copy()
            rmse = lambda x:float(np.sqrt(np.mean(np.square(x))))
            evaluation.append(dict(program=program,position_mean_RMSE_m=rmse(mean-truth),
                after_1s_mean_RMSE_m=rmse((mean-truth)[np.arange(61)/20>1]),
                static_RMSE_m=rmse(truth-known.points),
                interpretation='A future beyond acquired1s prefix' if program==PROGRAM_IDS[0] else 'B held reset-force prediction, not a continuing poke'))
            np.savez_compressed(output/('posterior-predictive-'+program+'.npz'),
                mean_positions_m=mean,material_posterior_std_m=std,times_s=np.arange(61)/20)
            with torch.no_grad():
                held_ranges = predict_ranges(mean[None],known.faces,held_queries,device=device,particle_chunk=64)
                actual_ranges = predict_ranges(truth[None],known.faces,held_queries,device=device,particle_chunk=64)
            shared = held_ranges.visible&actual_ranges.visible
            evaluation[-1].update(held_camera_depth_RMSE_of_mean_surface_m=rmse(
                held_ranges.residual_m[shared]-actual_ranges.residual_m[shared]) if shared.any() else None,
                held_camera_visibility_disagreements=int((held_ranges.visible!=actual_ranges.visible).sum()),
                depth_score_scope='Depth of posterior mean surface on common hits; not posterior predictive log score or unconditional coverage')
        truth_spec = read(checked(design['truth_spec']))
        physical_truth = [truth_spec['material'][k] for k in ['youngs_modulus_Pa','density_kg_m3','poissons_ratio','body_linear_drag_s_inverse']]
        first_reference = next(r for r in references if r['seed']==design['reference_plan']['seeds'][0] and r['particles']==max(counts))
        truth_checks = [dict(name=r['name'],truth=value,in_90_percent_interval=r['quantile05']<=value<=r['quantile95'])
                        for r,value in zip(first_reference['material'],physical_truth)]
        report = dict(schema='known-support-material-depth-oracle-result-v0.1',references=references,reference_stability=stability,
            predictive_evaluation=evaluation,one_truth_material_interval_checks=truth_checks,
            interpretation='One known-support engineering oracle. Independent-seed/count agreement must be assessed; ESS alone is not posterior accuracy. Modal/small-strain discrepancy remains.',
            calibrated_material_coverage_established=False,unknown_boundary_inference=False,
            active_probing_established=False,sparse_RGB_encoder_trained=False,
            physical_Gaussian_covariance_is_not_posterior_uncertainty=True)
        write(output/'report.json',report)
        status.update(terminal=True,completed=True,elapsed_s=time.monotonic()-started,report_sha256=sha(output/'report.json'))
        write(output/'execution.json',status)
        return report
    except BaseException as error:
        status.update(terminal=True,completed=False,error=type(error).__name__,message=str(error),elapsed_s=time.monotonic()-started)
        write(output/'execution.json',status)
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    report = run(args.design,args.output)
    print(json.dumps(report['predictive_evaluation']))
