"""Posthoc proper-prior predictive diagnostic, with frozen conditioned baseline.

Exactly8192 material draws, seed20261105, known support/H, no conditioning.
Chunk512 online displacement moments; no full8192-trajectory storage or fit.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
try:
    from campaign_material_forward_controls import ROOT,DESIGN,DESIGN_SHA,MARKER,resolve,read,checked,sha
    from campaign_hypothesis_inputs import known_hardware_from_public,make_prior
    from campaign_material_marginal import MaterialMarginalPrior
    from campaign_linear_hypothesis import LinearHypothesisDynamics
    from campaign_modal_hypothesis import FixedModalLinearDynamics
    from campaign_modal_static_lift import complete_static_lift
    from campaign_modal_material_forward import BatchedModalMaterialForward
    from campaign_exciting_commands import make_exciting_commands,PROGRAM_IDS
except ImportError:
    from experiments.campaign_material_forward_controls import ROOT,DESIGN,DESIGN_SHA,MARKER,resolve,read,checked,sha
    from experiments.campaign_hypothesis_inputs import known_hardware_from_public,make_prior
    from experiments.campaign_material_marginal import MaterialMarginalPrior
    from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
    from experiments.campaign_modal_hypothesis import FixedModalLinearDynamics
    from experiments.campaign_modal_static_lift import complete_static_lift
    from experiments.campaign_modal_material_forward import BatchedModalMaterialForward
    from experiments.campaign_exciting_commands import make_exciting_commands,PROGRAM_IDS


def moments(mean,variance,truth,conditioned_mean,conditioned_variance,MCSE,selection):
    a,b,c,d,e,f,g=[x[selection] for x in [mean,variance,truth,conditioned_mean,conditioned_variance,MCSE,mean-conditioned_mean]]
    rmse=lambda x:float(np.sqrt(np.mean(x*x)))
    return {'prior_mean_coordinate_RMSE_m':rmse(a-c),'conditioned_mean_coordinate_RMSE_m':rmse(d-c),
        'prior_to_conditioned_mean_change_coordinate_RMS_m':rmse(g),
        'prior_latent_coordinate_variance_mean_m2':float(b.mean()),
        'conditioned_latent_coordinate_variance_mean_m2':float(e.mean()),
        'prior_latent_coordinate_variance_RMS_m2':rmse(b),'conditioned_latent_coordinate_variance_RMS_m2':rmse(e),
        'prior_latent_coordinate_std_RMS_m':float(np.sqrt(b.mean())),
        'conditioned_latent_coordinate_std_RMS_m':float(np.sqrt(e.mean())),
        'conditioned_over_prior_mean_latent_variance_ratio':float(e.mean()/b.mean()) if b.mean()>0 else None,
        'prior_mean_per_coordinate_MCSE_RMS_m':rmse(f),'prior_mean_per_coordinate_MCSE_max_m':float(f.max()),
        'relative_mean_RMSE_change_from_prior':float(1-rmse(d-c)/rmse(a-c)) if rmse(a-c)>0 else None}


def run(output):
    marker_path=resolve(MARKER);marker=read(marker_path)
    if not (marker['inference_completed_without_truth_labels'] and marker['held_motion_opened'] is False and marker['material_truth_opened'] is False):raise ValueError('Frozen inference ordering invalid')
    reference=marker_path.parent
    if sha(reference/'reference-summaries.json')!=marker['reference_summaries_sha256'] or sha(resolve(DESIGN))!=DESIGN_SHA:raise ValueError('Frozen design/reference changed')
    for path,digest in marker['source_sha256'].items():
        if sha(resolve(path))!=digest:raise ValueError('Frozen dependency changed')
    design=read(resolve(DESIGN));public=read(checked(design['public_hardware']));template=read(checked(design['nominal_support_template']))
    known=known_hardware_from_public(public);prior=MaterialMarginalPrior(make_prior(public))
    if prior.metadata()!=design['material_prior']:raise ValueError('Prior differs from inference/reference')
    out=resolve(output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_device(0)
    names=['campaign_material_prior_predictive.py','campaign_material_forward_controls.py','campaign_material_marginal.py',
        'campaign_modal_hypothesis.py','campaign_modal_static_lift.py','campaign_modal_material_forward.py','campaign_linear_hypothesis.py']
    sources={'experiments/'+name:sha(ROOT/'experiments'/name) for name in names}
    protocol={'protocol':'posthoc-unconditioned-material-prior-predictive-v0.1','selection_status':'EXPLICITLY_POSTHOC_AFTER_POSTERIOR_RESULTS',
        'draws':8192,'seed':20261105,'chunk':512,'GPU':0,'CPU_threads':2,'programs':list(PROGRAM_IDS),
        'known_support_template':design['nominal_support_template'],'public_hardware':design['public_hardware'],
        'proper_prior':prior.metadata(),'source_sha256':sources,'design_sha256':DESIGN_SHA,'frozen_marker_sha256':sha(marker_path),
        'variance_estimator':'Online centered-displacement Chan merge, unbiasedsampleM2/(N-1). MeanMCSE=sqrt(samplevariance/N).',
        'conditioned_variance_scope':'Frozenroot2048systematicposteriorpredictive-resample populationvariance; no newposteriorquery.',
        'no_8192_full_trajectory_storage':True,'no_fit_or_inference_rerun':True,'new_finite_runs':0,
        'coverage_claim':False,'future_A_selection':'times>1s','held_B':'freshresetprogramB, not acontinuingpoke'}
    # Protocol exists before draw generation, model construction or baselinecall.
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.monotonic();state=prior.sample(8192,np.random.default_rng(20261105));physical=prior.physical_coordinates(state)
    full=LinearHypothesisDynamics.from_hypothesis(known,template);modal=complete_static_lift(FixedModalLinearDynamics(full,96))
    forward=BatchedModalMaterialForward(modal);rest=torch.tensor(known.points.copy(),dtype=torch.float64,device='cuda:0');saved={};timings={}
    for program in PROGRAM_IDS:
        clock=make_exciting_commands(program)
        torch.cuda.synchronize(0);begin=time.monotonic();mean=M2=None;count=0
        with torch.no_grad():
            for lo in range(0,8192,512):
                q=forward.predict(clock,torch.tensor(physical[lo:lo+512]),device='cuda:0')['positions_m']
                delta=q-rest;block_mean=delta.mean(0);block_M2=(delta-block_mean).square().sum(0);n=len(q)
                if mean is None:mean=block_mean;M2=block_M2;count=n
                else:
                    difference=block_mean-mean;combined=count+n
                    M2=M2+block_M2+difference.square()*(count*n/combined);mean=mean+difference*(n/combined);count=combined
                del q,delta,block_mean,block_M2
        variance=M2/(count-1)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all() and (variance>=0).all()):raise ValueError('Invalid streamed prior moments')
        mean_q=(rest+mean).cpu().numpy();var=variance.cpu().numpy();MCSE=np.sqrt(var/count)
        torch.cuda.synchronize(0);timings[program]=time.monotonic()-begin
        destination=out/('prior-predictive-'+program+'.npz')
        np.savez_compressed(destination,mean_positions_m=mean_q,material_prior_std_m=np.sqrt(var),
            material_prior_variance_m2=var,mean_MCSE_m=MCSE,times_s=np.arange(61)/20)
        saved[program]=(mean_q,var,MCSE,destination);print(json.dumps({'program':program,'draws':count,'forward_and_moment_s':timings[program]}),flush=True)
    # Allpriorpredictionmoments complete before any finite/posteriorpredictive
    # array is opened. Those arrays cannot affect draws, model orcommands.
    rows=[]
    for record in design['physics_cases']:
        p=read(checked(record));program=p['program_id'];directory=resolve(p['output'])
        execution=read(directory/'execution.json');summary=read(directory/'summary.json')
        if not (execution['terminal'] and execution['completed']) or sha(directory/'trajectory.npz')!=summary['trajectory_sha256']:raise ValueError('Finite response incomplete/changed')
        with np.load(directory/'trajectory.npz',allow_pickle=False) as f:truth=f['positions_m'].copy();times=f['times_s'].copy()
        posterior_path=reference/('posterior-predictive-'+program+'.npz')
        with np.load(posterior_path,allow_pickle=False) as f:conditioned=f['mean_positions_m'].copy();conditional_var=f['material_posterior_std_m']**2;ptimes=f['times_s'].copy()
        if not np.array_equal(times,np.arange(61)/20) or not np.array_equal(times,ptimes):raise ValueError('Comparison clocks differ')
        mean,var,MCSE,path=saved[program]
        rows.append({'program':program,'all61':moments(mean,var,truth,conditioned,conditional_var,MCSE,slice(None)),
            'after1s':moments(mean,var,truth,conditioned,conditional_var,MCSE,times>1),
            'acquired_prefix_0to1s':moments(mean,var,truth,conditioned,conditional_var,MCSE,times<=1),
            'prior_predictive_sha256':sha(path),'fixed_conditioned_predictive_sha256':sha(posterior_path),
            'finite_trajectory_sha256':sha(directory/'trajectory.npz')})
    for path,digest in sources.items():
        if sha(resolve(path))!=digest:raise ValueError('Frozen baseline source changed')
    report={'status':'COMPLETED_POSTHOC_PRIOR_VS_CONDITIONED_PREDICTIVE_DIAGNOSTIC','draws':8192,'seed':20261105,
        'rows':rows,'timings_s':timings,'elapsed_s':time.monotonic()-started,'protocol_sha256':sha(out/'protocol.json'),
        'physical_prior_draws_sha256':hashlib.sha256(physical.tobytes()).hexdigest(),'source_sha256':sources,
        'posterior_fit_or_reference_rerun':False,'new_finite_runs':0,'coverage_claim':False,
        'limitation':'Explicitlyposthoc one known-support truth. Material-only latent variance excludes observation noise, forward-modeldiscrepancy and numerical-reference error. CoordinateMCSE quantifiesprior-MonteCarlomean error, not acoverageguarantee. NoGaussian90%coverageclaimoractivepolicybenefit.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);a=parser.parse_args();run(a.output)
