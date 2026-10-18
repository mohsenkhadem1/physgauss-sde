"""Fresh four-arm hierarchical uncertainty comparison with recoverable sampling.

Pilot is observed-data-only. Final evaluation occurs only after saved chains,
cached-target replay, physical/innovation density replay and diagnostics.
No old-target checkpoint can be resumed under this model.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import pacnerf_innovation_metric as fitting
from pacnerf_discrepancy import make_model,make_target,DiscrepancyInnovation,gauss_newton
from pacnerf_comparison import cases,score,write
from physgauss.checkpoint_hmc import sample_in_chunks
from physgauss.split_hmc import SplitHMCConfig
from physgauss.chain_diagnostics import diagnose_quantities
from physgauss.cli import source_fingerprint


def file_hash(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


@torch.no_grad()
def replay(model,target,chains):
    white=chains['white'].flatten(0,1)
    ids=torch.linspace(0,len(white)-1,16).round().long()
    w=white[ids];native=target.coordinates(w)
    physical,jac=target.model.transform(native)
    density_error=float((model.objective(physical)-jac-target.model.objective(native)).abs().max())
    value,bad=target.log_ratio_no_grad(w)
    cache_error=float((value-chains['log_ratio'].flatten()[ids]).abs().max())
    if bool(bad.any()) or density_error>1e-5 or cache_error>1e-6:
        raise ValueError(f'Posterior density replay failed: {density_error}, {cache_error}')
    return {'maximum_physical_density_identity_error':density_error,
        'maximum_cached_target_error':cache_error,'probes':len(w),'all_pass':True}


@torch.no_grad()
def uncertainty_scores(model,z,inputs):
    # This is the first access to future positions in this worker.
    truth=torch.load(Path(inputs)/f"{model.data['scene']}-evaluation.pt",weights_only=True)['particles']
    q,_=model.decode(z)
    ids=torch.linspace(0,len(q)-1,min(2000,len(q))).round().long()
    gap=list(range(4,10)) if len(model.frames)==4 else list(range(5,9))
    x=model.data['gaussians'][None,None]+torch.einsum('btm,gcm->btgc',q[ids][:,gap],model.renderer.gaussian_basis)
    y=truth[gap]
    coverage=[]
    for level in [.5,.8,.9,.95]:
        lo,hi=torch.quantile(x,x.new_tensor([(1-level)/2,(1+level)/2]),dim=0)
        width=hi-lo
        interval=width+2/(1-level)*((lo-y).clamp_min(0)+(y-hi).clamp_min(0))
        coverage.append({'nominal':level,'coverage':float(((y>=lo)&(y<=hi)).double().mean()),
            'coverage_xyz':((y>=lo)&(y<=hi)).double().mean((0,1)).tolist(),
            'mean_width_m':float(width.mean()),'mean_interval_score_m':float(interval.mean())})
    ordered=x.sort(0).values
    weights=(2*torch.arange(1,len(x)+1,dtype=x.dtype)-len(x)-1).reshape(-1,1,1,1)
    crps=(x-y).abs().mean(0)-(weights*ordered).sum(0)/(len(x)**2)
    return {'coverage_curve':coverage,'mean_crps_m':float(crps.mean()),'interval_draws':len(x),
        'qualification':'descriptive coverage over two deterministic simulations; correlated coordinates do not count as independent calibration datasets; conditional on supplied geometry, fixed appearance and modal space'}


def run(inputs,output,index,pilot=False,chunk=50,sampler='elliptical'):
    began=time.time();scene,views,noise,force=cases()[index]
    name=f'{scene}__{views}views__{noise}__{force}';out=Path(output)/name
    out.mkdir(parents=True,exist_ok=True)
    model,bundle=make_model(inputs,index)
    settings={'case_index':index,'case':name,'source_fingerprint':source_fingerprint(),'pilot':pilot,
        'input_sha256':hashlib.sha256((Path(inputs)/f'{scene}-inputs.pt').read_bytes()).hexdigest(),
        'inference_files_sha256_lf':{f:file_hash(Path(__file__).parent/f) for f in [
            'pacnerf_discrepancy.py','compare_pacnerf_discrepancy.py','pacnerf_innovations.py',
            'pacnerf_innovation_metric.py','pacnerf_comparison.py','pacnerf_elliptical.py']},
        'model':model.specification(),'observed_pairs':model.specs,
        'sampler':sampler,
        'sampling':'four chains; burnin 100/500 pilot/full; retained 100 pilot or 2000/4000/8000 full; exact fixed-reference target; same rank-Rhat/ESS thresholds',
        'new_target':'Fresh starts; no legacy samples pooled or resumed under this model',
        'source_of_prior_scales':'gravity, frame duration, supplied initial geometry; no evaluation-based tuning'}
    settings=json.loads(json.dumps(settings))
    settings_path=out/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        raise ValueError('Target or protocol changed; use a fresh output folder')
    write(settings_path,settings)
    invocations=json.loads((out/'invocations.json').read_text()) if (out/'invocations.json').exists() else []
    invocations.append({'run_id':os.environ.get('GITHUB_RUN_ID'),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    write(out/'invocations.json',invocations)
    if (out/'reference.pt').exists():
        reference=torch.load(out/'reference.pt',weights_only=True)
    else:
        original=fitting.gauss_newton;fitting.gauss_newton=gauss_newton
        try:mean,precision,stationarity,info=fitting.fit_reference(DiscrepancyInnovation(model),checkpoint=out)
        finally:fitting.gauss_newton=original
        reference={'mean':mean,'precision':precision,'stationarity':stationarity,'fit':info,
                   'coordinates':'hierarchical-discrepancy-innovation-v1'}
        torch.save(reference,out/'reference.pt')
        write(out/'fit.json',{k:v for k,v in reference.items() if k not in ['mean','precision']})
    if not reference['stationarity']['stationary']:raise ValueError('Unfinished reference fit')
    target=make_target(model,reference)
    config=SplitHMCConfig(warmup=100 if pilot else 500,draws=100 if pilot else 2000,
        min_time=1.,max_time=4.,target_accept=.9,seed=240001+index)
    chains=torch.load(out/'chains.pt',weights_only=True) if (out/'chains.pt').exists() else None
    history=[]
    for budget in ([100] if pilot else [500,1000,2000,4000,8000]):
        if chains is not None and chains['white'].shape[1]>budget:continue
        count=budget-(0 if chains is None else chains['white'].shape[1])
        if count and sampler=='elliptical':
            from pacnerf_elliptical import sample
            chains=sample(target.log_ratio_no_grad,model.dimension,out,budget,
                warmup=100 if pilot else 500,seed=240001+index,chunk=chunk,
                progress=lambda row:print(name,row,flush=True))
        elif count:
            cfg=replace(config,draws=count,warmup=config.warmup if chains is None else 0,
                step_size=config.step_size if chains is None else chains['step_size'])
            chains=sample_in_chunks(target,model.dimension,cfg,out,chunk,resume=chains,
                progress=lambda row:print(name,row,flush=True))
        verification=replay(model,target,chains);write(out/'replay.json',verification)
        white=chains['white'].flatten(0,1)
        z=target.decode(white)
        with torch.no_grad():
            fm=torch.cat([model.force_conditionals(batch)[0] for batch in z.split(32)])
        log_target=chains['log_ratio']-.5*chains['white'].square().sum(-1)-target.center
        native=target.coordinates(white).reshape(4,budget,-1)
        monitored=torch.cat((z.reshape(4,budget,-1),native,log_target[...,None],fm.reshape(4,budget,-1)),-1)
        diagnostics=diagnose_quantities(monitored,
            [f'physical_{i}' for i in range(model.dimension)]+[f'innovation_{i}' for i in range(model.dimension)]
            +['log_target']+[f'conditional_load_{i}' for i in range(model.force_dimension)])
        write(out/'diagnostics.json',diagnostics)
        passed=diagnostics['all_monitored_quantities_pass']
        history.append({'draws_per_chain':budget,'all_pass':passed})
        if pilot or (passed and budget>=2000):break
    counts=chains['counts']['retained']
    summary={'case':name,'settings':settings,'draws_per_chain':budget,'sampling_history':history,
        'valid_for_ranking':bool(passed and not pilot and budget>=2000),'pilot':pilot,
        'worst_rhat':max(v['rank_folded_split_rhat'] or 999 for v in diagnostics['quantities'].values()),
        'minimum_bulk_ess':min(v['bulk_ess'] or 0 for v in diagnostics['quantities'].values()),
        'minimum_tail_ess':min(v['tail_ess'] or 0 for v in diagnostics['quantities'].values()),
        'retained_acceptance':[v/budget for v in counts['accepted']],
        'retained_invalid':counts['invalid'],'retained_divergent':counts['divergent'],
        'step_size':chains.get('step_size'),'gradient_batches':chains.get('gradient_evaluations',0),
        'target_evaluations':chains.get('target_evaluations'),
        'sampler':sampler,
        'acceptance_qualification':'Elliptical slice always closes its bracket; movement rates are not Metropolis acceptance probabilities. Inspect ESS/Rhat and bracket effort.' if sampler=='elliptical' else 'Metropolis acceptance',
        'replay':verification,'elapsed_this_invocation_s':time.time()-began}
    if pilot:
        summary['future_geometry_used']=False;write(out/'pilot_summary.json',summary);return summary
    with torch.no_grad():
        scores,posterior=score(model,z,inputs)
        uncertainty=uncertainty_scores(model,z,inputs)
        h=z[:,-3:]
        scales=torch.stack(model.scales(z),-1)
        posterior['hyperparameter_names']=['log_added_diffusion','log_smooth_modal_load','renderer_error_hyperparameter']
        posterior['hyperparameter_quantiles']=torch.quantile(h,h.new_tensor([.05,.5,.95]),dim=0).tolist()
        posterior['scale_names']=['added_modal_acceleration_diffusion','smooth_discrepancy_multiplier','rgb_sigma']
        posterior['scale_quantiles']=torch.quantile(scales,scales.new_tensor([.05,.5,.95]),dim=0).tolist()
        q,_=model.decode(z)
        q_chain=q.reshape(4,budget,14,model.modes)
        length=max(20,int(budget**.5));blocks=budget//length
        batch_q=q_chain[:,:blocks*length].reshape(4,blocks,length,14,model.modes).mean(2).flatten(0,1)
        posterior['modal_mean_batch_mcse_m']=(batch_q.std(0)/math.sqrt(len(batch_q))).tolist()
        h_chain=h.reshape(4,budget,3)
        batch_h=h_chain[:,:blocks*length].reshape(4,blocks,length,3).mean(2).flatten(0,1)
        posterior['hyperparameter_mean']=h.mean(0).tolist()
        posterior['hyperparameter_mean_batch_mcse']=(batch_h.std(0)/math.sqrt(len(batch_h))).tolist()
        posterior['mcse_batch_length']=length;posterior['mcse_batches']=len(batch_q)
        # Full law of total covariance, including scale and trajectory uncertainty.
        force_records=[model.force_conditionals(batch) for batch in z.split(32)]
        means=torch.cat([r[0] for r in force_records]);covariances=torch.cat([r[1] for r in force_records])
        within=covariances.mean(0);between=torch.cov(means.T)
        posterior['force']={'coefficient_mean':means.mean(0).tolist(),'coefficient_covariance':(within+between).tolist(),
            'mean_conditional_covariance':within.tolist(),'between_trajectory_covariance':between.tolist(),
            'definition':'whitened coefficients of original loads plus smooth modal discrepancy; learned amplitude makes physical-load distribution a non-Gaussian mixture; no force ground truth'}
        if force=='joint':
            explicit=z[:,model.base_dimension:model.core_dimension]
            posterior['force']['explicit_sample_coefficient_mean']=explicit.mean(0).tolist()
            posterior['force']['explicit_sample_coefficient_covariance']=torch.cov(explicit.T).tolist()
        # Independent NumPy position-mean replay, not the renderer's implementation.
        q,_=model.decode(z)
        independent=model.data['gaussians'].numpy()[None]+np.einsum('tm,gcm->tgc',q.numpy().mean(0),model.renderer.gaussian_basis.numpy())
        mean_error=float(np.max(np.abs(independent-np.asarray(posterior['particle_mean_m']))))
        if mean_error>1e-10:raise ValueError('Independent position mean replay failed')
        summary.update({'posterior_scores':scores,'uncertainty_scores':uncertainty,
            'independent_position_mean_replay_error':mean_error,
            'scope':'uncertainty in sampled paths, mechanics, added diffusion, smooth loads and RGB scale; fixed modal subspace and appearance remain assumptions'})
    write(out/'posterior.json',posterior);write(out/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    p.add_argument('--case-index',type=int,required=True);p.add_argument('--pilot',action='store_true');p.add_argument('--chunk-size',type=int,default=50)
    p.add_argument('--sampler',choices=['elliptical','hmc'],default='elliptical')
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    run(a.inputs,a.output,a.case_index,a.pilot,a.chunk_size,a.sampler)
