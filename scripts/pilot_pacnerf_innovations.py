"""Observed-data-only development pilot for exact innovation coordinates.

No future geometry, reconstruction score, or model ranking is computed. The
reference is the standard Gaussian innovation prior, without a fitted metric.
This pilot cannot replace any production result.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import checkpoint_pacnerf as checkpoints
from pacnerf_comparison import cases,make_model,Target,write
from pacnerf_innovations import InnovationModel
from physgauss.cli import source_fingerprint
from physgauss.likelihood_audit import _environment
from physgauss.split_hmc import SplitHMCConfig
from physgauss.chain_diagnostics import diagnose_quantities


def run(inputs,output,index):
    if source_fingerprint()!=checkpoints.FINGERPRINT:raise ValueError('Frozen physical model changed')
    scene,views,noise,force=cases()[index]
    name=f'{scene}__{views}views__{noise}__{force}'
    out=Path(output)/name;out.mkdir(parents=True,exist_ok=True)
    model,_=make_model(inputs,index);innovation=InnovationModel(model);d=model.dimension
    input_hash=hashlib.sha256((Path(inputs)/f'{scene}-inputs.pt').read_bytes()).hexdigest()
    manifest=json.loads((Path(inputs)/f'{scene}-manifest.json').read_text())
    if input_hash!=manifest['inputs_sha256']:raise ValueError('Prepared input hash differs')
    cfg=SplitHMCConfig(warmup=100,draws=100,min_time=1.,max_time=4.,target_accept=.9,seed=250001+index)
    settings={'case_index':index,'case':name,'source_fingerprint':source_fingerprint(),
        'environment':_environment(),'configuration':asdict(cfg),
        'input_sha256':input_hash,
        'inference_files_sha256_lf':{f:hashlib.sha256((Path(__file__).parent/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for f in ['pilot_pacnerf_innovations.py','pacnerf_innovations.py','checkpoint_pacnerf.py','pacnerf_comparison.py']},
        'reference':'standard Gaussian innovation prior; no fitted mean/metric; prior-dispersed initial chains',
        'development_only':True,'future_geometry_used':False,'valid_for_ranking':False}
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text())!=settings:
        raise ValueError('Pilot inputs/settings changed; choose a fresh directory')
    write(out/'settings.json',settings)
    target=Target(innovation,torch.zeros(d,dtype=torch.float64),torch.eye(d,dtype=torch.float64))
    chains=torch.load(out/'chains.pt',weights_only=True) if (out/'chains.pt').exists() else None
    count=0 if chains is None else chains['white'].shape[1]
    began=time.time()
    if count<cfg.draws:
        call=cfg if chains is None else replace(cfg,warmup=0,draws=cfg.draws-count,step_size=chains['step_size'])
        chains=checkpoints.sample_in_chunks(target,d,call,out,50,resume=chains,
            progress=lambda record:print(name,record,flush=True))
    sampling_seconds=time.time()-began
    z=chains['white'].flatten(0,1)
    with torch.no_grad():
        physical=torch.cat([innovation.to_physical(batch) for batch in z.split(32)])
        conditional=[model.force_conditionals(batch) for batch in physical.split(32)]
        fm,_,normalizers=[torch.cat([row[i] for row in conditional]) for i in range(3)]
        selected=torch.linspace(0,len(z)-1,16).round().long()
        probe=z[selected];old,jac=innovation.transform(probe)
        density_error=float((model.objective(old)-jac-innovation.objective(probe)).abs().max())
        replay,bad=target.log_ratio_no_grad(probe)
        cache_error=float((replay-chains['log_ratio'].flatten()[selected]).abs().max())
        if bool(bad.any()) or cache_error>1e-6 or density_error>1e-5:
            raise ValueError('Pilot density or cached target failed replay')
    n=chains['white'].shape[1]
    log_target=chains['log_ratio']-.5*chains['white'].square().sum(-1)-target.center
    monitor=torch.cat((physical.reshape(4,n,d),chains['white'],log_target[...,None],
        fm.reshape(4,n,24),normalizers.reshape(4,n,2)),-1)
    labels=([f'physical_z{i}' for i in range(d)]+[f'innovation_z{i}' for i in range(d)]+['log_target']+
        [f'conditional_force_mean{i}' for i in range(24)]+['logdet_Q','logdet_A'])
    diagnostics=diagnose_quantities(monitor,labels);write(out/'diagnostics.json',diagnostics)
    summary={'settings':settings,'github_run_id':os.environ.get('GITHUB_RUN_ID'),
        'development_only':True,'valid_for_ranking':False,'draws_per_chain':n,
        'step_size':chains['step_size'],'counts':chains['counts'],
        'gradient_batches':chains['gradient_evaluations'],'sampling_seconds_this_invocation':sampling_seconds,
        'maximum_cached_log_ratio_error':cache_error,'maximum_transformed_density_error':density_error,
        'worst_rhat':max(v['rank_folded_split_rhat'] or 999 for v in diagnostics['quantities'].values()),
        'minimum_bulk_ess':min(v['bulk_ess'] or 0 for v in diagnostics['quantities'].values()),
        'qualification':'100-draw observed-data-only feasibility pilot; no reconstruction/coverage evaluation; not a production comparison or convergence claim'}
    write(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',default='runs/pacnerf-inputs')
    p.add_argument('--output',default='runs/pacnerf-innovation-pilot');p.add_argument('--case-index',type=int,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    run(a.inputs,a.output,a.case_index)
