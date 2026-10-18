"""Replay saved warmup states of the bounded observed-only pilot, not posteriors."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from pacnerf_comparison import cases,make_model,Target,write
from pacnerf_innovations import InnovationModel
from physgauss.cli import source_fingerprint


@torch.no_grad()
def inspect(raw,inputs):
    raw=Path(raw);status=json.loads((raw/'workflow_status.json').read_text());rows=[]
    checks=json.loads((raw/'timeout-log-checks.json').read_text())
    if checks['run_id']!=status['run_id'] or len(checks['cases'])!=4:
        raise ValueError('Completed log checks do not match this pilot')
    for check in checks['cases']:
        jobs=[j for j in status['jobs'] if j['id']==check['job_id'] and j['name']==f"pilot ({check['index']})"]
        if not check['confirmed_25_minute_timeout'] or len(jobs)!=1 or jobs[0]['conclusion']!='failure':
            raise ValueError('Pilot timeout is not confirmed')
    for index in range(4):
        scene,views,noise,force=cases()[index];name=f'{scene}__{views}views__{noise}__{force}'
        folder=raw/f'case-{index}'/name
        settings=json.loads((folder/'settings.json').read_text())
        actual_hash=hashlib.sha256((Path(inputs)/f'{scene}-inputs.pt').read_bytes()).hexdigest()
        if settings['source_fingerprint']!=source_fingerprint() or settings['input_sha256']!=actual_hash:
            raise ValueError('Pilot source/input differs')
        if not settings['development_only'] or settings['future_geometry_used']:raise ValueError('Unexpected pilot plan')
        if (folder/'chains.pt').exists():raise ValueError('This audit expects the observed warmup-only failures')
        state=torch.load(folder/'warmup_state.pt',weights_only=True)
        model,_=make_model(inputs,index);innovation=InnovationModel(model);d=model.dimension
        target=Target(innovation,torch.zeros(d,dtype=torch.float64),torch.eye(d,dtype=torch.float64))
        z=state['position'];replay,bad=target.log_ratio_no_grad(z)
        cache_error=float((replay-state['log_ratio']).abs().max())
        physical,jac=innovation.transform(z)
        density_error=float((model.objective(physical)-jac-innovation.objective(z)).abs().max())
        if bool(bad.any()) or cache_error>1e-6 or density_error>1e-5:raise ValueError('Pilot warmup target replay differs')
        rows.append({'case_index':index,'case':name,'phase':'warmup_only','retained_draws':0,
            'last_recoverable_warmup_transition':state['next_iteration'],
            'saved_adapting_step_size':state['epsilon'],
            'gradient_batches_at_last_saved_state':state['gradient_evaluations'],
            'saved_warmup_counts':state['counts']['warmup'],
            'maximum_cached_target_error':cache_error,'maximum_density_identity_error':density_error,
            'checkpoint_sha256':hashlib.sha256((folder/'warmup_state.pt').read_bytes()).hexdigest(),
            'artifact':json.loads((folder.parent/'artifact_download.json').read_text()),
            'valid_for_ranking':False})
    return {'run_id':status['run_id'],'source_commit':status['source_commit'],
        'source_fingerprint':source_fingerprint(),'development_only':True,'valid_for_ranking':False,
        'completed_job_log_checks':checks,'cases':rows,'qualification':'all four jobs reached the predeclared 25-minute compute limit during warmup (confirmed in completed job logs); no retained posterior, reconstruction or coverage estimate; counters are recoverable checkpoint prefixes, excluding work since the last save; failure concerns this prior-reference/initialization/kernel setup, not all innovation-based inference'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True)
    p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    write(a.output,inspect(a.raw,a.inputs))
