"""Audit every planned PAC attempt, including time-limited incomplete chains.

Does not score reconstruction or relax minimum draws/convergence gates. Future
geometry is never loaded. Cached target replay checks numerical provenance only.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import torch
from pacnerf_comparison import cases,make_model,Target,make_target,write,stationarity_record
from physgauss.cli import source_fingerprint


def find(root,name,filename):
    paths=list(Path(root).glob(f'**/{name}/{filename}'))
    if len(paths)>1:raise ValueError('Ambiguous attempt artifact')
    return paths[0] if paths else None


@torch.no_grad()
def inspect(raw,inputs):
    raw=Path(raw);inputs=Path(inputs)
    status=json.loads((raw/'workflow_status.json').read_text())
    jobs={int(m[1]):job for job in status['jobs']
        if (m:=re.fullmatch(r'compare \((\d+)\)',job['name']))}
    rows=[];input_hashes={}
    for scene in ['torus','bird']:
        actual=hashlib.sha256((inputs/f'{scene}-inputs.pt').read_bytes()).hexdigest()
        expected=json.loads((inputs/f'{scene}-manifest.json').read_text())['inputs_sha256']
        if actual!=expected:raise ValueError('Prepared inputs differ')
        input_hashes[scene]=actual
    for index,(scene,views,noise,force) in enumerate(cases()):
        name=f'{scene}__{views}views__{noise}__{force}';job=jobs.get(index)
        result=raw/f'results-{index}';chain_root=raw/f'chains-{index}'
        settings_path=find(result,name,'settings.json')
        summary_path=find(result,name,'summary.json')
        fit_path=find(result,name,'fit.json')
        diagnostics_path=find(result,name,'diagnostics.json')
        reference_path=find(chain_root,name,'reference.pt')
        chain_path=find(chain_root,name,'chains.pt')
        row={'case_index':index,'case':name,'job':job,
            'attempted':bool(settings_path or (job and job['status']!='queued')),
            'has_final_worker_summary':bool(summary_path),'has_saved_reference':bool(reference_path),
            'has_saved_retained_chain':bool(chain_path),'retained_draws_per_chain':None,
            'finite_chain_checks_pass':False,'valid_for_ranking':False}
        if settings_path:
            settings=json.loads(settings_path.read_text())
            if settings['source_fingerprint']!=source_fingerprint() or settings['pilot']:
                raise ValueError('Attempt is not the frozen production model')
            if settings['input_sha256']!=input_hashes[scene]:raise ValueError('Attempt input differs')
            row['settings_input_source_checks_pass']=True
        if fit_path:
            stationarity=json.loads(fit_path.read_text())['stationarity']
            row['mode_stationarity']=dict(stationarity_record(stationarity),stationary=True)
        if chain_path:
            if reference_path is None or settings_path is None:raise ValueError('Chain lacks reference/settings')
            chains=torch.load(chain_path,weights_only=True)
            reference=torch.load(reference_path,weights_only=True)
            white=chains['white'];c,n,d=white.shape
            model,_=make_model(inputs,index)
            if c!=4 or d!=model.dimension or n<16:raise ValueError('Unexpected chain layout')
            target=make_target(model,reference)
            selected=torch.linspace(0,n-1,min(4,n)).round().long()
            replay,bad=target.log_ratio_no_grad(white[:,selected].reshape(-1,d))
            cache_error=float((replay-chains['log_ratio'][:,selected].flatten()).abs().max())
            if bool(bad.any()) or cache_error>1e-6:raise ValueError('Saved target cache differs')
            row.update({'retained_draws_per_chain':n,'minimum_retained_budget_reached':n>=2000,
                'step_size':chains['step_size'],'gradient_batches':chains['gradient_evaluations'],
                'counts':chains['counts'],'maximum_cached_log_ratio_replay_error':cache_error,
                'raw_chains_sha256':hashlib.sha256(chain_path.read_bytes()).hexdigest(),
                'cache_replay_pass':True})
        if diagnostics_path:
            diagnostic=json.loads(diagnostics_path.read_text())
            row['last_saved_diagnostics']={
                'all_monitored_quantities_pass':diagnostic['all_monitored_quantities_pass'],
                'worst_rhat':max(v['rank_folded_split_rhat'] or 999 for v in diagnostic['quantities'].values()),
                'minimum_bulk_ess':min(v['bulk_ess'] or 0 for v in diagnostic['quantities'].values()),
                'minimum_tail_ess':min(v['tail_ess'] or 0 for v in diagnostic['quantities'].values()),
                'qualification':'may precede the newest saved retained prefix; final summary required for eligibility'}
        if summary_path:
            summary=json.loads(summary_path.read_text())
            if summary['valid_for_ranking'] and (row['retained_draws_per_chain'] is None
                or row['retained_draws_per_chain']<2000 or diagnostics_path is None
                or not row['last_saved_diagnostics']['all_monitored_quantities_pass']):
                raise ValueError('Eligibility contradicts saved chain budget/diagnostics')
            row['finite_chain_checks_pass']=summary['valid_for_ranking']
            row['valid_for_ranking']=summary['valid_for_ranking']
            row['status']='finite_checks_pass' if summary['valid_for_ranking'] else 'finite_checks_failed'
        else:
            row['status']='incomplete_inference' if job and job['status']=='completed' else 'running' if row['attempted'] else 'not_started'
        for step in (job or {}).get('compute_steps',[]):
            if step.get('started_at') and step.get('completed_at'):
                start=datetime.fromisoformat(step['started_at'].replace('Z','+00:00'))
                end=datetime.fromisoformat(step['completed_at'].replace('Z','+00:00'))
                row['compute_step_wall_seconds']=(end-start).total_seconds()
        rows.append(row);print(json.dumps({k:row.get(k) for k in ['case_index','status','retained_draws_per_chain','step_size','gradient_batches','compute_step_wall_seconds']}),flush=True)
    return {'run_id':status['run_id'],'source_commit':status['source_commit'],
        'source_fingerprint':source_fingerprint(),'workflow_status':status['status'],
        'attempted_cases':sum(r['attempted'] for r in rows),
        'final_worker_summaries':sum(r['has_final_worker_summary'] for r in rows),
        'finite_chain_pass_cases':sum(r['finite_chain_checks_pass'] for r in rows),'cases':rows,
        'qualification':'attempt/checkpoint provenance audit, not reconstruction scoring or a relaxed convergence criterion; an incomplete run is not a valid posterior; wall seconds include fitting/sampling within this attempt, and exclude installation and other attempts'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True)
    p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    write(a.output,inspect(a.raw,a.inputs))
