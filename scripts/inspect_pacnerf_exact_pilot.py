"""Validate the observed-only four-arm pilot without future geometry/scoring."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import torch
from pacnerf_comparison import cases,make_model,make_target,stationarity_record,write
from physgauss.cli import source_fingerprint


@torch.no_grad()
def inspect(raw,inputs,output):
    raw,inputs,output=map(Path,(raw,inputs,output))
    if output.exists():raise ValueError('Choose an unused publication directory')
    output.mkdir(parents=True)
    status=json.loads((raw/'workflow_status.json').read_text());rows=[]
    for index in [8,9,10,11]:
        scene,views,noise,force=cases()[index];name=f'{scene}__{views}views__{noise}__{force}'
        result=raw/f'results-{index}'/name;folder=raw/f'chains-{index}'/name
        row={'case':name,'case_index':index,'feasibility_pass':False,'valid_for_ranking':False}
        if not (result/'pilot_summary.json').exists():
            row['status']='pilot incomplete or failed';rows.append(row);continue
        summary=json.loads((result/'pilot_summary.json').read_text());settings=summary['settings']
        if not settings['pilot'] or settings.get('coordinates')!='innovation':raise ValueError('Not the exact-innovation development pilot')
        if settings['source_fingerprint']!=source_fingerprint():raise ValueError('Physical source differs')
        if settings['worker_commit']!='219665b0776d9a1b5d092176331e21f724499bca':raise ValueError('Unexpected worker commit')
        if hashlib.sha256((inputs/f'{scene}-inputs.pt').read_bytes()).hexdigest()!=settings['input_sha256']:
            raise ValueError('Prepared inputs differ')
        model,_=make_model(inputs,index)
        reference=torch.load(folder/'reference.pt',weights_only=True)
        stationarity=stationarity_record(reference['stationarity'])
        chains=torch.load(folder/'chains.pt',weights_only=True)
        white=chains['white'];c,n,d=white.shape
        if (c,n,d)!=(4,100,model.dimension):raise ValueError('Unexpected development chain layout')
        target=make_target(model,reference)
        ids=torch.linspace(0,n-1,4).round().long();probe=white[:,ids].reshape(-1,d)
        replay,bad=target.log_ratio_no_grad(probe)
        error=float((replay-chains['log_ratio'][:,ids].flatten()).abs().max())
        native=target.coordinates(probe);physical,jac=target.model.transform(native)
        density_error=float((model.objective(physical)-jac-target.model.objective(native)).abs().max())
        if bool(bad.any()) or error>1e-6 or density_error>1e-5:raise ValueError('Pilot raw target replay failed')
        counts=chains['counts']['retained'];acceptance=[v/n for v in counts['accepted']]
        diagnostic=json.loads((result/'diagnostics.json').read_text())
        passed=min(acceptance)>=.5 and max(counts['invalid'])==0 and max(counts['divergent'])==0
        row.update({'status':'short feasibility check only','feasibility_pass':passed,'draws_per_chain':n,
            'stationarity':stationarity,'step_size':chains['step_size'],'retained_acceptance':acceptance,
            'retained_invalid':counts['invalid'],'retained_divergent':counts['divergent'],
            'maximum_cached_log_ratio_error':error,'maximum_density_identity_error':density_error,
            'worst_rhat':max(v['rank_folded_split_rhat'] or 999 for v in diagnostic['quantities'].values()),
            'minimum_bulk_ess':min(v['bulk_ess'] or 0 for v in diagnostic['quantities'].values()),
            'gradient_batches':chains['gradient_evaluations'],'seconds':summary['seconds']})
        rows.append(row)
        destination=output/'cases'/name;destination.mkdir(parents=True)
        for filename in ['pilot_summary.json','settings.json','fit.json','diagnostics.json','coordinate_replay.json','checkpoint_worker.json']:
            shutil.copyfile(result/filename,destination/filename)
        for kind in ['chains','results']:
            marker=raw/f'{kind}-{index}'/'artifact_download.json'
            shutil.copyfile(marker,destination/f'{kind}-artifact.json')
        print(json.dumps(row),flush=True)
    summary={'run_id':status['run_id'],'source_fingerprint':source_fingerprint(),
        'worker_commit':'219665b0776d9a1b5d092176331e21f724499bca',
        'all_four_feasibility_checks_pass':all(r['feasibility_pass'] for r in rows),'cases':rows,
        'valid_for_ranking':False,'future_geometry_used':False,
        'qualification':'Four 100-warmup/100-retained-chain development checks; feasibility only. No reconstruction, coverage, force accuracy or convergence claim. Full study starts fresh and requires its stricter predeclared diagnostics.'}
    write(output/'summary.json',summary);write(output/'workflow-status.json',status)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['case'],r['status'],r.get('retained_acceptance'),r.get('retained_divergent'),r['feasibility_pass'],r.get('step_size'),r.get('worst_rhat'),r.get('minimum_bulk_ess')])+'</tr>' for r in rows)
    page='''<!doctype html><html><meta charset="utf-8"><title>PAC-NeRF inference development</title><body style="font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px"><h1>Exact-innovation four-arm pilot</h1><p><strong>Development only: no reconstruction or uncertainty ranking.</strong></p><p>Bird, three later images plus frame zero. Supplied initial geometry and eleven initial appearance views. Exact linear covariance/load integrals; constant/neural noise crossed with marginalized/explicit loads. A fixed full-RGB Gauss-Newton metric assists proposals; the nonlinear target remains exact.</p><p>All four feasibility checks pass: '''+str(summary['all_four_feasibility_checks_pass'])+'''</p><table cellpadding="8"><tr><th>Case</th><th>Status</th><th>Acceptance per chain</th><th>Large-energy proposals</th><th>Conservative screen</th><th>Step</th><th>Worst Rhat</th><th>Min bulk ESS</th></tr>'''+table+'''</table><p>The conservative screen requires at least 50% acceptance in each chain and zero invalid or large-energy proposals; failed screens are preserved. Short-chain diagnostics are shown to expose limitations. Passing this pilot does not establish posterior convergence. No held-out geometry was read or scored.</p><p><a href="summary.json">Raw verification summary</a> · <a href="https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/'''+str(status['run_id'])+'''">Actions run</a></p></body></html>'''
    (output/'report.html').write_text(page,encoding='utf-8')
    for path in output.rglob('*'):
        if path.is_file() and path.suffix in ['.json','.html','.csv','.md']:
            path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    hashes={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
    write(output/'publication.json',{'files_sha256':hashes})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    result=inspect(a.raw,a.inputs,a.output)
    print(json.dumps({'all_four_feasibility_checks_pass':result['all_four_feasibility_checks_pass']}))
