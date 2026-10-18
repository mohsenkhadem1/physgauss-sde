"""Verify an Actions batch, publish compact results and optionally commit them.

Raw checkpoints never enter Git. Publication uses a separate disposable
worktree, refuses an existing report, and never force-pushes. Incomplete or
failed diagnostics remain visible and cannot produce a scientific ranking.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from physgauss.cli import source_fingerprint
from pacnerf_comparison import write
from inspect_pacnerf_attempts import inspect
from verify_pacnerf import verify_case
from audit_pacnerf_posterior_transition import audit as transition_audit
from audit_pacnerf_innovations import audit as innovation_audit
from audit_pacnerf_renderer_reference import audit as renderer_audit
from audit_pacnerf_chain_traces import audit as trace_audit
from audit_pacnerf_noise_transfer import audit as noise_audit
from publish_pacnerf import publish


def git(*args,cwd=None):
    return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()


def finalize(raw,inputs,output,push=False):
    raw,inputs,output=map(Path,(raw,inputs,output))
    if not (raw/'workflow_status.json').exists():raise ValueError('Workflow status is required')
    # Reporting may advance independently, but inference modules must remain
    # byte-identical to the pinned worker used for these exact chains.
    for name in ['pacnerf_comparison.py','pacnerf_innovations.py','pacnerf_innovation_metric.py','checkpoint_pacnerf.py']:
        actual=(Path(__file__).parent/name).read_bytes().replace(b'\r\n',b'\n')
        expected=subprocess.check_output(['git','show','219665b:scripts/'+name])
        if actual!=expected:raise ValueError('Frozen inference module changed: '+name)
    for path in raw.glob('results-*/**/settings.json'):
        settings=json.loads(path.read_text())
        if settings.get('worker_commit')!='219665b0776d9a1b5d092176331e21f724499bca' or settings.get('coordinates')!='innovation':
            raise ValueError('Attempt does not use the pinned exact-innovation worker')
        for name,digest in settings['inference_files_sha256_lf'].items():
            if hashlib.sha256((Path(__file__).parent/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
                raise ValueError('Saved inference module hash differs')
    attempts=inspect(raw,inputs);write(raw/'attempts.json',attempts)
    available=[r['case_index'] for r in attempts['cases'] if r['has_final_worker_summary']]
    errors=[]
    for index in available:
        try:
            verification=verify_case(raw,inputs,index)
            write(raw/f'verification-{index}.json',verification)
            trace_audit(raw,inputs,index)
        except Exception as exc:
            # Preserve the batch and expose failed verification, rather than
            # silently dropping an arm or publishing it as a successful result.
            errors.append({'case_index':index,'stage':'raw verification','error':str(exc)})
            write(raw/f'verification-{index}.json',{'case':attempts['cases'][index]['case'],'all_pass':False,'error':str(exc)})
    verification_rows=[json.loads((raw/f'verification-{index}.json').read_text()) for index in available]
    write(raw/'verification.json',{'source_fingerprint':source_fingerprint(),
        'all_pass':len(verification_rows)==16 and all(r['all_pass'] for r in verification_rows),
        'cases':verification_rows,'qualification':'independent scoring/moment/cache replay; separate from convergence and physics validity'})
    saved=any(r['has_saved_retained_chain'] for r in attempts['cases'])
    numerical=False
    if saved:
        try:
            transition_audit(raw,inputs,raw/'posterior-transition-diagnostic.json')
            records=json.loads((raw/'posterior-transition-diagnostic.json').read_text())['cases']
            numerical=bool(records) and all(r['maximum_covariance_relative_eigenvalue_error']<1e-5
                and r['maximum_load_relative_frobenius_error']<1e-5
                and r['minimum_approximate_covariance_eigenvalue']>0 for r in records)
            write(raw/'innovation-coordinate-diagnostic.json',innovation_audit(raw,inputs))
            noise_audit(raw,inputs,raw/'noise-transfer-diagnostic.json')
        except Exception as exc:
            errors.append({'stage':'posterior numerical/density audit','error':str(exc)})
            numerical=False
    initial=json.loads((inputs/'transition-audit.json').read_text())
    if initial['source_fingerprint']!=source_fingerprint():raise ValueError('Initial geometry audit source differs')
    numerical=numerical and initial['all_pass']
    write(raw/'numerical-target-audit.json',{'source_fingerprint':source_fingerprint(),
        'historical_target_superseded':False,'model_integrals_numerically_valid':bool(numerical),
        'reason':'Initial-geometry and stratified saved-state independent matrix checks '+('passed' if numerical else 'are incomplete or failed')+'. These checks do not validate contact, material/renderer fidelity, unsampled tails or uncertainty calibration.',
        'saved_state_tolerance':1e-5,'errors':errors})
    renderer_audit(inputs,raw)
    publish(raw,inputs,output)
    summary=json.loads((output/'summary.json').read_text())
    run_id=attempts['run_id'];url=f'https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/{run_id}'
    status=('All predeclared finite-chain, pair-agreement, raw replay and numerical checks pass.'
        if summary['all_cases_eligible_for_ranking'] else 'Incomplete or failed validation: no overall scientific ranking.')
    lines=['# Corrected PAC-NeRF sparse-view comparison','',status,'',
        f"{summary['available_cases']}/16 final worker summaries; {summary['finite_chain_pass_cases']}/16 finite-chain passes.",
        '',f'[Actions experiment]({url}) · [Offline report](https://github.com/mohsenkhadem1/physgauss-sde/blob/main/examples/pacnerf-exact/report.html) · [Frozen protocol](https://github.com/mohsenkhadem1/physgauss-sde/blob/main/docs/PACNERF_EXACT_PROTOCOL.md)',
        '', 'Frozen worker `219665b`; exact linear integrals, exact causal innovation coordinates, fixed full-RGB Gauss-Newton proposal metric, full nonlinear Metropolis target.',
        '', '| Case | Eligible finite chains | Gap coordinate RMSE (mm) | 90% component coverage |',
        '|---|---:|---:|---:|']
    for row in summary['cases']:
        if row['missing']:lines.append(f"| {row['case']} | incomplete | — | — |")
        else:
            score=row['posterior_scores']
            lines.append(f"| {row['case']} | {row['valid_for_ranking']} | {1000*score['gap_particle_coordinate_rmse_m']:.3f} | {100*score['gap_component_90_coverage']:.2f}% |")
    lines+=['','Two deterministic official elastic simulations, Torus and Bird. Initial geometry and eleven-camera initial appearance are supplied; 4/6 frames mean 3/5 later images. Camera switching is not continuous moving-camera capture. The model is a twelve-mode elastic graph with frame-held contact and 32px Gaussian rendering, not the original MPM.',
        '', 'Marginal and joint loads have the same integrated motion posterior. Differences beyond Monte Carlo uncertainty invalidate equivalence claims. Force coefficients are a restricted acceleration field; no force ground truth is supplied. Coverage is descriptive, not repeated-dataset calibration.',
        '', 'Failed or unfinished cells are excluded from quality rankings. The report preserves their status and any failed-chain scores as diagnostics. Gradient-batches/pose-ESS are a compute proxy, not a hardware-normalized runtime claim. Checkpoint counters exclude unsaved work after interrupted attempts.',
        '', 'Compact reports are committed; raw states and inputs remain in Actions artifacts and local copies. Artifact retention is 30 days.']
    document='\n'.join(lines)+'\n'
    (output/'RESULTS.md').write_text(document,encoding='utf-8')
    publication=json.loads((output/'publication.json').read_text())
    publication['files_sha256_lf']['RESULTS.md']=hashlib.sha256(document.encode()).hexdigest()
    write(output/'publication.json',publication)
    if push:
        if os.environ.get('GITHUB_REPOSITORY')!='mohsenkhadem1/physgauss-sde':raise ValueError('Publication is restricted to the authorized repository')
        git('fetch','origin','main');destination=Path('runs/pacnerf-publication-checkout').resolve()
        if destination.exists():raise ValueError('Publication checkout already exists')
        git('worktree','add','--detach',str(destination),'origin/main')
        target=destination/'examples'/'pacnerf-exact'
        if target.exists():raise ValueError('A corrected report already exists; refusing to overwrite it')
        for path in output.rglob('*'):
            if path.is_file() and path.suffix not in ['.json','.html','.csv','.png','.md']:
                raise ValueError('Non-compact artifact cannot be committed')
        shutil.copytree(output,target)
        result_document=destination/'docs'/'PACNERF_EXACT_RESULTS.md'
        source_document=Path('docs/PACNERF_EXACT_RESULTS.md')
        if result_document.exists() and (not source_document.exists() or result_document.read_bytes()!=source_document.read_bytes()):
            raise ValueError('Results document changed after dispatch; refusing to overwrite it')
        result_document.write_text(document,encoding='utf-8')
        git('config','user.name','github-actions[bot]',cwd=destination)
        git('config','user.email','41898282+github-actions[bot]@users.noreply.github.com',cwd=destination)
        git('add','--','examples/pacnerf-exact','docs/PACNERF_EXACT_RESULTS.md',cwd=destination)
        git('commit','-m',f'Publish audited exact PAC-NeRF comparison from Actions {run_id}',cwd=destination)
        git('push','origin','HEAD:main',cwd=destination)
    print(json.dumps({'available_cases':summary['available_cases'],
        'all_cases_eligible_for_ranking':summary['all_cases_eligible_for_ranking'],'publication_pushed':push}),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--push-compact-report',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    finalize(a.raw,a.inputs,a.output,a.push_compact_report)
