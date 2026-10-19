"""Publish a new immutable comparison snapshot, preserving failed/missing cells."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
from pacnerf_comparison import cases,write
from verify_pacnerf_discrepancy import verify_case


def snapshot_push(output,name,document=None):
    import re
    if not re.fullmatch(r'pacnerf-[a-z0-9-]+',name):raise ValueError('Unsafe snapshot name')
    def git(*args,cwd=None):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
    git('fetch','origin','main')
    worktree=Path('runs')/('publication-'+name)
    worktree=worktree.resolve()
    if worktree.exists():raise ValueError('Snapshot checkout already exists')
    git('worktree','add','--detach',str(worktree),'origin/main')
    target=worktree/'examples'/name
    if target.exists():raise ValueError('Published snapshot already exists; refusing replacement')
    for path in Path(output).rglob('*'):
        if path.is_file() and path.suffix not in ['.json','.md','.csv','.png','.html']:
            raise ValueError('Raw/noncompact files cannot be committed')
    shutil.copytree(output,target)
    paths=['examples/'+name]
    if document:
        doc=worktree/'docs'/(name.upper().replace('-','_')+'_RESULTS.md')
        if doc.exists():raise ValueError('Results document exists')
        doc.write_text(document,encoding='utf-8',newline='\n');paths.append('docs/'+doc.name)
    git('config','user.name','github-actions[bot]',cwd=worktree)
    git('config','user.email','41898282+github-actions[bot]@users.noreply.github.com',cwd=worktree)
    git('add','--',*paths,cwd=worktree)
    git('commit','-m','Publish audited '+name,cwd=worktree)
    # A simultaneous independent report may advance main. Rebase this snapshot
    # commit (unique paths only) onto it; never overwrite or force push.
    for attempt in range(3):
        result=subprocess.run(['git','push','origin','HEAD:main'],cwd=worktree)
        if result.returncode==0:return
        git('fetch','origin','main',cwd=worktree);git('rebase','origin/main',cwd=worktree)
    raise ValueError('Publication push failed after three ordinary rebases')


def publish(raw,inputs,output,baseline,name,push=False):
    raw,inputs,output,baseline=map(Path,(raw,inputs,output,baseline))
    if output.exists():raise ValueError('Choose a fresh publication folder')
    output.mkdir(parents=True)
    rows=[];errors=[]
    for index,(scene,views,noise,force) in enumerate(cases()):
        case=f'{scene}__{views}views__{noise}__{force}'
        paths=list((raw/f'results-{index}').glob(f'**/{case}/summary.json'))
        old=baseline/'cases'/case/'summary.json'
        row={'case':case,'case_index':index,'missing':not bool(paths),'valid_for_ranking':False,
            'baseline':json.loads(old.read_text()) if old.exists() else None}
        old_proof=baseline/'cases'/case/'raw_verification.json'
        row['baseline_prediction_scores']=(json.loads(old_proof.read_text()).get('prediction_scores') if old_proof.exists() else None)
        if paths:
            if len(paths)!=1:raise ValueError('Ambiguous final worker summary')
            summary=json.loads(paths[0].read_text())
            row.update(summary);row['missing']=False;row['valid_for_ranking']=False
            try:
                proof=verify_case(raw,inputs,index)
                row['verification']=proof
                row['valid_for_ranking']=bool(summary['valid_for_ranking'])
            except Exception as exc:
                row['verification']={'all_pass':False,'error':str(exc)}
                errors.append({'case':case,'stage':'raw replay','error':str(exc)})
            destination=output/'cases'/case;destination.mkdir(parents=True)
            for path in paths[0].parent.glob('*.json'):shutil.copyfile(path,destination/path.name)
            shutil.copyfile(paths[0],destination/'worker_summary.json')
            write(destination/'summary.json',row)
            write(destination/'independent_verification.json',row['verification'])
        else:
            # Preserve progress/errors rather than silently dropping timeouts.
            root=raw/f'results-{index}'
            destination=output/'incomplete'/case
            if root.exists():
                destination.mkdir(parents=True)
                for path in root.glob(f'**/{case}/*.json'):shutil.copyfile(path,destination/path.name)
        rows.append(row)
    agreement=[]
    for scene in ['torus','bird']:
        for views in [4,6]:
            for noise in ['constant','neural']:
                pair=[next(r for r in rows if r['case']==f'{scene}__{views}views__{noise}__{f}') for f in ['marginal','joint']]
                if not all(r['valid_for_ranking'] for r in pair):continue
                post=[json.loads((output/'cases'/r['case']/'posterior.json').read_text()) for r in pair]
                means=[np.r_[np.asarray(p['modal_pose_mean_m'])[1:].ravel(),p['material_parameter_mean'],p['hyperparameter_mean'],p['force']['coefficient_mean']] for p in post]
                mcse=[np.r_[np.asarray(p['modal_mean_batch_mcse_m'])[1:].ravel(),p['material_parameter_mean_batch_mcse'],p['hyperparameter_mean_batch_mcse'],p['force']['coefficient_mean_batch_mcse']] for p in post]
                standardized=np.abs(means[0]-means[1])/np.maximum(np.sqrt(mcse[0]**2+mcse[1]**2),1e-15)
                agreement.append({'scene':scene,'views':views,'noise':noise,
                    'coordinates_tested':len(standardized),'maximum_difference_over_combined_batch_mcse':float(standardized.max())})
    tests=sum(a['coordinates_tested'] for a in agreement)
    cutoff=float(torch.special.ndtri(torch.tensor(1-.05/(2*max(1,tests)),dtype=torch.float64)))
    pair_pass=len(agreement)==8 and all(a['maximum_difference_over_combined_batch_mcse']<=cutoff for a in agreement)
    finite=sum(r['valid_for_ranking'] for r in rows)
    baseline_finite=sum(bool((r.get('baseline') or {}).get('valid_for_ranking')) for r in rows)
    baseline_summary=json.loads((baseline/'summary.json').read_text()) if (baseline/'summary.json').exists() else {}
    baseline_valid=bool(baseline_summary.get('all_cases_eligible_for_ranking',False))
    summary={'snapshot':name,'available_cases':sum(not r['missing'] for r in rows),
        'finite_chain_and_raw_replay_pass_cases':finite,'baseline_finite_chain_pass_cases':baseline_finite,
        'new_model_all_cases_pass':finite==16 and pair_pass,
        'baseline_all_validation_gates_pass':baseline_valid,
        'all_cases_eligible_for_overall_comparison':finite==16 and pair_pass and baseline_finite==16 and baseline_valid,
        'pair_agreement':agreement,'pair_agreement_global_bonferroni_95_cutoff':cutoff,
        'pair_agreement_qualification':'approximate batch-mean CLT check of identical marginal path/hyperparameter posteriors; not an exact finite-sample test',
        'cases':rows,'verification_errors':errors,
        'qualification':'post-audit exploratory combined model correction; two previously inspected deterministic scenes; no independent real-world calibration or attribution to individual correction components'}
    write(output/'summary.json',summary)
    # Keep run/artifact provenance with compact report. Full states stay in artifacts.
    provenance=output/'provenance';provenance.mkdir()
    for path in raw.glob('*/artifact_download.json'):shutil.copyfile(path,provenance/(path.parent.name+'.json'))
    if (raw/'workflow_status.json').exists():shutil.copyfile(raw/'workflow_status.json',provenance/'workflow-status.json')
    for path in inputs.glob('*-manifest.json'):shutil.copyfile(path,provenance/path.name)
    table=[]
    for r in rows:
        s=r.get('posterior_scores',{});old=(r.get('baseline') or {}).get('posterior_scores',{})
        old_probability=r.get('baseline_prediction_scores') or {}
        table.append({'case':r['case'],'eligible':r['valid_for_ranking'],'status':'incomplete' if r['missing'] else 'passed' if r['valid_for_ranking'] else 'failed validation',
            'baseline_eligible':bool((r.get('baseline') or {}).get('valid_for_ranking',False)),
            'baseline_rmse_mm':None if not old else 1000*old['gap_particle_coordinate_rmse_m'],
            'new_rmse_mm':None if not s else 1000*s['gap_particle_coordinate_rmse_m'],
            'baseline_coverage':old.get('gap_component_90_coverage'),'new_coverage':s.get('gap_component_90_coverage'),
            'new_width_mm':None if not s else 1000*s['gap_mean_width_m'],
            'baseline_crps_mm':None if 'mean_crps_m' not in old_probability else 1000*old_probability['mean_crps_m'],
            'new_crps_mm':None if r['missing'] else 1000*r['uncertainty_scores']['mean_crps_m']})
    with (output/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# Hierarchical uncertainty correction: PAC-NeRF comparison','',
        f'{finite}/16 cells pass finite-chain diagnostics and independent raw replay. Overall paired comparison: {summary["all_cases_eligible_for_overall_comparison"]}.',
        '', 'Missing/failed cells are preserved; their scores, when present, are diagnostic only.',
        '', '| Case | New / baseline valid | Old / new RMSE (mm) | Old / new 90% coverage | New width (mm) | Old / new CRPS (mm) |',
        '|---|---|---:|---:|---:|---:|']
    def number(x,digits=2):return '—' if x is None else f'{x:.{digits}f}'
    for r in table:
        lines.append(f"| {r['case']} | {r['status']} / {r['baseline_eligible']} | {number(r['baseline_rmse_mm'])} / {number(r['new_rmse_mm'])} | {number(None if r['baseline_coverage'] is None else 100*r['baseline_coverage'])}% / {number(None if r['new_coverage'] is None else 100*r['new_coverage'])}% | {number(r['new_width_mm'])} | {number(r['baseline_crps_mm'])} / {number(r['new_crps_mm'])} |")
    lines+=['', 'The new model infers additional acceleration diffusion, smooth modal load discrepancy and renderer-error scales, with correlated image residuals. Both force arms have the same expanded Gaussian load prior; force marginalization changes inference, not the ideal path posterior.',
        '', 'Prior scales use gravity, frame duration and supplied rest geometry. Spatial correlation uses supplied initial appearance residuals. Sparse observations alone infer scale posteriors. No future trajectory or held-out-image error selected these settings.',
        '', 'The floor-conditioned target includes opaque floor visibility and a 2mm softened nonpenetration event. Sampling uses a parameter-dependent Gaussian state guide conditioned on linearized sparse images, with the complete nonlinear physical density and Jacobian correction. Conditional state HMC alternates with five parameter elliptical slices; discarded preliminary draws learn only the sampling reference. Joint loads remain explicit in the joint target. The validated Bird four-view constant/marginal checkpoint is reused as case8, without a new replicate claim. The other fifteen cells start fresh after seven additional observed-only mixing pilots. The initial poor-mixing candidate is excluded.',
        '', 'Accuracy, coverage, interval width, interval score and CRPS must be read together. Wider intervals alone are not a demonstrated improvement. Two deterministic trajectories cannot establish repeated-case calibration. The twelve-mode space, fixed appearance, frame-held contact and known initial geometry remain limitations.',
        '', 'Changes were developed after inspecting baseline errors, so this is an exploratory comparison. The combined correction does not isolate causality among added dynamics error, smooth loads and correlated rendering. Missing baseline cells retain their original status.',
        '', 'Raw checkpoints are retained as Actions artifacts for 30 days. Code and compact reports are committed.']
    document='\n'.join(lines)+'\n';(output/'RESULTS.md').write_text(document,encoding='utf-8',newline='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    available=[r for r in table if r['new_rmse_mm'] is not None]
    fig,axes=plt.subplots(1,3,figsize=(16,6));y=np.arange(len(available));labels=[r['case'].replace('__',' / ') for r in available]
    for axis,key,oldkey,label in [(axes[0],'new_rmse_mm','baseline_rmse_mm','Gap RMSE (mm)'),(axes[1],'new_coverage','baseline_coverage','90% component coverage'),(axes[2],'new_width_mm',None,'New interval width (mm)')]:
        axis.barh(y,[r[key] for r in available],color=['#26734d' if r['eligible'] else '#aaa' for r in available])
        if oldkey:
            axis.scatter([np.nan if not r['baseline_eligible'] or r[oldkey] is None else r[oldkey] for r in available],y,color='black',s=15,label='Valid baseline')
            axis.scatter([np.nan if r['baseline_eligible'] or r[oldkey] is None else r[oldkey] for r in available],y,color='gray',marker='x',s=15,label='Invalid baseline diagnostic')
        axis.set_xlabel(label);axis.set_yticks(y);axis.set_yticklabels(labels if axis==axes[0] else [],fontsize=7)
    axes[1].axvline(.9,color='red',linestyle='--');axes[1].set_xlim(0,1);axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    curves=[r for r in rows if r['valid_for_ranking']]
    fig,ax=plt.subplots(figsize=(7,5));ax.plot([0,1],[0,1],'k--')
    for r in curves:
        values=r['uncertainty_scores']['coverage_curve'];ax.plot([v['nominal'] for v in values],[v['coverage'] for v in values],'.-',label=r['case'])
    ax.set(xlim=(.45,1),ylim=(0,1),xlabel='Nominal component coverage',ylabel='Observed gap coverage')
    if curves:ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(output/'coverage.png',dpi=150);plt.close(fig)
    encoded=lambda filename:base64.b64encode((output/filename).read_bytes()).decode()
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>PAC-NeRF uncertainty correction</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Hierarchical uncertainty correction</h1><p>'+html.escape(f'{finite}/16 cells pass diagnostics and raw replay. Overall comparison eligible: {summary["all_cases_eligible_for_overall_comparison"]}')+'</p><img alt="Comparison" src="data:image/png;base64,'+encoded('comparison.png')+'"><img alt="Coverage" src="data:image/png;base64,'+encoded('coverage.png')+'"><pre>'+html.escape(document)+'</pre>',encoding='utf-8',newline='\n')
    hashes={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n') if p.suffix in ['.json','.md','.csv','.html'] else p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
    write(output/'publication.json',{'snapshot':name,'files_sha256_lf':hashes})
    if push:snapshot_push(output,name,document)
    print(json.dumps({'available':summary['available_cases'],'finite_pass':finite,'overall_valid':summary['all_cases_eligible_for_overall_comparison'],'pushed':push}),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    p.add_argument('--baseline',default='examples/pacnerf-exact');p.add_argument('--name',required=True);p.add_argument('--push',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    publish(a.raw,a.inputs,a.output,a.baseline,a.name,a.push)
