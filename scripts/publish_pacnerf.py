"""Publish auditable compact PAC-NeRF results, including failed cases."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pacnerf_comparison import cases, make_model
from physgauss.cli import source_fingerprint


def publish(raw, inputs, output):
    raw, inputs, output = map(Path, (raw, inputs, output))
    if output.exists(): raise ValueError('Choose an unused publication directory')
    output.mkdir(parents=True)
    attempts={}
    if (raw/'attempts.json').exists():
        attempt_audit=json.loads((raw/'attempts.json').read_text())
        if attempt_audit['source_fingerprint']!=source_fingerprint():raise ValueError('Attempt audit source differs')
        attempts={r['case']:r for r in attempt_audit['cases']}
        if len(attempts)!=16:raise ValueError('Attempt audit does not cover the plan')
        shutil.copyfile(raw/'attempts.json',output/'attempts.json')
    rows = []
    for index, (scene, views, noise, force) in enumerate(cases()):
        name = f'{scene}__{views}views__{noise}__{force}'
        preferred=raw/f'results-{index}'
        src = next(iter(preferred.glob(f'**/{name}/summary.json')), None) if preferred.exists() else None
        if src is None:src=next(iter(raw.glob(f'**/{name}/summary.json')), None)
        if src is None:
            attempt=attempts.get(name,{})
            rows.append({'case': name, 'missing': True, 'valid_for_ranking': False,
                'attempted':attempt.get('attempted',False),'attempt_status':attempt.get('status','not_available'),
                'saved_retained_draws_per_chain':attempt.get('retained_draws_per_chain')}); continue
        row = json.loads(src.read_text()); row['missing'] = False
        row['attempted']=True
        row['attempt_status']='finite_checks_pass' if row['valid_for_ranking'] else 'finite_checks_failed'
        row['saved_retained_draws_per_chain']=attempts.get(name,{}).get('retained_draws_per_chain',row['sampling_history'][-1]['draws_per_chain'])
        if row['settings']['source_fingerprint']!=source_fingerprint():raise ValueError('Numerical source differs from production')
        fit_record=json.loads((src.parent/'fit.json').read_text())
        stationarity=fit_record['stationarity']
        if not stationarity['stationary']:raise ValueError('Sampling reference failed the stationarity gate')
        final_step=stationarity['history'][-1]
        if stationarity['newton_decrement']>=.02 or final_step['curvature']['minimum_observed_eigenvalue']<=0:
            raise ValueError('Mode stationarity flag could not be reproduced')
        row['fit']['phase']='initial direct-coordinate L-BFGS, before precision-coordinate/Newton refinement'
        row['final_mode']={'objective':final_step['objective'],
            'regularized_metric_decrement':stationarity['newton_decrement'],
            'minimum_observed_curvature_eigenvalue':final_step['curvature']['minimum_observed_eigenvalue']}
        diagnostics=json.loads((src.parent/'diagnostics.json').read_text())
        flags=[]
        for v in diagnostics['quantities'].values():
            valid=(v['rank_folded_split_rhat'] is not None and v['rank_folded_split_rhat']<1.01
                and v['bulk_ess'] is not None and v['bulk_ess']>=400
                and v['tail_ess'] is not None and v['tail_ess']>=400 and v['frozen_chains']==0)
            if valid != v['diagnostic_pass']: raise ValueError('Diagnostic flag not reproduced')
            flags.append(valid)
        if all(flags) != diagnostics['all_monitored_quantities_pass']: raise ValueError('Diagnostic aggregate differs')
        if row['valid_for_ranking'] != (all(flags) and not row['settings']['pilot']): raise ValueError('Ranking flag differs')
        manifest=json.loads((inputs/f'{scene}-manifest.json').read_text())
        if row['settings']['input_sha256']!=manifest['inputs_sha256']:raise ValueError('Input artifact hash differs')
        posterior=json.loads((src.parent/'posterior.json').read_text())
        force=posterior['force']
        np.testing.assert_allclose(np.array(force['coefficient_covariance']),
            np.array(force['mean_conditional_covariance'])+np.array(force['between_trajectory_covariance']),atol=1e-10,rtol=1e-10)
        destination = output/'cases'/name; destination.mkdir(parents=True)
        for f in src.parent.glob('*.json'): shutil.copyfile(f, destination/f.name)
        shutil.copyfile(src,destination/'worker_summary.json')
        replay=raw/f'verification-{index}.json'
        if replay.exists():shutil.copyfile(replay,destination/'raw_verification.json')
        (destination/'summary.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
        rows.append(row)
    for f in inputs.glob('*-manifest.json'): shutil.copyfile(f, output/f.name)
    provenance=output/'provenance';provenance.mkdir()
    for f in raw.glob('*/artifact_download.json'):
        shutil.copyfile(f,provenance/(f.parent.name+'.json'))
    for folder,name in [(inputs,'input-artifact.json'),(raw,'workflow-status.json')]:
        src=folder/('artifact_download.json' if folder==inputs else 'workflow_status.json')
        if src.exists():shutil.copyfile(src,provenance/name)
    complete = all(not r['missing'] for r in rows)
    passed = complete and all(r['valid_for_ranking'] for r in rows)
    agreement = []
    for scene in ['torus', 'bird']:
        for views in [4, 6]:
            for noise in ['constant', 'neural']:
                names = [f'{scene}__{views}views__{noise}__{force}' for force in ['marginal','joint']]
                files = [output/'cases'/n/'posterior.json' for n in names]
                if not all(f.exists() for f in files): continue
                a,b = [json.loads(f.read_text()) for f in files]
                delta = np.array(a['modal_pose_mean_m'])[1:]-np.array(b['modal_pose_mean_m'])[1:]
                sd = np.sqrt(np.array(a['modal_mean_batch_mcse_m'])[1:]**2+np.array(b['modal_mean_batch_mcse_m'])[1:]**2)
                standardized=np.abs(delta)/np.maximum(sd,1e-15)
                cutoff=float(torch.special.ndtri(torch.tensor(1-.05/(2*delta.size),dtype=torch.float64)))
                agreement.append({'scene':scene,'views':views,'noise':noise,
                    'max_modal_mean_difference_m':float(np.abs(delta).max()),
                    'max_difference_over_combined_mcse':float(standardized.max()),
                    'approximate_bonferroni_95_cutoff':cutoff,
                    'coordinates_exceeding_approximate_cutoff':int((standardized>cutoff).sum()),
                    'qualification':'approximate batch-mean CLT/Bonferroni diagnostic; not an exact finite-sample test or calibration proof; same posterior in theory; finite convergence required'})
    global_tests=sum(13*12 for _ in agreement)
    global_cutoff=float(torch.special.ndtri(torch.tensor(1-.05/(2*max(1,global_tests)),dtype=torch.float64)))
    agreement_pass=len(agreement)==8 and all(a['max_difference_over_combined_mcse']<=global_cutoff for a in agreement)
    verification=None
    if (raw/'verification.json').exists():
        verification=json.loads((raw/'verification.json').read_text())
        shutil.copyfile(raw/'verification.json',output/'verification.json')
    expected_cases={f'{s}__{v}views__{n}__{f}' for s,v,n,f in cases()}
    raw_replay_pass=(verification is not None and verification['all_pass']
        and verification['source_fingerprint']==source_fingerprint()
        and len(verification['cases'])==16
        and {r['case'] for r in verification['cases']}==expected_cases
        and all(r['all_pass'] for r in verification['cases']))
    audit_links=['attempts.json'] if attempts else []
    for root,name in [(inputs,'transition-audit.json'),(raw,'contact-integration-diagnostic.json'),
        (raw,'noise-transfer-diagnostic.json'),(raw,'calibration-replay-diagnostic.json'),
        (raw,'posterior-transition-diagnostic.json'),(raw,'renderer-reference-diagnostic.json'),
        (raw,'innovation-coordinate-diagnostic.json')]:
        if (root/name).exists():
            shutil.copyfile(root/name,output/name);audit_links.append(name)
    if (raw/'renderer-reference-audit.png').exists():
        shutil.copyfile(raw/'renderer-reference-audit.png',output/'renderer-reference-audit.png')
    trace_images=[]
    for record in raw.glob('chain-trace-audit-*.json'):
        shutil.copyfile(record,output/record.name);audit_links.append(record.name)
        png=record.with_suffix('.png')
        if png.exists():
            shutil.copyfile(png,output/png.name)
            if not json.loads(record.read_text())['finite_chain_checks_pass']:trace_images.append(png.name)
    ready=passed and agreement_pass and raw_replay_pass
    summary = {'complete':complete, 'all_cases_pass_finite_chain_checks':passed,
        'all_cases_eligible_for_ranking':ready, 'cases':rows,
        'available_cases':sum(not r['missing'] for r in rows),
        'attempted_cases':sum(r['attempted'] for r in rows),
        'finite_chain_pass_cases':sum(r['valid_for_ranking'] for r in rows),
        'raw_checkpoint_verification_pass':raw_replay_pass,'numerical_audit_files':audit_links,
        'paired_sampler_agreement_screen':{'approximate_global_95_cutoff':global_cutoff,
            'coordinates':global_tests,'complete_pairs':len(agreement),'passes':agreement_pass,
            'qualification':'approximate batch-mean Gaussian CLT with Bonferroni correction across all paired modal means; not proof of global exploration or physical calibration'},
        'marginal_joint_agreement':agreement,
        'qualification':'two official elastic simulations; oracle initial geometry and 11-camera initial appearance; sparse later images; reduced model and transferred noise; no force ground truth'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fields = ['case','attempt_status','saved_retained_draws_per_chain','valid_for_ranking','gap_rmse_mm','oracle_subspace_gap_rmse_mm','heldout_psnr_db','background_difference_region_psnr_db','coverage90','width_mm','worst_rhat','minimum_bulk_ess','minimum_tail_ess','seconds']
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            common={key:row[key] for key in ['case','attempt_status','saved_retained_draws_per_chain','valid_for_ranking']}
            if row['missing']:writer.writerow(common);continue
            s=row['posterior_scores'];writer.writerow({**common,
                'gap_rmse_mm':s['gap_particle_coordinate_rmse_m']*1000,'heldout_psnr_db':s['heldout_rgb_psnr_db'],
                'oracle_subspace_gap_rmse_mm':s['evaluation_only_subspace_oracle_gap_rmse_m']*1000,
                'background_difference_region_psnr_db':s['heldout_background_difference_region_psnr_db'],
                'coverage90':s['gap_component_90_coverage'],'width_mm':s['gap_mean_width_m']*1000,
                **{k:row[k] for k in ['worst_rhat','minimum_bulk_ess','minimum_tail_ess','seconds']}})
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for axis,scene in zip(axes,['torus','bird']):
        labels=[];values=[];colors=[]
        for row in rows:
            if row['missing'] or not row['case'].startswith(scene):continue
            labels.append(row['case'].replace(scene+'__','').replace('__','\n'))
            values.append(row['posterior_scores']['gap_particle_coordinate_rmse_m']*1000)
            colors.append('#278298' if row['valid_for_ranking'] else '#aaaaaa')
        axis.bar(range(len(values)),values,color=colors);axis.set_xticks(range(len(labels)),labels,fontsize=7)
        axis.set_title(scene.title());axis.set_ylabel('Gap particle coordinate RMSE (mm)')
    fig.suptitle('PAC-NeRF sparse-image transfer: gray bars fail convergence or are pilots')
    fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(11,5.5),layout='constrained')
    for row,index in enumerate([0,8]):
        model,_=make_model(inputs,index)
        truth=torch.load(inputs/f"{model.data['scene']}-evaluation.pt",weights_only=True)['particles']
        eye,rotation,k=model.data['cameras'][0]
        for col,frame in enumerate([0,3,6,13]):
            camera=(rotation@(truth[frame]-eye).T).T
            uv=camera[:,:2]/camera[:,2:]*k.diagonal()[:2]+k[:2,2]-.5
            ax=axes[row,col];ax.imshow(model.data['images'][0,frame]);ax.scatter(uv[:,0],uv[:,1],s=3,c='cyan',alpha=.5)
            ax.set_title(model.data['scene'].title()+f' frame {frame}')
            ax.set_xlim(-.5,31.5);ax.set_ylim(31.5,-.5);ax.axis('off')
    fig.suptitle('Reference alignment audit: cyan dots are supplied truth, not model predictions\nCamera 0; reference geometry is used here for evaluation only',fontsize=12)
    fig.savefig(output/'reference_alignment.png',dpi=150);plt.close(fig)
    # Representative actual withheld image, selected by a fixed case/frame/view.
    index=6;name='torus__6views__neural__marginal'
    record=output/'cases'/name/'posterior.json'
    if record.exists():
        model,_=make_model(inputs,index)
        p=json.loads(record.read_text());q=torch.tensor(p['modal_pose_mean_m'],dtype=torch.float64)
        specs=[(6,0),(6,7)]
        with torch.no_grad():rgb=model.renderer(q[[6,6]][None],specs)[0].clamp(0,1)
        constant_file=output/'cases'/'torus__6views__constant__marginal'/'posterior.json'
        constant_rgb=None
        if constant_file.exists():
            const_q=torch.tensor(json.loads(constant_file.read_text())['modal_pose_mean_m'],dtype=torch.float64)
            with torch.no_grad():constant_rgb=model.renderer(const_q[[6,6]][None],specs)[0].clamp(0,1)
        columns=4 if constant_rgb is not None else 3
        fig,axes=plt.subplots(2,columns,figsize=(3*columns,6),layout='constrained')
        for i,(f,c) in enumerate(specs):
            target=model.data['images'][c,f]
            axes[i,0].imshow(target)
            if constant_rgb is not None:
                axes[i,1].imshow(constant_rgb[i]);axes[i,1].set_title('Constant noise: mean pose')
            neural_column=columns-2;error_column=columns-1
            axes[i,neural_column].imshow(rgb[i]);axes[i,error_column].imshow((target-rgb[i]).abs().mean(-1),vmin=0,vmax=.3,cmap='magma')
            for ax in axes[i]:ax.axis('off')
            axes[i,0].set_title(f'Official image: frame {f}, camera {c}')
            axes[i,neural_column].set_title('Neural noise: mean pose');axes[i,error_column].set_title('Neural absolute error (0–0.3)')
        fig.suptitle('Torus six-view schedule; marginalized loads; both image pairs withheld')
        fig.savefig(output/'images.png',dpi=150);plt.close(fig)
    force_files=[(noise,arm,output/'cases'/f'torus__6views__{noise}__{arm}'/'posterior.json')
        for noise in ['constant','neural'] for arm in ['marginal','joint']]
    force_files=[r for r in force_files if r[2].exists()]
    if force_files:
        time=np.arange(14)/24;knots=np.linspace(0,13/24,4)
        temporal=np.exp(-.5*((time[:,None]-knots)/.14)**2)
        temporal/=np.sqrt((temporal**2).sum(-1,keepdims=True))
        mapping=np.zeros((14,3,4,3,2))
        for axis in range(3):mapping[:,axis,:,axis,:]=2*temporal[:,:,None]
        mapping=mapping.reshape(14,3,24)
        fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True,layout='constrained')
        for noise,arm,file in force_files:
            color='#197a91' if noise=='constant' else '#c37b25'
            label=noise.title()+' noise, '+('marginalized' if arm=='marginal' else 'explicit joint')+' loads'
            force=json.loads(file.read_text())['force']
            mean=np.einsum('tck,k->tc',mapping,np.array(force['coefficient_mean']))
            variance=np.einsum('tck,kl,tcl->tc',mapping,np.array(force['coefficient_covariance']),mapping)
            sd=np.sqrt(np.maximum(variance,0))
            for axis,ax in enumerate(axes):
                ax.plot(time,mean[:,axis],color=color,linestyle='-' if arm=='marginal' else '--',label=label)
                ax.fill_between(time,mean[:,axis]-sd[:,axis],mean[:,axis]+sd[:,axis],color=color,alpha=.09)
        for axis,ax in enumerate(axes):
            ax.set_ylabel(f'{"xyz"[axis]} (m/s²)');ax.axvspan(5/24,8/24,color='#aaaaaa',alpha=.15)
        axes[0].legend(fontsize=8,ncol=2);axes[-1].set_xlabel('Time (s)')
        fig.suptitle('Torus six-view schedule: mean modeled unknown acceleration\nPosterior mean ± one SD; restricted profiles; excludes known gravity; not force ground truth')
        fig.savefig(output/'loads.png',dpi=150);plt.close(fig)
    header=('Finite diagnostics, paired-sampler agreement and raw replay pass' if ready else
        'Finite diagnostics pass, but paired samplers disagree — no final ranking' if passed else
        'Development / incomplete inference — no scientific ranking')
    if passed and agreement_pass and not raw_replay_pass:header='Finite diagnostics and paired agreement pass; raw verification pending'
    def embed(name):
        path=output/name
        return '<img style="max-width:100%" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">' if path.exists() else ''
    table=''
    for r in rows:
        cells=[html.escape(r['case']),html.escape(r['attempt_status'].replace('_',' ')),
            str(r['saved_retained_draws_per_chain']) if r['saved_retained_draws_per_chain'] is not None else '—']
        if r['missing']:cells+=['—']*4
        else:
            s=r['posterior_scores']
            cells+=[f"{s['gap_particle_coordinate_rmse_m']*1000:.2f}",
                f"{100*s['gap_component_90_coverage']:.1f}%",f"{s['gap_mean_width_m']*1000:.2f}",
                f"{s['heldout_background_difference_region_psnr_db']:.2f}"]
        table+='<tr>'+''.join('<td>'+cell+'</td>' for cell in cells)+'</tr>'
    page=f'''<!doctype html><html><meta charset="utf-8"><title>PAC-NeRF sparse comparison</title><body style="font:17px system-ui;max-width:1150px;margin:40px auto;padding:20px;background:#fafafa;color:#182c38">
<h1>PAC-NeRF elastic objects: sparse dynamic images</h1><h2>{header}</h2>
<p>{summary['attempted_cases']} of 16 cases launched; {summary['available_cases']} final worker results available; {summary['finite_chain_pass_cases']} pass finite-chain checks. A numerical replay pass does not turn an unconverged posterior into a valid estimate. Saved draws in unfinished attempts are checkpoint prefixes, not eligible posterior results.</p>
<p>{html.escape(summary['qualification'])}.</p><p>Constant and neural noise each use either analytically marginalized Gaussian loads or the identical explicit joint load prior. Their integrated motion posteriors should agree. Marginalization changes inference, not physical assumptions.</p>
<p>The 4/6-view schedule labels include frame zero: they supply 3/5 later dynamic images over 14 timestamps (24 fps), one selected camera per observed time. The fixed-camera dataset is subsampled across cameras; this is not continuous moving-camera capture. All eleven initial views initialize appearance and initial geometry is supplied. Ground truth after frame zero is used only after sampling for evaluation.</p>
{embed('comparison.png')}{embed('images.png')}{embed('loads.png')}
<h2>Reference data audit</h2><p>The following overlays show the supplied particle truth projected into official images. They check coordinate/time conventions and are not reconstructions from the model. Future particle geometry is used only for evaluation.</p>{embed('reference_alignment.png')}
<p>The next audit renders supplied true particle centers and their least-squares projection into the fixed shape subspace. Colors and all rendering settings remain fixed from initialization. It exposes image-model error even when geometry is supplied. These are evaluation-only diagnostic references, not sparse reconstructions or bounds on achievable RGB scores.</p>{embed('renderer-reference-audit.png')}
<p>Interval coverage is descriptive over correlated particles, not a calibration guarantee. Model reduction, explicit ground contact, transferred diffusion, and 32px rendering limit physical conclusions. Stiffness scale is not a continuum Young’s modulus estimate. No force-error score is claimed.</p>
<p>The linear propagator passes independent matrix-exponential checks. A post-hoc nominal zero-noise contact check found 16.4/17.2 mm whole-sequence coordinate differences for Torus/Bird against a refined solve of the same reduced ODE. Frame-held contact is therefore a material approximation. This is not an error decomposition of the fitted posterior or validation against the original MPM. The frozen comparison was not retuned after this finding.</p>
<table cellpadding="8"><tr><th>Case</th><th>Inference status</th><th>Saved draws per chain</th><th>Gap coordinate RMSE (mm)</th><th>90% interval coverage</th><th>Mean interval width (mm)</th><th>Background-difference region PSNR (dB)</th></tr>{table}</table>
{('<h2>Failed convergence audits</h2><p>These runs are excluded from scientific ranking. Checking material parameters alone can miss slow transitions in motion and force variables.</p>'+''.join(embed(name) for name in trace_images)) if trace_images else ''}
<p><a href="summary.json">Results and diagnostics summary</a> · <a href="comparison.csv">CSV</a> · {('<a href="attempts.json">Attempt and checkpoint audit</a> · ' if attempts else '')}<a href="https://github.com/mohsenkhadem1/physgauss-sde/blob/main/docs/PACNERF_PROTOCOL.md">Protocol and limitations</a> · <a href="https://github.com/xuan-li/PAC-NeRF">Original data source</a></p></body></html>'''
    (output/'report.html').write_text(page,encoding='utf-8')
    hashes={str(f.relative_to(output)).replace('\\','/'):hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n') if f.suffix in ['.json','.csv','.html'] else f.read_bytes()).hexdigest() for f in output.rglob('*') if f.is_file()}
    (output/'publication.json').write_text(json.dumps({'files_sha256_lf':hashes},indent=2)+'\n')
    print(json.dumps({'complete':complete,'all_cases_eligible_for_ranking':ready,'files':len(hashes)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2);publish(a.raw,a.inputs,a.output)
