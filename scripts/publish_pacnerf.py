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


def publish(raw, inputs, output):
    raw, inputs, output = map(Path, (raw, inputs, output))
    if output.exists(): raise ValueError('Choose an unused publication directory')
    output.mkdir(parents=True)
    rows = []
    for index, (scene, views, noise, force) in enumerate(cases()):
        name = f'{scene}__{views}views__{noise}__{force}'
        src = next(iter(raw.glob(f'**/{name}/summary.json')), None)
        if src is None:
            rows.append({'case': name, 'missing': True, 'valid_for_ranking': False}); continue
        row = json.loads(src.read_text()); row['missing'] = False
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
        rows.append(row)
    for f in inputs.glob('*-manifest.json'): shutil.copyfile(f, output/f.name)
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
                agreement.append({'scene':scene,'views':views,'noise':noise,
                    'max_modal_mean_difference_m':float(np.abs(delta).max()),
                    'max_difference_over_combined_mcse':float((np.abs(delta)/np.maximum(sd,1e-15)).max()),
                    'qualification':'descriptive batch-mean CLT diagnostic; same posterior in theory; finite convergence required'})
    summary = {'complete':complete, 'all_cases_eligible_for_ranking':passed, 'cases':rows,
        'marginal_joint_agreement':agreement,
        'qualification':'two official elastic simulations; oracle initial geometry and 11-camera initial appearance; sparse later images; reduced model and transferred noise; no force ground truth'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fields = ['case','valid_for_ranking','gap_rmse_mm','oracle_subspace_gap_rmse_mm','heldout_psnr_db','background_difference_region_psnr_db','coverage90','width_mm','worst_rhat','minimum_bulk_ess','minimum_tail_ess','seconds']
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            if row['missing']:writer.writerow({'case':row['case'],'valid_for_ranking':False});continue
            s=row['posterior_scores'];writer.writerow({'case':row['case'],'valid_for_ranking':row['valid_for_ranking'],
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
    # Representative actual withheld image, selected by a fixed case/frame/view.
    index=6;name='torus__6views__neural__marginal'
    record=output/'cases'/name/'posterior.json'
    if record.exists():
        model,_=make_model(inputs,index)
        p=json.loads(record.read_text());q=torch.tensor(p['modal_pose_mean_m'],dtype=torch.float64)
        specs=[(6,0),(6,7)]
        with torch.no_grad():rgb=model.renderer(q[[6,6]][None],specs)[0].clamp(0,1)
        fig,axes=plt.subplots(2,3,figsize=(9,6),layout='constrained')
        for i,(f,c) in enumerate(specs):
            target=model.data['images'][c,f]
            axes[i,0].imshow(target);axes[i,1].imshow(rgb[i]);axes[i,2].imshow((target-rgb[i]).abs().mean(-1),vmin=0,vmax=.3,cmap='magma')
            for ax in axes[i]:ax.axis('off')
            axes[i,0].set_title(f'Official image: frame {f}, camera {c}')
            axes[i,1].set_title('Rendered posterior mean pose');axes[i,2].set_title('Absolute RGB error (0–0.3)')
        fig.savefig(output/'images.png',dpi=150);plt.close(fig)
    header='Converged comparison' if passed else 'Development / incomplete inference — no scientific ranking'
    def embed(name):
        path=output/name
        return '<img style="max-width:100%" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">' if path.exists() else ''
    table=''.join('<tr><td>'+html.escape(r['case'])+'</td><td>'+('missing' if r['missing'] else str(r['valid_for_ranking']))+'</td><td>'+('—' if r['missing'] else f"{r['posterior_scores']['gap_particle_coordinate_rmse_m']*1000:.2f}")+'</td></tr>' for r in rows)
    page=f'''<!doctype html><html><meta charset="utf-8"><title>PAC-NeRF sparse comparison</title><body style="font:17px system-ui;max-width:1150px;margin:40px auto;padding:20px;background:#fafafa;color:#182c38">
<h1>PAC-NeRF elastic objects: sparse dynamic images</h1><h2>{header}</h2>
<p>{html.escape(summary['qualification'])}.</p><p>Constant and neural noise each use either analytically marginalized Gaussian loads or the identical explicit joint load prior. Their integrated motion posteriors should agree. Marginalization changes inference, not physical assumptions.</p>
<p>4 or 6 observed dynamic images over 14 timestamps (24 fps), one selected camera per timestamp. The fixed-camera dataset is subsampled across cameras; this is not continuous moving-camera capture. All eleven initial views initialize appearance. Ground truth after frame zero is used only after sampling for evaluation.</p>
{embed('comparison.png')}{embed('images.png')}
<p>Interval coverage is descriptive over correlated particles, not a calibration guarantee. Model reduction, explicit ground contact, transferred diffusion, and 32px rendering limit physical conclusions. Stiffness scale is not a continuum Young’s modulus estimate. No force-error score is claimed.</p>
<table cellpadding="8"><tr><th>Case</th><th>Eligible for ranking</th><th>Gap RMSE (mm)</th></tr>{table}</table>
<p><a href="summary.json">Complete results and diagnostics summary</a> · <a href="comparison.csv">CSV</a> · <a href="https://github.com/mohsenkhadem1/physgauss-sde/blob/main/docs/PACNERF_PROTOCOL.md">Protocol and limitations</a> · <a href="https://github.com/xuan-li/PAC-NeRF">Original data source</a></p></body></html>'''
    (output/'report.html').write_text(page,encoding='utf-8')
    hashes={str(f.relative_to(output)).replace('\\','/'):hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n') if f.suffix in ['.json','.csv','.html'] else f.read_bytes()).hexdigest() for f in output.rglob('*') if f.is_file()}
    (output/'publication.json').write_text(json.dumps({'files_sha256_lf':hashes},indent=2)+'\n')
    print(json.dumps({'complete':complete,'all_cases_eligible_for_ranking':passed,'files':len(hashes)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2);publish(a.raw,a.inputs,a.output)
