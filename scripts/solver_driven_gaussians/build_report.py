"""Build compact scientific figures and a complete numerical audit."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'papers/cvpr2027-physical-response/tools'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figure_style
from PIL import Image
from scripts.solver_driven_gaussians.score_motion import metrics
base=ROOT/'runs/solver-driven-gaussians-v02';report=ROOT/'reports/solver-driven-gaussians-v02';figdir=ROOT/'papers/cvpr2027-physical-response/figures';doc=json.loads((report/'final-results.json').read_text());obs=json.loads((report/'observability.json').read_text())
rows=[]
for case in ['calibration','higher_drop']:
 c=np.load(base/'reference'/(case+'.npz'));f=np.load(base/'reference-halfdt'/(case+'.npz'));rows.append(dict(case=case,**metrics(c['x'],f['x'],c['rest'])))
(report/'time-step-sensitivity.json').write_text(json.dumps(rows,indent=2))
# Same declared 0.24-s frame and shared crop for every method; never warp predictions.
fig,axes=plt.subplots(2,3,figsize=(9,4.9));plt.subplots_adjust(left=.025,right=.995,top=.93,bottom=.06,wspace=.015,hspace=.18)
for row,case in enumerate(['side_poke','pair_drop']):
 panel=np.asarray(Image.open(ROOT/'output/solver-driven-gaussians-v02'/(case+'-comparison.png')));ref=panel[28:268,:320];mask=np.max(abs(ref.astype(float)-245),axis=-1)>14;y,x=np.where(mask);box=[max(0,x.min()-16),min(320,x.max()+17),max(0,y.min()-16),min(240,y.max()+17)]
 for col,label in enumerate(['Reference','Ours','Analytic control']):
  ax=axes[row,col];im=panel[28:268,col*320:(col+1)*320];ax.imshow(im);ax.set_xlim(box[:2]);ax.set_ylim(box[3],box[2]);ax.axis('off')
  if row==0:ax.set_title(label,fontsize=14)
  if col==0:text='Side poke' if row==0 else 'Two deformable objects'
  else:
   g=next(v for v in doc['geometry'] if v['case']==case and v['method']==['learned','analytic'][col-1]);text=f"Forecast error: {g['position_rms_mm']:.2f} mm"
  ax.text(.5,-.04,text,transform=ax.transAxes,ha='center',fontsize=12)
fig.savefig(figdir/'solver-driven-comparisons.pdf',bbox_inches='tight');fig.savefig(figdir/'solver-driven-comparisons.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(10,3.25));plt.subplots_adjust(left=.06,right=.99,bottom=.29,top=.83,wspace=.42)
cases=['calibration','side_poke','early_release','force_pulse','higher_drop','pair_drop'];names=['Continuation','Side poke','Release','Force','Higher drop','Two objects'];x=np.arange(6)
for j,(method,color,label) in enumerate([('learned','#26898c','Ours'),('analytic','#dba04d','Analytic')]):
 vals=[next(g['position_rms_mm'] for g in doc['geometry'] if g['case']==c and g['method']==method) for c in cases];axes[0].bar(x+(j-.5)*.36,vals,.36,color=color,label=label)
 vals=[next(g['union_psnr'] for g in doc['render_summary'] if g['case']==c and g['method']==method and g['views']=='novel') for c in cases];axes[1].plot(x,vals,'o-',color=color,label=label)
for ax,title,y in zip(axes[:2],['Frozen-parameter forecasts','Unseen-camera forecasts'],['Position RMS (mm)','Union-foreground PSNR (dB)']):ax.set_title(title,fontsize=12);ax.set_ylabel(y);ax.set_xticks(x);ax.set_xticklabels(names,rotation=50,ha='right',fontsize=9);ax.spines[['top','right']].set_visible(False)
axes[0].legend(frameon=False,fontsize=9);vals=[obs['results'][k]['position_rms_mm'] for k in ['factor2','prior_mean','selected']];axes[2].bar(range(3),vals,color=['#bc7070','#b9bbbd','#26898c']);axes[2].set_xticks(range(3));axes[2].set_xticklabels(['Wrong scale','Prior mean','After force'],rotation=40,ha='right');axes[2].set_ylabel('Separate-force error (mm)');axes[2].set_title('A force resolves a visual ambiguity',fontsize=11);axes[2].spines[['top','right']].set_visible(False)
fig.savefig(figdir/'solver-driven-results.pdf',bbox_inches='tight');fig.savefig(figdir/'solver-driven-results.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Seal source and inference inputs, and assert no failure is silently excluded.
checks={};checks['six_cases_both_methods']=len(doc['geometry'])==12;checks['positive_jacobians']=min(x['minimum_J'] for x in doc['geometry'])>0;checks['actual_rgb_gradient']=json.loads((report/'render-gradient.json').read_text())['passed'];checks['component_tests']=json.loads((report/'unit-tests.json').read_text())['passed'];checks['probe_selected_before_targets']=not obs['probe_selection']['probe_targets_read'];checks['test_not_used_for_probe_selection']=not obs['test_target_read_before_selection'];checks['new_view_scale_ambiguity']=obs['new_view_max_rgb_difference']==0
for kind in ['learned','analytic']:
 fit=base/f'selected-{kind}-fit.json';seal=hashlib.sha256(fit.read_bytes()).hexdigest();checks[kind+'_seals']=all(json.loads((base/f'final-{kind}'/(c+'.json')).read_text())['fit_sha256']==seal for c in cases)
audit=dict(passed=all(checks.values()),checks=checks,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'scripts/solver_driven_gaussians').glob('*.py'))},scope='Integrity/numerical checks, not independent replication, calibrated uncertainty, or real-world certification. No sub-agent audit was performed in this stage.')
(report/'final-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit['checks'],indent=2));assert audit['passed']
