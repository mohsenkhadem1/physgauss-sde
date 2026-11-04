"""Replay exposed-sloth metric audit only; no training or fresh data access."""
import json,pickle,hashlib
from pathlib import Path
import numpy as np
from scripts.external_benchmark.data import interpolate_matrix,phystwin_position_mask
from scripts.gaussian_boundary_flow_v02.object_score import geometry_score,expand_points
R=Path.cwd();out=R/'reports/gaussian-boundary-flow-v02';case='single_lift_sloth'
oldroot=R/'runs/third_party/PhysTwin/data/different_types'/case
newroot=R/'runs/editable-scene-v23/phystwin-full-rgb/data/different_types'/case
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(oldroot/'final_data.pkl')==sha(newroot/'final_data.pkl')
with (oldroot/'final_data.pkl').open('rb')as f:raw=pickle.load(f)
target=raw['object_points']@np.diag([1.,-1.,-1.]);initial=target[0];mask=phystwin_position_mask(raw,len(target));official=raw['object_visibilities'].astype(bool)
start,end=59,85;archived=json.loads((R/'reports/real-transfer-v22/scores.json').read_text());archive=next(r for r in archived['rows']if r['case']==case)
rows={};hashes={};budgets={}
for name in ['neural','analytic','official_phystwin']:
 if name=='official_phystwin':
  path=R/'runs/third_party/PhysTwin/reference/experiments'/case/'inference.pkl'
  with path.open('rb')as f:prediction=np.asarray(pickle.load(f))@np.diag([1.,-1.,-1.])
  prediction=prediction[:,:len(initial)]
  budgets[name]={'fit_frames':'author official prefix, exact sampled fitting frames not verified','point_ids':'author preprocessing; training observation budget not verified','nodes':'native representation'}
 else:
  directory=R/'runs/real-transfer-v19/fits/v22-corrected-mask240'/case/name;path=directory/'prediction.npz'
  complete=json.loads((directory/'complete.json').read_text());assert sha(path)==complete['prediction_sha256']
  with np.load(path,allow_pickle=False)as f:prediction=expand_points(f['predicted'],f['nodes'],interpolate_matrix(f['nodes'],initial),initial)
  plan=json.loads((directory/'plan.json').read_text());budgets[name]={'fit_frames':plan['selected_frames'],'point_ids':{'range':[0,len(initial)],'count':len(initial)},'nodes':240,'fit_mask':'incoming-v22'}
 hashes[str(path.relative_to(R))]=sha(path)
 rows[name]={'incoming_v22':geometry_score(prediction,target,mask,start,end)['summary'],'official_visibility':geometry_score(prediction,target,official,start,end)['summary'],'archived_official':archive['results'][name]['geometry']}
 assert abs(rows[name]['official_visibility']['coordinate_rmse_mm']-rows[name]['archived_official']['coordinate_rmse_mm'])<1e-5
for file in ['sloth-object-scores.json','sloth-geometry-oracle-scores.json']:
 result=json.loads((out/file).read_text());hashes[str((out/file).relative_to(R))]=sha(out/file)
 for name,scores in result['methods'].items():
  key=file.replace('-scores.json','')+'/'+name
  rows[key]={m:scores[m]['summary']for m in ['incoming_v22','official_visibility']}
  budgets[key]={'fit_frames':result['baseline_budget']['frames'],'point_ids':result['baseline_budget']['point_ids'],'nodes':result['nodes'],'fit_mask':'incoming-v22'}
result={'case':case,'verdict':'Evaluation coordinates, point identities, suffix frames and mask can be harmonized; training budgets and representations are NOT matched. No controlled superiority claim.',
'evaluation':{'interval':[start,end],'point_ids':{'range':[0,len(initial)],'count':len(initial)},'incoming_mask':'visibility[t] AND motions_valid[t-1], finite reference','same_raw_file_sha256':sha(oldroot/'final_data.pkl'),'aggregation':'equal frame coordinate RMSE, same object_score implementation','manual_scores_compared':False,'manual_reason':'Archived nearest-point mapping uses each method predicted initial shape; new scorer uses fixed observed_initial, so archived manual scores not directly comparable.'},
'rows':rows,'training_budgets':budgets,'archived_position_ridge':{'metrics_official_visibility':archive['results']['position_ridge']['geometry'],'budget':'all59 prefix frames and all5203 observations, unlike new12frames/48points','rescored_incoming':False},
'best_old':{'coordinate_among_old_physics_models_official':'analytic27.916959mm vs neural28.034263mm','coordinate_including_native_official':'official_phystwin22.661340mm','coordinate_including_ridge_official':'position_ridge22.258110mm','caution':'Descriptive case-level archive ranking, not matched-budget method selection or evidence on untouched data'},
'next_softvt_case':{'case':'object-soft-task1-demo0','reason':'Most informative exposed stress case:3/4 readiness worlds completed, seed913 failed frame52; larger shape error2.6–2.9mm and strong compression expose support/rest/continuation shortcomings before scaling. Score failed forecasts explicitly. Use object-soft-task2-demo0 as secondary numerical control (4/4 completed) rather than substituting it to hide failures.','scope':'Expose prefix only to prediction worker; suffix only after inference seal. All are historical development cases.'},'source_sha256':hashes}
(out/'archived-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
lines=[result['verdict'],'Same full5203point identities; suffix59:85. Below incoming-v22 / official visibility coordinate RMSE(mm), shape RMSE(mm):']
for name,r in rows.items():
 a=r['incoming_v22'];b=r['official_visibility'];lines.append(f"{name}: coordinate {a['coordinate_rmse_mm']:.3f}/{b['coordinate_rmse_mm']:.3f}; shape {a['aligned_shape_rmse_mm']:.3f}/{b['aligned_shape_rmse_mm']:.3f}")
lines+=['Archived analytic/neural fit18frames and5203points with240nodes; new methods fit12frames and48points with64nodes. Native exact training sampling is unverified. Native and old method outputs were re-scored on exactly the new target identities and masks; official-only scores reproduce archived values within1e-5mm.','Archive position-ridge22.258mm uses all59prefixframes/all5203points; it is not an equal-budget baseline for new12x48inputs.','Manual errors are excluded from direct comparison because initial matching conventions differ.','Next informative exposed SoftVT pilot: object-soft-task1-demo0, including its known numerical failure region. Secondary control: object-soft-task2-demo0. No fresh cases touched.']
(out/'archived-comparison.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
