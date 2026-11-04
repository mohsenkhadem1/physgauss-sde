"""Exposed development audit; shared scoring identities, unequal archive fit budgets."""
import json
from pathlib import Path
import numpy as np
from scripts.gaussian_boundary_flow_v02.object_score import geometry_score,sha
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import load_softvt_evaluation
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.softvt_score import target_observation_ids
out=Path('reports/gaussian-boundary-flow-v02');results={};sources={}
for case,file in [('object-soft-task1-demo0','softvt-package-pilot-scores.json'),('single_push_sloth','push-sloth-package-pilot-scores.json')]:
 rows={}
 for label,path in [('learned',out/file),('retrieval',Path('runs/gaussian-boundary-flow-v02-retrieval')/case/'scores.json')]:
  d=json.loads(path.read_text());sources[str(path)]=sha(path)
  for name,row in d['methods'].items():rows[label+'/'+name]=row.get('incoming_v22',row)['summary']
 results[case]={'rows':rows,'scope':'Same package, selected prefix observations, full suffix, geometry and incoming evaluation mask; learned and nearest-bank selection share same bank. Pure baselines use same observed prefix budget.'}
case='object-soft-task1-demo0';data=load_package(Path('runs/gaussian-boundary-flow-v02-inputs')/case)
eval=load_softvt_evaluation('object-soft',1,0);ids=target_observation_ids(data,eval);target=eval['points'][:,ids];mask=eval['mask'][:,ids];archive={}
for category,tag in [('fits','primary'),('fits','primary-no-field'),('fits','physical-release'),('fits','physical-release-no-field'),('fits','adaptive'),('fits','adaptive-no-field'),('native','primary')]:
 folder=Path('runs/external-benchmark-v11')/category/tag/case;path=folder/'prediction.npz'
 if not path.exists():continue
 with np.load(path,allow_pickle=False)as f:
  assert np.array_equal(f['nodes'],data['support_geometry']),str(path)
  pred=f['predicted'][:,ids]
 meta=json.loads((folder/'fit.json').read_text());assert meta['fit_end']==72
 archive[category+'/'+tag]={'scores':geometry_score(pred,target,mask,72,120)['summary'],'fit_frames':meta.get('selected_frames','all72prefixframes in native train.pkl'),'fit_nodes':241,'fit_observations':48,'prediction_sha256':sha(path),'scope':meta.get('scope','Archived external-FEM adapter; inspect recipe, no identical training budget claim')}
results[case]['archive']=archive
results[case]['archive_comparability']='Exactly same48 observed identities/order, suffix72:120, all-valid mask. Archived mechanics use241nodes/36fittingframes; native72prefixframes. New64nodes/12frames. Temporal budgets and physical models differ; archive is contextual, not matched-budget superiority evidence.'
results['conclusions']=['SoftVT learned forecasts improve coordinate motion versus persistence/velocity, but shape error remains worse than near-rigid persistence. Compare nearest retrieval before attributing benefit to neural inference.','Sloth pushing remains unsupported: compare against velocity, not only persistence.','SoftVT top8 retrieval has one frozen member failure at frame110; both ensemble channels retain all48 suffix frames as failures. No member replacement or exclusion.','No retrieval posterior coverage: top8 diversity is descriptive, not calibrated uncertainty. All cases historically exposed development.']
result={'results':results,'score_sources_sha256':sources}
(out/'retrieval-comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
lines=[]
for case in ['object-soft-task1-demo0','single_push_sloth']:
 lines.append(case+' — coordinate / rigid-aligned shape RMSE(mm); failed suffix frames')
 for name,s in results[case]['rows'].items():lines.append(f"{name}: {s['coordinate_rmse_mm']} / {s['aligned_shape_rmse_mm']}; failed={s['failed_frames']}")
 if 'archive' in results[case]:
  lines.append(results[case]['archive_comparability'])
  for name,item in archive.items():
   s=item['scores'];lines.append(f"ARCHIVE {name}: {s['coordinate_rmse_mm']} / {s['aligned_shape_rmse_mm']}; failed={s['failed_frames']}")
lines.extend(results['conclusions']);(out/'retrieval-comparison.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
