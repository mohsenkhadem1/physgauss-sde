"""Protocol and numerical audit; requires all declared scenarios to be reported."""
from pathlib import Path
import json,hashlib,math
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/pidg-solver-forecast-v01';REPORT=ROOT/'reports/pidg-solver-forecast-v01';checks={};pending=[]
for scene in ['balls_reaction','rubber_duck','motion_kuro']:
 data=BASE/'data'/scene;groups={k:json.loads((data/(k+'.json')).read_text())['frames'] for k in ['fit','appearance_validation','forecast_validation','forecast_test']};ids={k:{f['file_path'] for f in v} for k,v in groups.items()};checks[scene+'_fit_past_only']=max(f['time'] for f in groups['fit'])<=.6;checks[scene+'_validation_future']=all(.6<f['time']<=.7 for f in groups['forecast_validation']);checks[scene+'_test_future']=all(f['time']>.7 for f in groups['forecast_test']);checks[scene+'_disjoint']=all(not ids[a]&ids[b] for a in ids for b in ids if a<b)
 for arm in ['native-base','native-refined','hybrid']:
  path=BASE/scene/arm/'run.json'
  if not path.exists() or json.loads(path.read_text()).get('status')!='training_complete':pending.append(scene+'/'+arm);continue
  run=json.loads(path.read_text());checks[scene+'_'+arm+'_prefix_source']='pidg-solver-forecast-v01' in run['arguments']['data_root'];checks[scene+'_'+arm+'_no_old_checkpoint']=not run['arguments']['warm_from'] or 'pidg-solver-forecast-v01' in run['arguments']['warm_from']
 results=REPORT/scene/'results.json'
 if results.exists():
  d=json.loads(results.read_text());checks[scene+'_finite_metrics']=all(not isinstance(v,float) or math.isfinite(v) for r in d['rows'] for v in r.values());checks[scene+'_declared_methods']=set(d['summary'])=={'native-refined','hybrid','solver','constant-velocity'};checks[scene+'_failure_counts']=all(v['completed_frames']<=v['expected_frames'] for v in d['summary'].values());seal=json.loads((BASE/scene/'evaluation/selection.json').read_text());checks[scene+'_sealed_before_test']=not seal['test_targets_read'];p=BASE/scene/'solver/best.pt';checks[scene+'_fit_hash']=(not seal['solver']['completed']) or hashlib.sha256(p.read_bytes()).hexdigest()==seal['solver']['checkpoint_sha256']
 else:pending.append(scene+'/evaluation')
for scene in ['balls_reaction','rubber_duck','motion_kuro']:
 identity=REPORT/scene/'identity.json'
 if identity.exists():checks[scene+'_initial_appearance_identity']=json.loads(identity.read_text())['passed']
 else:pending.append(scene+'/identity')
for name in ['geometry-tests','model-tests','diagnostic-tests']:
 checks[name]=json.loads((REPORT/(name+'.json')).read_text())['passed']
report=dict(passed=all(checks.values()) and not pending,checks=checks,pending=pending,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'scripts/pidg_solver_forecast').glob('*.py'))},scope='Data separation, saved-checkpoint integrity and numerical unit checks. Not independent replication or physical ground-truth certification.')
(REPORT/'audit.json').write_text(json.dumps(report,indent=2));print(dict(passed=report['passed'],failed=[k for k,v in checks.items() if not v],pending=pending))
assert all(checks.values())
