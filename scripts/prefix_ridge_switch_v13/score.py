"""Evaluator stage: access suffix targets only after all148 attempts are terminal."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.prefix_ridge_switch_v13.run import BASE,REPORT,PARENT,read,write,sha,guard
METHOD='ridge_switch_selected'
COMPARATORS=('model_selection_selected','model_selection_rigid','rigid_scene_state','prefix_calibrated_rigid','position_ridge','official_phystwin')

def terminal_gate():
 p,_=guard();freeze=read(REPORT/'method-freeze.json');attempts=read(BASE/'attempts.json')
 if attempts['protocol_sha256']!=sha(REPORT/'protocol.json') or attempts['method_freeze_sha256']!=sha(REPORT/'method-freeze.json'):raise RuntimeError('Terminal manifest freeze mismatch')
 if len(p['cases'])!=148 or len(attempts['cases'])!=148:raise RuntimeError('All148 forecast attempts required before evaluator access')
 indexed={row['case']['name']:row for row in attempts['cases']}
 if len(indexed)!=148:raise RuntimeError('Duplicate terminal cases')
 for case in p['cases']:
  row=indexed.get(case['name'])
  if row is None or row['case']!=case or row['state'] not in ('complete','failed','timeout'):raise RuntimeError('Nonterminal or undeclared case')
  folder=BASE/'forecasts'/case['cohort']/case['name']
  for name,key in [('prediction.npz','prediction_sha256'),('metadata.json','metadata_sha256')]:
   path=folder/name;expected=row.get(key)
   if expected is not None and (not path.is_file() or sha(path)!=expected):raise RuntimeError('Terminal output changed')
   if row['state']=='complete' and expected is None:raise RuntimeError('Complete case lacks forecast artifacts')
   if expected is None and path.exists():raise RuntimeError('Unbound terminal artifact')
 inputs=freeze.get('input_sha256')
 if not isinstance(inputs,dict) or not inputs:raise RuntimeError('Frozen input manifest required')
 for rel,expected in inputs.items():
  path=Path(rel)
  if path.is_absolute() or '..' in path.parts or sha(ROOT/path)!=expected:raise RuntimeError('Frozen input changed: '+rel)
 for case in p['cases']:
  parent=PARENT/'model-selection-evaluation'/case['cohort']/case['name']
  for name in ('metrics.json','predictions.npz'):
   path=parent/name
   if str(path.relative_to(ROOT)) not in inputs:raise RuntimeError('Archived comparison not pinned: '+str(path))
 return p,indexed,sha(BASE/'attempts.json')

def missing_summary(data):
 total=len(data['times'])-data['train_end']
 return dict(summary=dict(total_frames=total,valid_frames=0,failed_frames=total,coordinate_rmse_mm=None,aligned_shape_rmse_mm=None,manual_track_mean_mm=None),unavailable='Terminal attempt has no auditable forecast; retained as incomplete')

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true',required=True,help='Explicitly run evaluator after frozen forecasts');parser.parse_args()
 p,attempts,attempts_sha=terminal_gate()
 # Evaluator-capable imports occur only after every terminal/input/source check.
 from scripts.contact_benchmark_v12.data import load
 from scripts.external_benchmark.evaluate import expand_prediction
 from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
 from scripts.contact_benchmark_v12.experiment import finite_metadata
 rows=[];evidence={}
 for case in p['cases']:
  name,cohort=case['name'],case['cohort'];attempt=attempts[name];folder=BASE/'forecasts'/cohort/name;parent=PARENT/'model-selection-evaluation'/cohort/name
  data=load(case['spec']);archived=read(parent/'metrics.json')
  if archived['name']!=name or archived['cohort']!=cohort or archived['dataset']!=data['dataset'] or archived['interval']!=[data['train_end'],len(data['times'])]:raise RuntimeError('Archived comparator identity/window mismatch')
  results={m:archived['results'][m] for m in COMPARATORS if m in archived['results']};fallback_exact=None;metadata={}
  if attempt.get('prediction_sha256') is not None and attempt.get('metadata_sha256') is not None:
   metadata=read(folder/'metadata.json')
   if metadata['case']!=case:raise RuntimeError('Forecast case mismatch')
   with np.load(folder/'prediction.npz',allow_pickle=False) as f:prediction=f['predicted'].copy()
   if prediction.shape!=(len(data['times']),len(data['nodes']),3):raise RuntimeError('Saved forecast shape mismatch')
   if not metadata.get('switched',False):
    with np.load(parent/'predictions.npz',allow_pickle=False) as f:old=f['model_selection_selected'].copy()
    fallback_exact=prediction.dtype==old.dtype and prediction.shape==old.shape and prediction.tobytes()==old.tobytes()
    if not fallback_exact:raise RuntimeError('No-switch forecast is not byte-exact archived array')
   results[METHOD]=safe_score(data,expand_prediction(data,prediction),data['train_end'],len(data['times']))
   for file in ('prediction.npz','metadata.json'):evidence[str((folder/file).relative_to(ROOT))]=sha(folder/file)
  else:results[METHOD]=missing_summary(data)
  row=finite_metadata(dict(name=name,cohort=cohort,dataset=data['dataset'],case=case,attempt_state=attempt['state'],attempt_error=attempt.get('error'),switched=bool(metadata.get('switched',False)),fallback_exact=fallback_exact,interval=[data['train_end'],len(data['times'])],results=results,missing_comparators=[m for m in COMPARATORS if m not in results],scope='All cases historically inspected; exploratory guarded empirical ridge; not coupled transfer'))
  write(BASE/'evaluation'/cohort/name/'metrics.json',row);rows.append(row)
  for file in ('metrics.json','predictions.npz'):evidence[str((parent/file).relative_to(ROOT))]=sha(parent/file)
 if sha(BASE/'attempts.json')!=attempts_sha:raise RuntimeError('Forecast manifest changed during scoring')
 terminal_gate()
 write(REPORT/'scores.json',dict(scope='All148 exploratory; metrics separate by dataset; fixed forecast attempts preceded evaluator access',protocol_sha256=sha(REPORT/'protocol.json'),method_freeze_sha256=sha(REPORT/'method-freeze.json'),attempts_sha256=attempts_sha,scorer_sha256=sha(Path(__file__)),cases=rows,evidence_sha256=evidence))
if __name__=='__main__':main()
