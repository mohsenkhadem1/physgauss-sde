"""Separate CPU input-only forecast commit, then explicit reference-scoring stage."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.stable_zero_control import guard,PLAN,OUT,ROOT,REPORT,sha
from scripts.compositional_neural_mechanics.oracle_campaign import runtime_records,OUT as ORIGINAL,FREEZE,validate_value
from scripts.compositional_neural_mechanics.direct_baseline import load_checkpoint
from scripts.compositional_neural_mechanics.feature_pack import load_frozen_laws
from scripts.compositional_neural_mechanics.forward_evaluation import direct_forward,score_prediction
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.dataset_runner import write,npz
COMMIT=OUT/'predictions-commit.json'

def predict():
 if COMMIT.exists():raise ValueError('No forecast overwrite')
 control=guard();plan,records=runtime_records();q=np.asarray(plan['material_query']);torch.set_num_threads(1)
 laws=load_frozen_laws(*(ROOT/plan['checkpoints'][k]['path']for k in ('bulk','interface')),{k:plan['checkpoints'][k]['sha256']for k in ('bulk','interface')},'cpu')
 models={k:load_checkpoint(ROOT/v['path'])[0]for k,v in control['variants'].items()};rows=[]
 for record in records:
  for action in record['actions']:
   for name,model in models.items():
    row=dict(scene_id=record['scene']['name'],cohort=record['role'],action=action['name'],method=name)
    try:
     value=direct_forward(record['scene'],action['pose'],q,predictor=model,nodes=plan['node_material'],bulk_law=laws[0],interface_law=laws[1],probes=plan['probes'],checkpoint_sha256={k:plan['checkpoints'][k]['sha256']for k in ('bulk','interface')});validate_value(value,q)
     path=OUT/'predictions'/record['scene']['name']/action['name']/(name+'.npz');npz(path,material_query=q,displacement=value['displacement'],wrenches=value['wrenches'])
     row.update(state='complete',artifact=str(path.relative_to(ROOT)),sha256=sha(path))
    except Exception as error:row.update(state='failed',error=repr(error))
    rows.append(row)
 guard();runtime_records()
 write(COMMIT,dict(control_plan_sha256=sha(PLAN),parent_freeze_sha256=sha(FREEZE),records_sha256=sha(ORIGINAL/'reserved-records.json'),expected=96,rows=rows,reference_access=False))

def score():
 control=guard();plan,records=runtime_records();commit=json.loads(COMMIT.read_text())
 if commit['control_plan_sha256']!=sha(PLAN) or commit['parent_freeze_sha256']!=sha(FREEZE) or commit['records_sha256']!=sha(ORIGINAL/'reserved-records.json'):raise ValueError('Forecast binding changed')
 expected={(r['scene']['name'],a['name'],m)for r in records for a in r['actions']for m in control['variants']}
 rows=commit['rows']
 if len(rows)!=96 or {(r['scene_id'],r['action'],r['method'])for r in rows}!=expected:raise ValueError('All96 terminal forecasts required before reference access')
 for row in rows:
  if row['state'] not in ('complete','failed'):raise ValueError('Nonterminal forecast')
  if row['state']=='complete' and sha(ROOT/row['artifact'])!=row['sha256']:raise ValueError('Forecast changed')
 evidence={str(COMMIT.relative_to(ROOT)):sha(COMMIT)}
 for row in rows:
  if row['state']!='complete':continue
  reference=ORIGINAL/row['scene_id']/row['action']/'reference.npz'
  try:
   scene=next(r['scene']for r in records if r['scene']['name']==row['scene_id'])
   with np.load(reference,allow_pickle=False)as ref,np.load(ROOT/row['artifact'],allow_pickle=False)as pred:
    validate_value(ref,plan['material_query'])
    if not np.array_equal(ref['material_query'],np.asarray(plan['material_query'])):raise ValueError('Reference query mismatch')
    row['scores']=score_prediction(pred,ref,GeometrySpec(**scene['geometry']).map(np.asarray(plan['material_query'])))
   evidence[str(reference.relative_to(ROOT))]=sha(reference)
  except Exception as error:row['scoring_error']=repr(error)
 summaries=[]
 for cohort in ('reserved_composition','reserved_geometry'):
  for method in control['variants']:
   selected=[r for r in rows if r['cohort']==cohort and r['method']==method];valid=[r for r in selected if 'scores'in r]
   entry=dict(cohort=cohort,method=method,expected_commands=24,scored_commands=len(valid),complete=len(valid)==24)
   if len(valid)==24:entry['object_mean_scores']={k:float(np.mean([r['scores'][k]for r in valid]))for k in valid[0]['scores']}
   summaries.append(entry)
 write(REPORT/'stable-zero-control-results.json',dict(scope='Post-freeze source-audit control, both variants retained beside unchanged original five methods.',control_plan_sha256=sha(PLAN),evidence_sha256=evidence,rows=rows,summaries=summaries))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--predict',action='store_true');p.add_argument('--score',action='store_true');a=p.parse_args()
 if a.predict and not a.score:predict()
 elif a.score and not a.predict:score()
 else:p.error('Exactly one explicit stage required')
