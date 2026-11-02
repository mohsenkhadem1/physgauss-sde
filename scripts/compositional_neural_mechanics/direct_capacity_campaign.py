"""Validation-motivated capacity control; both budgets retained, no selection."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.train_direct_campaign import guard as original_guard
from scripts.compositional_neural_mechanics.direct_baseline import train_direct
from scripts.compositional_neural_mechanics.feature_pack import batch_records
from scripts.compositional_neural_mechanics.dataset_runner import write
from scripts.compositional_neural_mechanics_reporting.legacy_dependency_audit import closure,sha
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/compositional-neural-mechanics-v16';BASE=ROOT/'runs/compositional-neural-mechanics-v16'
PLAN=REPORT/'direct-capacity-plan.json';OUT=BASE/'direct-capacity-campaign'

def prepare():
 if PLAN.exists():raise ValueError('Preserve declared capacity plan')
 original,_=original_guard();resultpath=REPORT/'direct-training-results.json';result=json.loads(resultpath.read_text())
 inventory=closure([Path(__file__).resolve()]);inputs=dict(result['input_sha256'])
 for path in (resultpath,REPORT/'direct-training-plan.json',BASE/'direct-baseline-campaign/direct-baseline.pt'):inputs[str(path.relative_to(ROOT))]=sha(path)
 for name,digest in inputs.items():
  if sha(ROOT/name)!=digest:raise ValueError('Original feature/data evidence changed')
 plan=dict(scope=__doc__,motivation='Original four-scene validation results were inspected. This is an explicitly validation-motivated capacity check, not a blind model-choice experiment. Report both6000 and30000-step variants; do not choose based on favorable comparison.',original_plan_sha256=sha(REPORT/'direct-training-plan.json'),source_sha256=inventory,input_sha256=inputs,split_manifest=original['split_manifest'],architecture='Unchanged DeepSets width128,3hidden layers; unchangedfeaturecontract',training=dict(steps=30000,batch_size=16,learning_rate=.001,seed=926701,device='cuda:0',maximum_seconds=900),initialization='From scratch, same initialization seed; cosine schedule spans30000steps, no warm start.',checkpoint_selection='Final fixed-budget iterate only, irrespective ofvalidation score',checkpoint=str((OUT/'direct-baseline.pt').relative_to(ROOT)),data='Exactly same324training/24validation records and training-only normalization; no new labels or reserved access.')
 write(PLAN,plan)
 return plan

def guard():
 plan=json.loads(PLAN.read_text());original,dataset=original_guard()
 if plan['source_sha256']!=closure([Path(__file__).resolve()]):raise ValueError('Capacity source changed')
 for path,digest in plan['input_sha256'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Committed capacity input changed')
 if plan['training']!=dict(steps=30000,batch_size=16,learning_rate=.001,seed=926701,device='cuda:0',maximum_seconds=900):raise ValueError('Fixed capacity budget changed')
 if plan['split_manifest']!=original['split_manifest']:raise ValueError('Capacity scene split changed')
 return plan,dataset

def load_records(dataset):
 rows=[]
 for entry in dataset['scenes']:
  for action in entry['actions']:
   folder=BASE/'direct-baseline-campaign/features'/entry['scene']['name']/action['name'];meta=json.loads((folder/'complete.json').read_text());path=folder/'features.npz'
   if sha(path)!=meta['artifact_sha256']:raise ValueError('Feature cache changed')
   with np.load(path,allow_pickle=False)as data:row={key:data[key].copy()for key in data.files}
   row.update(split=entry['role'],scene_id=entry['scene']['name'],feature_contract=meta['feature_contract']);rows.append(row)
 if len(rows)!=348:raise ValueError('No missing cases allowed')
 return rows

def training_metrics(model,pack,device):
 total=np.zeros(3);count=0
 with torch.no_grad():
  for first in range(0,len(pack['scene_ids']),16):
   sl=slice(first,first+16);d,w=model(*(pack[k][sl].to(device)for k in ('nodefeatures','globalfeatures','queryfeatures')))
   du=d-pack['displacement'][sl].to(device);dw=w-pack['wrenches'][sl].to(device);batch=len(d)
   total+=batch*np.array([float(du.square().mean()),float(dw[...,:3].square().mean()),float(dw[...,3:].square().mean())]);count+=batch
 return dict(zip(('displacement_coordinate_rmse_m','force_coordinate_rmse_N','torque_coordinate_rmse_Nm'),np.sqrt(total/count).tolist()))

def run():
 plan,dataset=guard();rows=load_records(dataset);train,contract=batch_records([r for r in rows if r['split']=='train']);validation,other=batch_records([r for r in rows if r['split']=='validation'])
 if contract!=other or len(train['scene_ids'])!=324 or len(validation['scene_ids'])!=24:raise ValueError('Exact matching contract/cohort required')
 started=time.monotonic();model,metadata=train_direct(train,validation,split_manifest=plan['split_manifest'],feature_contract=contract,checkpoint_path=ROOT/plan['checkpoint'],**plan['training'])
 metadata['final_training_metrics']=training_metrics(model,train,plan['training']['device']);guard()
 result=dict(scope=plan['scope'],motivation=plan['motivation'],plan_sha256=sha(PLAN),checkpoint_sha256=sha(ROOT/plan['checkpoint']),source_sha256=plan['source_sha256'],input_sha256=plan['input_sha256'],training=metadata,feature_contract=contract,train_records=324,validation_records=24,total_wall_seconds=time.monotonic()-started,reserved_access=False,comparison_policy='Retain original6000-step andthis30000-step variants as separate reported baselines, no selection')
 write(REPORT/'direct-capacity-results.json',result)
 print(json.dumps(dict(training=metadata['final_training_metrics'],validation=metadata['validation'],seconds=metadata['seconds']),indent=2),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',action='store_true');a=parser.parse_args()
 if a.prepare:prepare()
 elif a.execute:run()
 else:guard()
