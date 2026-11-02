"""Post-freeze input-normalization control; original weights/results remain unchanged."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,copy,json
from datetime import datetime,timezone
from pathlib import Path
import torch
from scripts.compositional_neural_mechanics.direct_capacity_campaign import guard as capacity_guard,load_records
from scripts.compositional_neural_mechanics.feature_pack import batch_records
from scripts.compositional_neural_mechanics.direct_baseline import load_checkpoint
from scripts.compositional_neural_mechanics.dataset_runner import write
from scripts.compositional_neural_mechanics_reporting.legacy_dependency_audit import closure,sha
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/compositional-neural-mechanics-v16';BASE=ROOT/'runs/compositional-neural-mechanics-v16'
PLAN=REPORT/'stable-zero-control-plan.json';OUT=BASE/'stable-zero-control'

def replacement_unit(field,index):
 if field=='globalfeatures':
  if 5<=index<17:return .02 if (index-5)%6<3 else .3
  if 17<=index<20:return 9.81
 return 1.

def corrected(saved,train):
 result=copy.deepcopy(saved);changes=[]
 for field in ('nodefeatures','globalfeatures','queryfeatures'):
  x=train[field].reshape(-1,train[field].shape[-1]);mean=saved['state_dict']['normalization.'+field+'_mean']
  zero=(x==0).all(0)&(mean==0)
  for index in torch.nonzero(zero).flatten().tolist():
   key='normalization.'+field+'_std';old=float(saved['state_dict'][key][index]);value=replacement_unit(field,index)
   result['state_dict'][key][index]=value;changes.append(dict(field=field,index=index,old_std=old,new_std=value))
 # Every tensor except precisely declared std entries must remain bit-identical.
 allowed={('normalization.'+c['field']+'_std',c['index']) for c in changes}
 for key,value in saved['state_dict'].items():
  after=result['state_dict'][key]
  if key.endswith('_std') and key.startswith('normalization.'):
   mask=torch.ones_like(value,dtype=torch.bool)
   for k,i in allowed:
    if key==k:mask[i]=False
   assert torch.equal(value[mask],after[mask])
  else:assert torch.equal(value,after)
 return result,changes

def prepare():
 if PLAN.exists() or OUT.exists():raise ValueError('Preserve existing control artifacts')
 cap,dataset=capacity_guard();rows=load_records(dataset);train,contract=batch_records([r for r in rows if r['split']=='train'])
 if len(train['scene_ids'])!=324:raise ValueError('Exact train census required')
 inputs=dict(cap['input_sha256']);inputs[str((REPORT/'direct-capacity-plan.json').relative_to(ROOT))]=sha(REPORT/'direct-capacity-plan.json')
 paths={'custom_direct_stable':BASE/'direct-baseline-campaign/direct-baseline.pt','custom_direct_long_stable':BASE/'direct-capacity-campaign/direct-baseline.pt'}
 for p in paths.values():inputs[str(p.relative_to(ROOT))]=sha(p)
 plan=dict(status='prospective-source-audit-control',declared_utc=datetime.now(timezone.utc).isoformat(),scope=__doc__,motivation='Source-only audit found exactly zero training inputs acquire huge standardized values under public held commands. No reserved references or metric files read to design this control.',rule='Only exactly zero TRAIN input columns with stored mean exactly zero; pose translation .02m, rotation .3rad, gravity9.81m/s2, other structural zeros1. Positive constants, nonconstant inputs, targets and weights unchanged.',source_sha256=closure([Path(__file__).resolve(),ROOT/'scripts/compositional_neural_mechanics/stable_zero_evaluate.py']),input_sha256=inputs,train_records=324,variants={},comparison='Retain all original five methods and both stable controls, no variant selection.')
 OUT.mkdir(parents=True)
 # Prospective rule is persisted before checkpoint derivation or any reserved input access.
 write(PLAN,plan)
 for name,path in paths.items():
  original,saved=load_checkpoint(path);assert saved['feature_contract']==contract
  amended,changes=corrected(saved,train);target=OUT/(name+'.pt');torch.save(amended,target);model,_=load_checkpoint(target)
  with torch.no_grad():
   for start in range(0,324,16):
    args=[train[k][start:start+16]for k in ('nodefeatures','globalfeatures','queryfeatures')]
    before=original(*args);after=model(*args)
    if not all(torch.equal(a,b)for a,b in zip(before,after)):raise ValueError('Training predictions changed')
  plan['variants'][name]=dict(path=str(target.relative_to(ROOT)),sha256=sha(target),parent=str(path.relative_to(ROOT)),parent_sha256=sha(path),changes=changes,train_predictions_bit_identical=True)
 for p,h in inputs.items():
  if sha(ROOT/p)!=h:raise ValueError('Training evidence changed')
 plan['status']='frozen';write(PLAN,plan)
 print(json.dumps({k:v['changes']for k,v in plan['variants'].items()},indent=2))

def guard():
 plan=json.loads(PLAN.read_text())
 if plan['status']!='frozen' or len(plan['variants'])!=2:raise ValueError('Completed control freeze required')
 for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():
  if sha(ROOT/p)!=h:raise ValueError('Control source/input changed: '+p)
 for v in plan['variants'].values():
  if sha(ROOT/v['path'])!=v['sha256']:raise ValueError('Control checkpoint changed')
 return plan
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();prepare()if a.prepare else guard()
