"""Pinned custom DeepSets train/validation campaign, never reserved inputs.

Feature caching may proceed incrementally from verified completed records. Model
training is gated on ALL348 successes; no failed or absent records are dropped.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,ast,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.dataset_runner import verify_complete,validate as validate_dataset,write,npz
from scripts.compositional_neural_mechanics.feature_pack import load_frozen_laws,default_probes,feature_contract,load_labelled_record,batch_records
from scripts.compositional_neural_mechanics.direct_baseline import train_direct
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/compositional-neural-mechanics-v16'
BASE=ROOT/'runs/compositional-neural-mechanics-v16'
PLAN=REPORT/'direct-training-plan.json'
OUT=BASE/'direct-baseline-campaign'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_inventory():
 pending=[Path(__file__).resolve()];seen=set()
 while pending:
  path=pending.pop()
  if path in seen:continue
  seen.add(path)
  for node in ast.walk(ast.parse(path.read_text())):
   names=[x.name for x in node.names]if isinstance(node,ast.Import)else [node.module]if isinstance(node,ast.ImportFrom)and node.module else []
   for name in names:
    if name.startswith('scripts.'):
     candidate=ROOT/(name.replace('.','/')+'.py')
     if candidate.exists()and candidate not in seen:pending.append(candidate)
 return {str(p.relative_to(ROOT)):sha(p)for p in sorted(seen)}
def grid(n):return np.stack(np.meshgrid(*([np.linspace(0,1,n)]*3),indexing='ij'),-1).reshape(-1,3)
def membership(dataset):
 roles={role:[r['scene']['name']for r in dataset['scenes']if r['role']==role]for role in ('train','validation')}
 if len(roles['train'])!=54 or len(roles['validation'])!=4 or len(dataset['scenes'])!=58:raise ValueError('Exact54train/4validation cohort required')
 if any(len(r['actions'])!=6 for r in dataset['scenes']):raise ValueError('Six actions required for every declared scene')
 if not np.array_equal(np.asarray(dataset['material_query']),grid(7)):raise ValueError('Fixed7³ material query grid required')
 return roles

def prepare(dataset_plan=REPORT/'training-data-plan.json',dataset_output=BASE/'training-data-n4',bulk=BASE/'scale-separated-prototype/shared-law.pt',interface=BASE/'interface-wide-range-warm/shared-interface.pt'):
 if PLAN.exists():raise ValueError('Preserve prior direct training declaration')
 dataset=json.loads(Path(dataset_plan).read_text());validate_dataset(dataset);roles=membership(dataset)
 checkpoints={k:dict(path=str(Path(p).resolve().relative_to(ROOT)),sha256=sha(p))for k,p in [('bulk',bulk),('interface',interface)]}
 plan=dict(scope='Custom geometry-conditioned DeepSets supervised baseline; not authors MeshGraphNet. Alloracle descriptors and frozen-law numeric fingerprints shared withmechanics.',dataset_plan=str(Path(dataset_plan).resolve().relative_to(ROOT)),dataset_plan_sha256=sha(dataset_plan),dataset_output=str(Path(dataset_output).resolve().relative_to(ROOT)),checkpoints=checkpoints,source_sha256=source_inventory(),node_material=grid(5).tolist(),material_query=grid(7).tolist(),split_manifest=roles,expected_records={'train':324,'validation':24,'all':348},probes=default_probes(),training=dict(steps=6000,batch_size=16,learning_rate=.001,seed=926701,device='cuda:0',maximum_seconds=1800),checkpoint_selection='Final fixed6000-step iterate only; no test or validation selection',normalization='Fit node/global/query features and displacement/wrench targets on324trainrecords only; preserve repeatedsceneIDs acrossactions',runtime_versions=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__))
 write(PLAN,plan);return plan

def guard(plan_path=PLAN):
 plan=json.loads(Path(plan_path).read_text())
 if plan['source_sha256']!=source_inventory():raise ValueError('Direct training source changed')
 dataset_path=ROOT/plan['dataset_plan']
 if sha(dataset_path)!=plan['dataset_plan_sha256']:raise ValueError('Dataset plan changed')
 dataset=json.loads(dataset_path.read_text());validate_dataset(dataset)
 if membership(dataset)!=plan['split_manifest']:raise ValueError('Dataset membership changed')
 for item in plan['checkpoints'].values():
  if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen law checkpoint changed')
 if plan['training']!=dict(steps=6000,batch_size=16,learning_rate=.001,seed=926701,device='cuda:0',maximum_seconds=1800):raise ValueError('Fixed training budget changed')
 return plan,dataset

def records(plan_path=PLAN,*,incremental=False):
 torch.set_num_threads(1);plan,dataset=guard(plan_path);pin=sha(plan_path);dataset_output=ROOT/plan['dataset_output']
 checkpoints={k:v['sha256']for k,v in plan['checkpoints'].items()};bulk,interface=load_frozen_laws(*(ROOT/plan['checkpoints'][k]['path']for k in ('bulk','interface')),checkpoints,device='cpu')
 contract=feature_contract(plan['probes'],checkpoints);ready=[];missing=[];evidence={str(Path(plan_path).resolve().relative_to(ROOT)):pin};rows=[]
 for entry in dataset['scenes']:
  scene=entry['scene'];role=entry['role']
  for action in entry['actions']:
   identity=dict(scene_id=scene['name'],split=role,action=action['name'],prescribed_pose=action['pose'],n=dataset['n'],order=dataset['order'])
   source=dataset_output/'scenes'/scene['name']/action['name'];complete=verify_complete(source,plan['dataset_plan_sha256'],dataset['source_sha256'],identity)
   if complete is None:
    missing.append(identity);rows.append(dict(identity=identity,state='missing_or_failed'));continue
   label=source/'labels.npz';cp=source/'complete.json';evidence.update({str(p.relative_to(ROOT)):sha(p)for p in (label,cp)})
   cache=OUT/'features'/scene['name']/action['name'];cache.mkdir(parents=True,exist_ok=True);target=cache/'features.npz';meta_path=cache/'complete.json'
   expected=dict(plan_sha256=pin,identity=identity,label_sha256=sha(label),label_complete_sha256=sha(cp),feature_contract=contract)
   if meta_path.exists():
    meta=json.loads(meta_path.read_text())
    if any(meta.get(k)!=v for k,v in expected.items())or sha(target)!=meta['artifact_sha256']:raise ValueError('Changed cached feature commitment')
   else:
    record=load_labelled_record(scene,dict(scene_id=scene['name'],split=role,prescribed_pose=action['pose']),label,expected_sha256=sha(label),node_material=plan['node_material'],bulk_law=bulk,interface_law=interface,probes=plan['probes'],checkpoint_sha256=checkpoints,allowed_scenes_by_split=plan['split_manifest'])
    npz(target,**{k:record[k]for k in ('nodefeatures','globalfeatures','queryfeatures','displacement','wrenches')})
    write(meta_path,dict(**expected,artifact_sha256=sha(target)))
   with np.load(target,allow_pickle=False)as data:
    if set(data.files)!={'nodefeatures','globalfeatures','queryfeatures','displacement','wrenches'}or any(not np.isfinite(data[k]).all()for k in data.files):raise ValueError('Invalid cached arrays')
    record={k:data[k].copy()for k in data.files}
   if not np.array_equal(record['nodefeatures'][:,:3],np.asarray(plan['node_material']))or not np.array_equal(record['queryfeatures'][:,:3],np.asarray(plan['material_query'])):raise ValueError('Cached public point grid differs from declared grid')
   record.update(scene_id=scene['name'],split=role,feature_contract=contract);ready.append(record)
   evidence.update({str(p.relative_to(ROOT)):sha(p)for p in (target,meta_path)});rows.append(dict(identity=identity,state='complete',feature_sha256=sha(target)))
 guard(plan_path)
 for path,digest in evidence.items():
  if sha(ROOT/path)!=digest:raise ValueError('Feature evidence changed during preparation')
 write(OUT/'feature-progress.json',dict(plan_sha256=pin,expected=348,complete=len(ready),missing_or_failed=len(missing),rows=rows,evidence_sha256=evidence))
 if not incremental and missing:raise ValueError(f'All348records required: {len(missing)} missing/failed; no filtering')
 if not incremental and len(ready)!=348:raise ValueError('Unexpected total record count')
 return ready,contract,evidence

def run(plan_path=PLAN):
 plan,dataset=guard(plan_path);checkpoint=OUT/'direct-baseline.pt'
 if checkpoint.exists():raise ValueError('Preserve trained checkpoint')
 ready,contract,evidence=records(plan_path,incremental=False)
 train,_=batch_records([r for r in ready if r['split']=='train']);validation,_=batch_records([r for r in ready if r['split']=='validation'])
 if len(train['scene_ids'])!=324 or len(validation['scene_ids'])!=24:raise ValueError('Training/validation count mismatch')
 start=time.monotonic();wrapped,metadata=train_direct(train,validation,split_manifest=plan['split_manifest'],feature_contract=contract,checkpoint_path=checkpoint,**plan['training'])
 guard(plan_path)
 with torch.no_grad():
  totals=np.zeros(3);count=0
  for first in range(0,len(train['scene_ids']),16):
   sl=slice(first,first+16);d,w=wrapped(*(train[k][sl].to(plan['training']['device'])for k in ('nodefeatures','globalfeatures','queryfeatures')))
   du=d-train['displacement'][sl].to(d.device);dw=w-train['wrenches'][sl].to(w.device);batch=len(d)
   totals+=batch*np.array([float(du.square().mean()),float(dw[...,:3].square().mean()),float(dw[...,3:].square().mean())]);count+=batch
  metadata['final_training_metrics']=dict(zip(('displacement_coordinate_rmse_m','force_coordinate_rmse_N','torque_coordinate_rmse_Nm'),np.sqrt(totals/count).tolist()))
 if any(sha(ROOT/path)!=digest for path,digest in evidence.items()):raise ValueError('Training evidence changed')
 write(REPORT/'direct-training-results.json',dict(scope=plan['scope'],plan_sha256=sha(plan_path),checkpoint_sha256=sha(checkpoint),source_sha256=plan['source_sha256'],input_sha256=evidence,training=metadata,train_records=324,validation_records=24,train_scenes=54,validation_scenes=4,total_wall_seconds=time.monotonic()-start,reserved_access=False))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--prepare-features',action='store_true');parser.add_argument('--execute-training',action='store_true');parser.add_argument('--plan',type=Path,default=PLAN);args=parser.parse_args()
 if args.prepare:prepare()
 elif args.prepare_features:records(args.plan,incremental=True)
 elif args.execute_training:run(args.plan)
 else:guard(args.plan)
if __name__=='__main__':main()
