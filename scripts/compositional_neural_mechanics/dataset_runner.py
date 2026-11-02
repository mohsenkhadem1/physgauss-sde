"""Pinned analytic-teacher scene generation; train/validation/calibration only.

One externally timed process per scene, one device per worker, independent action
resets with declared continuation. No reserved sampling or scene generation here.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,ast,concurrent.futures,fcntl,hashlib,json,queue,re,signal,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.pending');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def npz(path,**arrays):
 path=Path(path);temporary=path.with_name(path.name+'.pending.npz');np.savez_compressed(temporary,**arrays);temporary.replace(path)
def source_inventory():
 pending=[Path(__file__).resolve(),ROOT/'scripts/compositional_neural_mechanics/scenes.py'];seen=set()
 while pending:
  path=pending.pop()
  if path in seen:continue
  seen.add(path)
  tree=ast.parse(path.read_text())
  for node in ast.walk(tree):
   names=[x.name for x in node.names]if isinstance(node,ast.Import)else [node.module]if isinstance(node,ast.ImportFrom)and node.module else []
   for name in names:
    if name.startswith('scripts.'):
     candidate=ROOT/(name.replace('.','/')+'.py')
     if candidate.exists()and candidate not in seen:pending.append(candidate)
 return {str(p.relative_to(ROOT)):sha(p)for p in sorted(seen)}
def validate(plan):
 if plan['n']<1 or plan['order']<2:raise ValueError('Positive FEM resolution and quadrature required')
 q=np.asarray(plan['material_query'],float)
 if q.ndim!=2 or q.shape[1]!=3 or not len(q)or not np.isfinite(q).all()or (q<0).any()or(q>1).any():raise ValueError('Material queries must lie in unit cube')
 if plan['continuation']!=[.25,.5,.75,1.]:raise ValueError('Declared four-stage continuation required')
 if not np.isfinite([plan['force_tolerance_N'],plan['per_scene_timeout_seconds'],plan['stage_timeout_seconds']]).all()or not 0<plan['force_tolerance_N']<=1e-6 or min(plan['per_scene_timeout_seconds'],plan['stage_timeout_seconds'])<=0:raise ValueError('Finite positive solver/time budgets required')
 devices=plan['devices']
 if not devices or any(d not in ('cpu','cuda:0','cuda:1','cuda:2')for d in devices):raise ValueError('Only CPU/GPU0–2 allowed')
 if any(devices.count(d)>1 for d in devices if d!='cpu'):raise ValueError('At most one scene worker per GPU')
 names=[]
 for record in plan['scenes']:
  if record['role']not in ('train','validation','calibration'):raise ValueError('Reserved/test generation forbidden in this runner')
  name=record['scene']['name'];names.append(name)
  if not re.fullmatch('[A-Za-z0-9_-]+',name):raise ValueError('Unsafe scene name')
  actions=record['actions'];anames=[]
  if not actions:raise ValueError('Explicit action list required')
  for action in actions:
   pose=np.asarray(action['pose'],float);anames.append(action['name'])
   if not re.fullmatch('[A-Za-z0-9_-]+',action['name'])or pose.shape!=(2,6)or not np.isfinite(pose).all():raise ValueError('Safe action name and finite2x6 command required')
  if len(set(anames))!=len(anames):raise ValueError('Duplicate actions')
 if not names or len(set(names))!=len(names):raise ValueError('Distinct nonempty scene IDs required across splits')
 if plan['source_sha256']!=source_inventory():raise ValueError('Exact transitive source inventory mismatch')
 return plan

def prepare(plan_path,output):
 plan_path=Path(plan_path);output=Path(output);plan=validate(json.loads(plan_path.read_text()));pin=sha(plan_path)
 marker=output/'prepared.json'
 if output.exists():
  if not marker.exists():raise ValueError('Unrecognized existing output directory')
  old=json.loads(marker.read_text())
  if old['plan_sha256']!=pin or old['source_sha256']!=plan['source_sha256']or sha(output/'plan.json')!=pin:raise ValueError('Existing output belongs to a different plan/source')
 else:
  output.mkdir(parents=True);(output/'plan.json').write_bytes(plan_path.read_bytes());write(marker,dict(plan_sha256=pin,source_sha256=plan['source_sha256']))
 return plan,pin

def verify_complete(folder,plan_sha,source_sha,identity):
 path=folder/'complete.json'
 if not path.exists():return None
 record=json.loads(path.read_text())
 if record['state']!='complete' or record['plan_sha256']!=plan_sha or record['source_sha256']!=source_sha or record['identity']!=identity:raise ValueError('Completed action commitment mismatch')
 if set(record['artifact_sha256'])!={'labels.npz','full-state.npz'}:raise ValueError('Incomplete artifact inventory')
 if any(sha(folder/name)!=digest for name,digest in record['artifact_sha256'].items()):raise ValueError('Completed action artifact changed')
 with np.load(folder/'labels.npz',allow_pickle=False)as data:
  if set(data.files)!={'material_query','displacement','wrenches'}or any(not np.isfinite(data[k]).all()for k in data.files):raise ValueError('Invalid cached labels')
 return record

def run_scene(plan_path,output,index,device):
 import torch
 from scripts.compositional_neural_mechanics.scenes import make_model
 torch.set_num_threads(1);plan,pin=prepare(plan_path,output)
 if device not in plan['devices']:raise ValueError('Undeclared execution device')
 scene_record=plan['scenes'][index];scene=scene_record['scene'];folder=Path(output)/'scenes'/scene['name'];folder.mkdir(parents=True,exist_ok=True)
 # Analytic teacher only: no learned checkpoint is used to generate labels.
 model=make_model(scene,n=plan['n'],order=plan['order'],device=device);q=np.asarray(plan['material_query'],float);rows=[];start=time.monotonic()
 for action in scene_record['actions']:
  validate(plan);identity=dict(scene_id=scene['name'],split=scene_record['role'],action=action['name'],prescribed_pose=action['pose'],n=plan['n'],order=plan['order'])
  actiondir=folder/action['name'];actiondir.mkdir(parents=True,exist_ok=True)
  cached=verify_complete(actiondir,pin,plan['source_sha256'],identity)
  if cached is not None:rows.append(dict(action=action['name'],state='complete',cached=True,complete_sha256=sha(actiondir/'complete.json')));continue
  attempts=actiondir/'attempts';attempts.mkdir(exist_ok=True);attempt=len(list(attempts.glob('*.json')))+1;started=time.monotonic()
  try:
   state=None;continuation=[]
   for fraction in plan['continuation']:
    model.set_controls(np.asarray(action['pose'])*fraction)
    state=model.solve(initial=None if state is None else state['displacement_local_m'],force_tolerance_N=plan['force_tolerance_N'])
    continuation.append(dict(fraction=fraction,iterations=state['iterations'],residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(np.min(state['J']))))
   field=model.sample_material(state['displacement_local_m'],q);w=model.reaction_wrenches(state);wrench=np.concatenate([w['forces_N'],w['torques_Nm']],axis=-1)
   if field.shape!=(len(q),3)or wrench.shape!=(2,6)or not all(np.isfinite(x).all()for x in (field,wrench,state['displacement_local_m'],state['J'])):raise ValueError('Nonfinite teacher output')
   npz(actiondir/'labels.npz',material_query=q,displacement=field,wrenches=wrench)
   npz(actiondir/'full-state.npz',rest=model.rest,elements=model.elements,material_nodes=model.material_coordinates,displacement=state['displacement_local_m'],jacobians=state['J'],prescribed_pose=np.asarray(action['pose']),forces_N=w['forces_N'],torques_Nm=w['torques_Nm'],origins_world_m=w['origins_world_m'])
   validate(plan)
   record=dict(state='complete',identity=identity,plan_sha256=pin,source_sha256=plan['source_sha256'],device=device,artifact_sha256={name:sha(actiondir/name)for name in ('labels.npz','full-state.npz')},residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(np.min(state['J'])),force_balance_N=float(w['force_balance_N']),wall_seconds=time.monotonic()-started,continuation=continuation,attempt=attempt)
   write(actiondir/'complete.json',record);write(attempts/f'{attempt:04d}.json',record);rows.append(dict(action=action['name'],state='complete',cached=False,complete_sha256=sha(actiondir/'complete.json')))
  except Exception as error:
   record=dict(state='failed',identity=identity,plan_sha256=pin,source_sha256=plan['source_sha256'],device=device,error=repr(error),wall_seconds=time.monotonic()-started,attempt=attempt)
   write(attempts/f'{attempt:04d}.json',record);write(actiondir/'status.json',record);rows.append(dict(action=action['name'],state='failed',error=repr(error)))
  write(folder/'progress.json',dict(plan_sha256=pin,rows=rows,expected_actions=len(scene_record['actions'])))
 write(folder/'complete.json',dict(state='complete'if all(r['state']=='complete'for r in rows)else'failed',plan_sha256=pin,rows=rows,wall_seconds=time.monotonic()-start))
 return rows

def run(plan_path,output):
 plan,pin=prepare(plan_path,output);output=Path(output)
 lock=(output/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 startpath=output/'stage-start.json'
 if not startpath.exists():write(startpath,dict(started_unix=time.time(),deadline_unix=time.time()+plan['stage_timeout_seconds'],plan_sha256=pin))
 stage=json.loads(startpath.read_text());deadline=stage['deadline_unix']
 if stage['plan_sha256']!=pin:raise ValueError('Stage deadline commitment mismatch')
 pending=queue.Queue()
 for i in range(len(plan['scenes'])):pending.put(i)
 progress_lock=threading.Lock();rows={}
 def summarize():
  allrows=[rows.get(i,dict(scene_id=r['scene']['name'],state='pending'))for i,r in enumerate(plan['scenes'])]
  counts={s:sum(r['state']==s for r in allrows)for s in ('pending','running','complete','failed','timeout','stage_timeout')}
  history=[json.loads(p.read_text())for p in (output/'scene-attempts').glob('*/*.json')]
  write(output/'progress.json',dict(plan_sha256=pin,source_sha256=plan['source_sha256'],deadline_unix=deadline,expected_scenes=len(allrows),counts=counts,historical_attempt_errors=sum(r['state']in ('failed','timeout')for r in history),rows=allrows))
 def worker(device):
  while True:
   try:index=pending.get_nowait()
   except queue.Empty:return
   record=plan['scenes'][index];name=record['scene']['name'];remaining=deadline-time.time();validate(plan)
   cached_actions=[]
   for action in record['actions']:
    folder=output/'scenes'/name/action['name'];identity=dict(scene_id=name,split=record['role'],action=action['name'],prescribed_pose=action['pose'],n=plan['n'],order=plan['order'])
    cached=verify_complete(folder,pin,plan['source_sha256'],identity)
    cached_actions.append(dict(action=action['name'],state='complete'if cached else'missing',complete_sha256=sha(folder/'complete.json')if cached else None))
   if all(a['state']=='complete'for a in cached_actions):
    with progress_lock:rows[index]=dict(scene_id=name,state='complete',cached=True,actions=cached_actions);summarize()
    continue
   if remaining<=0:
    with progress_lock:rows[index]=dict(scene_id=name,state='stage_timeout',actions=[dict(action=a['name'],state='not_run_deadline')for a in record['actions']]);summarize()
    continue
   attempts=output/'scene-attempts'/name;attempts.mkdir(parents=True,exist_ok=True);attempt_index=len(list(attempts.glob('*.json')))+1
   log=output/'logs'/name/f'{attempt_index:04d}.log';log.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic()
   with progress_lock:rows[index]=dict(scene_id=name,state='running',device=device);summarize()
   command=[sys.executable,'-B','-m','scripts.compositional_neural_mechanics.dataset_runner','--worker-index',str(index),'--device',device,'--plan',str(output/'plan.json'),'--output',str(output)]
   env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0,1,2');timedout=False
   with log.open('w')as stream:
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=child.wait(timeout=min(remaining,plan['per_scene_timeout_seconds']))
    except subprocess.TimeoutExpired:timedout=True;os.killpg(child.pid,signal.SIGKILL);code=child.wait()
   validate(plan);actionrows=[]
   for action in record['actions']:
    folder=output/'scenes'/name/action['name'];identity=dict(scene_id=name,split=record['role'],action=action['name'],prescribed_pose=action['pose'],n=plan['n'],order=plan['order'])
    cached=verify_complete(folder,pin,plan['source_sha256'],identity)
    actionrows.append(dict(action=action['name'],state='complete'if cached else'timeout'if timedout else'failed',complete_sha256=sha(folder/'complete.json')if cached else None))
   state='timeout'if timedout else'complete'if code==0 and all(a['state']=='complete'for a in actionrows)else'failed'
   row=dict(scene_id=name,state=state,device=device,exit_code=code,wall_seconds=time.monotonic()-started,actions=actionrows,log=str(log),log_sha256=sha(log))
   write(attempts/f'{attempt_index:04d}.json',row)
   with progress_lock:rows[index]=row;summarize()
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(plan['devices']))as pool:
  futures=[pool.submit(worker,d)for d in plan['devices']]
  for future in futures:future.result()
 summarize();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 return json.loads((output/'progress.json').read_text())

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--source-inventory',action='store_true');parser.add_argument('--execute',action='store_true');parser.add_argument('--worker-index',type=int);parser.add_argument('--device',default='cpu');args=parser.parse_args()
 if args.source_inventory:print(json.dumps(source_inventory(),indent=2));return
 if args.worker_index is not None:run_scene(args.plan,args.output,args.worker_index,args.device)
 elif args.execute:run(args.plan,args.output)
 else:prepare(args.plan,args.output)
if __name__=='__main__':main()
