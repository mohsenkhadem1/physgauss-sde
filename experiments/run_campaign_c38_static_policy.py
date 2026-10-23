"""Sequential PUBLIC64 policy controller, gated by the complete initial seal.

One owned CPU2 subprocess computes both models for every object. No private
archive, outcome, target or evaluator is imported or accepted. Numerical failures
remain in the record and use the declared fixed-down fallback, never dropping IDs.
"""
from __future__ import annotations
import argparse,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_campaign_c38_initial_inference_v02 as initial
from experiments import run_campaign_c38_controlled_observations as codec
from experiments import campaign_c38_controlled_observations as obs
from experiments import campaign_c38_static_policy as policy
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_campaign_c38_static_policy.py'
MODELS=('nonlinear_table','linear_rest_tangent')
RESOURCES=dict(cores=[30,31],threads=2,global_wall_s=600,RSS_bytes=2*1024**3,disk_free_reserve_bytes=30*1024**3,projected_output_bytes=256*1024**2,no_CUDA=True,cleanup_s=3)
sha=initial.sha;local=initial.local;checked=initial.checked;record=initial.record;read_json=initial.read_json;write_new=initial.write_new;atomic=initial.atomic


def source_names():return sorted(set(initial.source_names())|{SOURCE,'experiments/campaign_c38_static_policy.py','tests/test_campaign_c38_static_policy.py','tests/test_run_campaign_c38_static_policy.py'})


def prepare(protocol,output,*,initial_protocol,initial_protocol_sha256,root=ROOT):
 root=Path(root).resolve();dest=local(root,protocol);out=local(root,output)
 if dest.exists() or out.exists():raise FileExistsError('Exclusive fresh protocol/output; no retry')
 ip=initial.validate(initial_protocol,initial_protocol_sha256,root=root,full_inputs=True)
 p=dict(schema='C38-static-policy-execution-v01',status='FROZEN_BEFORE_POLICY_PREFIX_ANALYSIS',output=output,initial_protocol=dict(path=initial_protocol,sha256=initial_protocol_sha256),expected_initial_seal=ip['output']+'/prediction-seal.json',objects=ip['objects'],models=list(MODELS),resources=RESOURCES,runtime=initial.runtime(),source_sha256={f:sha(root/f) for f in source_names()},taskset=dict(path=shutil.which('taskset'),sha256=sha(shutil.which('taskset'))),private_probe_outcomes_access=False,private_target_or_latent_access=False,raw_local=True,retries=0,policy='All64 initial128 predictions must be sealed and terminal before public analysis; all64 policy choices freeze before ANY private evaluation/acquisition. Both models computed; primary choices are nonlinear_table. Numerical failures retained with fixed-down info/risk fallback, fixed/random unchanged.')
 dest.parent.mkdir(parents=True,exist_ok=True);write_new(dest,p);return sha(dest)


def validate(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p=read_json(root,dict(path=protocol,sha256=digest))
 if p.get('schema')!='C38-static-policy-execution-v01' or p.get('status')!='FROZEN_BEFORE_POLICY_PREFIX_ANALYSIS' or p.get('resources')!=RESOURCES or p.get('models')!=list(MODELS) or p.get('private_probe_outcomes_access') is not False or p.get('private_target_or_latent_access') is not False or p.get('retries')!=0:raise ValueError('Exact public-only policy/resources required')
 if p.get('runtime')!=initial.runtime() or p.get('source_sha256')!={f:sha(root/f) for f in source_names()}:raise ValueError('Full source/runtime pin mismatch')
 if p['taskset']['path']!=shutil.which('taskset') or p['taskset']['sha256']!=sha(shutil.which('taskset')):raise ValueError('Taskset changed')
 ip=initial.validate(p['initial_protocol']['path'],p['initial_protocol']['sha256'],root=root,full_inputs=False)
 if p['objects']!=ip['objects'] or p['expected_initial_seal']!=ip['output']+'/prediction-seal.json':raise ValueError('Exact initial public identities/seal binding required')
 local(root,p['output']);return p,ip


def validate_initial_seal(seal,execution,*,protocol_digest,ids):
 """Pure metadata gate; no prediction arrays or observed values needed."""
 if seal.get('schema')!='C38-initial-public-inference-seal-v02' or seal.get('completed') is not True or seal.get('count')!=64 or seal.get('model_executions')!=128 or seal.get('protocol_sha256')!=protocol_digest or seal.get('private_truth_access') is not False or seal.get('all_predictions_sealed_before_private_evaluation') is not True:raise ValueError('Complete untouched PUBLIC initial128 seal required')
 rows=seal.get('objects',[])
 if len(rows)!=64 or [r.get('object_id') for r in rows]!=ids or len(set(ids))!=64:raise ValueError('Exact all64 sorted public identities required')
 if execution.get('completed') is not True or execution.get('all_owned_workers_reaped') is not True or execution.get('private_truth_access') is not False or execution.get('technical_failure') is not False or len(execution.get('records',[]))!=64:raise ValueError('Initial execution terminal/completed/no technical failures required')
 if sorted(r.get('object_id') for r in execution['records'])!=ids:raise ValueError('Initial terminal exact64 identity set required')
 byid={r['object_id']:r for r in execution['records']}
 if any(byid[r['object_id']].get('result')!=r.get('result') for r in rows):raise ValueError('Initial execution/seal result records differ')
 return rows


def initial_barrier(root,p,ip):
 # Entire metadata and hash barrier precedes ANY prefix/model NPZ extraction.
 seal_record=record(root,local(root,p['expected_initial_seal']));s=read_json(root,seal_record);e=read_json(root,record(root,local(root,ip['output']+'/execution.json')));ids=[r['object_id'] for r in p['objects']]
 rows=validate_initial_seal(s,e,protocol_digest=p['initial_protocol']['sha256'],ids=ids)
 if e.get('prediction_seal')!=seal_record:raise ValueError('Initial terminal/seal hash mismatch')
 try:os.kill(int(e['pid']),0)
 except ProcessLookupError:pass
 else:raise ValueError('Initial controller must be terminal before policy analysis')
 for row in rows:
  r=read_json(root,row['result'])
  if r.get('completed') is not True or r.get('object_id')!=row['object_id'] or r.get('private_truth_access') is not False or set(r.get('models',{}))!=set(MODELS):raise ValueError('Complete bound initial two-model object required')
  for model in r['models'].values():
   if model.get('execution_completed') is not True or set(model.get('files',{}))!={'posterior-arrays.npz','posterior-structure.json','numerics.json'}:raise ValueError('Both initial model outputs must be retained')
   for f in model['files'].values():checked(root,f)
 barrier_record=e['model_input_barrier'];b=read_json(root,barrier_record)
 if barrier_record['path']!=ip['output']+'/model-input-barrier.json' or b.get('completed') is not True or b.get('protocol_sha256')!=p['initial_protocol']['sha256'] or b.get('candidate')!=ip['model_inputs']['candidate_archive'] or b.get('public_manifest')!=ip['model_inputs']['public_manifest'] or b.get('private_truth_access') is not False:raise ValueError('Exact completed public input barrier required')
 if initial.public_plan(root,ip['model_inputs']['public_manifest'])!=p['objects']:raise ValueError('Public manifest/prefix hashes changed')
 checked(root,b['linear_library']);checked(root,b['candidate']);return seal_record,b


def guard(root):
 if set(os.sched_getaffinity(0))!=set(RESOURCES['cores']) or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('Exact CPU30/31/thread2/noCUDA contract required')
 if current_rss(os.getpid())>RESOURCES['RSS_bytes']:raise MemoryError('2GiB RSS cap')
 if shutil.disk_usage(root).free-RESOURCES['projected_output_bytes']<RESOURCES['disk_free_reserve_bytes']:raise OSError('Free filesystem reserve30GiB+projected output required')


def current_rss(pid):
 try:
  text=Path(f'/proc/{pid}/status').read_text();lines=[x for x in text.splitlines() if x.startswith('VmRSS:')]
  return int(lines[0].split()[1])*1024 if lines else 0
 except FileNotFoundError:return 0 # Poll exit separately; never turn exit/reap race into failure.


def choices_from_result(result,object_id):
 numeric=result.get('completed') is True and result.get('numerical_cloud_controls_passed') is True
 random=int(np.random.default_rng(policy._seed(policy.RANDOM_ROOT_SEED,object_id)).integers(2))
 c=dict(information=-2.,target_risk=-2.,fixed_down=-2.,fixed_up=2.,random=policy.ACTIONS[random])
 if numeric:
  for name in ['information','target_risk']:c[name]=float(result[name]['selected_force_Fz_N'])
 if any(v not in (-2.,2.) for v in c.values()):raise ValueError('Exact equal-cost known actions only')
 return c,numeric


def worker(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p,ip=validate(protocol,digest,root=root);guard(root);seal_record,b=initial_barrier(root,p,ip);out=local(root,p['output'])
 from experiments.campaign_c38_candidate_point_forward import CandidatePointForward,ARCHIVE_KEYS
 with np.load(checked(root,b['candidate']),allow_pickle=False) as z:
  if set(z.files)!=ARCHIVE_KEYS:raise ValueError('Candidate-only forward archive required')
  nonlinear=CandidatePointForward({k:z[k].copy() for k in ARCHIVE_KEYS},cells=(40,20,20))
 linear=initial.load_linear_library(root,b['linear_library']);design=obs.load_frozen_design(root);models=dict(zip(MODELS,(nonlinear,linear)));rows=[];model_choices={n:[] for n in MODELS}
 for public_row in p['objects']:
  guard(root);token=public_row['object_id'];public=codec.load_public_prefix(root,public_row['prefix'],design=design)
  if public['object_id']!=token:raise ValueError('Public identity mismatch')
  d=out/'objects'/token;d.mkdir();state=dict(completed=False,object_id=token,models={},private_probe_outcomes_access=False,private_target_or_latent_access=False)
  for name,model in models.items():
   guard(root);result=policy.design_static_policy(model,public,design=design);path=d/(name+'.json');write_new(path,initial.jsonable(result));choices,numeric=choices_from_result(result,token);r=dict(choices=choices,result=record(root,path),numerical_controls_passed=numeric,fallback_due_to_numerical_failure=not numeric);state['models'][name]=r;atomic(d/'result.json',state)
   model_choices[name].append(dict(object_id=token,**r))
  state['completed']=True;atomic(d/'result.json',state);rows.append(dict(object_id=token,choices=state['models']['nonlinear_table']['choices'],result=record(root,d/'result.json')));atomic(out/'progress.json',dict(completed=False,count=len(rows),total=64,private_probe_outcomes_access=False))
 if len(rows)!=64 or [r['object_id'] for r in rows]!=[r['object_id'] for r in p['objects']]:raise ValueError('No dropped identities')
 validate(protocol,digest,root=root);initial_barrier(root,p,ip)
 seal=dict(schema='C38-static-policy-choices-seal-v01',completed=True,count=64,model='nonlinear_table',model_executions=128,protocol=dict(path=protocol,sha256=digest),initial_prediction_seal=seal_record,choices=rows,model_choices=model_choices,private_probe_outcomes_access=False,private_target_or_latent_access=False,all_choices_sealed_before_private_evaluation=True,all_policy_numerical_controls_passed=all(r['numerical_controls_passed'] for rs in model_choices.values() for r in rs),source_sha256=p['source_sha256'],raw_local=True)
 write_new(out/'choices-seal.json',seal)


class Stop(BaseException):pass
def run(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();start=time.monotonic();p,ip=validate(protocol,digest,root=root);guard(root);initial_barrier(root,p,ip);out=local(root,p['output'])
 if out.exists():raise FileExistsError('Fresh output only; failures preserved, no retry')
 out.mkdir(parents=True);(out/'objects').mkdir();state=dict(completed=False,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),private_probe_outcomes_access=False,private_target_or_latent_access=False);proc=None;log=None
 def save():state['elapsed_s']=time.monotonic()-start;atomic(out/'execution.json',state)
 def stop(*args):raise Stop('Owned controller terminated')
 old={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]}
 try:
  log=(out/'worker.log').open('xb');cmd=[p['taskset']['path'],'-c','30,31',sys.executable,'-m','experiments.run_campaign_c38_static_policy','worker','--protocol',protocol,'--protocol-sha256',digest];proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state.update(worker_pid=proc.pid,PGID=proc.pid,command=cmd);save()
  while proc.poll() is None:
   guard(root)
   if time.monotonic()-start>RESOURCES['global_wall_s']:raise TimeoutError('600s global cap')
   if current_rss(proc.pid)>RESOURCES['RSS_bytes']:raise MemoryError('Owned worker2GiB RSS cap')
   time.sleep(.2)
  code=proc.wait();state['exit_code']=code
  if code!=0:raise RuntimeError('Policy worker technical failure '+str(code))
  seal_record=record(root,out/'choices-seal.json');s=read_json(root,seal_record)
  if s.get('completed') is not True or s.get('count')!=64 or s.get('protocol')!={'path':protocol,'sha256':digest}:raise ValueError('Complete exact64 choices seal required')
  validate(protocol,digest,root=root);state.update(completed=True,choices_seal=seal_record)
 except BaseException as error:state.update(failure=repr(error),traceback=traceback.format_exc())
 finally:
  if proc is not None:state['exit_code']=initial.cleanup_owned(proc,RESOURCES['cleanup_s']);state['owned_worker_reaped']=True
  if log is not None:log.close()
  save()
  for s,h in old.items():signal.signal(s,h)
 return state


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','worker']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--initial-protocol');a.add_argument('--initial-protocol-sha256');v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output,initial_protocol=v.initial_protocol,initial_protocol_sha256=v.initial_protocol_sha256))
 elif v.mode=='worker':worker(v.protocol,v.protocol_sha256)
 else:r=run(v.protocol,v.protocol_sha256);print(json.dumps(r));sys.exit(0 if r['completed'] else 1)
