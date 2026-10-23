"""C39 typed public-prefix choices. No DOWN outcomes/posteriors or SIDE acquisition.

One owned CPU2 worker seals every identity and both models; fixedDOWN fallback
on declared cloud failure. Forwardqualification/allinitial128 barrier precedes
any hypothesis/prefix arrays. No retries, private evaluator or noise producer.
"""
from pathlib import Path
import argparse,json,os,shutil,signal,subprocess,sys,time,traceback
import numpy as np
from experiments import run_campaign_c38_static_policy_v02 as base
from experiments import run_campaign_c38_initial_inference_v03 as io
from experiments import campaign_c39_public_runtime as common
from experiments import campaign_c39_typed_static_policy as policy
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_campaign_c39_static_policy.py'
RESOURCES={**base.RESOURCES,'global_wall_s':300}
MODELS=io.MODELS


def source_names(ip):return sorted(set(base.source_names(ip))|{SOURCE,'experiments/run_campaign_c38_static_policy_v02.py','experiments/run_campaign_c38_initial_inference_v03.py','experiments/campaign_c39_public_runtime.py','experiments/campaign_c39_typed_static_policy.py','experiments/campaign_c39_side_point_forward.py','tests/test_campaign_c39_public_runtime.py','tests/test_run_campaign_c39_static_policy.py'})


def prepare(protocol,output,*,initial_protocol,initial_protocol_sha256,forward_validation,root=ROOT):
 root=Path(root).resolve();dest=io.local(root,protocol);out=io.local(root,output)
 if dest.exists()or out.exists():raise FileExistsError('Fresh exclusive C39protocol/output')
 initial=dict(path=initial_protocol,sha256=initial_protocol_sha256);ip=base.validate_initial_protocol(root,initial,full_inputs=True);side=common.side_plan(root,io.record(root,io.local(root,forward_validation)))
 p=dict(schema='C39-typed-static-policy-execution-v01',status='FROZEN_BEFORE_TYPED_POLICY_PREFIX_ANALYSIS',extension_design=policy.DESIGN_RECORD,output=output,initial_protocol=initial,expected_initial_seal=ip['output']+'/prediction-seal.json',objects=ip['objects'],models=list(MODELS),side_inputs=side,resources=RESOURCES,runtime=io.runtime(),source_sha256={n:io.sha(root/n)for n in source_names(ip)},taskset=dict(path=shutil.which('taskset'),sha256=io.sha(shutil.which('taskset'))),private_SIDE_outcomes_access=False,acquired_DOWN_observations_or_posteriors_access=False,private_target_or_latent_access=False,retries=0,raw_local=True)
 # This metadata barrier never opens oldposterior arrays or private responses.
 base.initial_barrier(root,p,ip);dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p=io.read_json(root,dict(path=protocol,sha256=digest))
 if p.get('schema')!='C39-typed-static-policy-execution-v01'or p.get('status')!='FROZEN_BEFORE_TYPED_POLICY_PREFIX_ANALYSIS'or p.get('extension_design')!=policy.DESIGN_RECORD or p.get('resources')!=RESOURCES or p.get('runtime')!=io.runtime()or p.get('models')!=list(MODELS)or p.get('retries')!=0:raise ValueError('ExactC39typed/public/runtime/resource contract')
 for key in ('private_SIDE_outcomes_access','acquired_DOWN_observations_or_posteriors_access','private_target_or_latent_access'):
  if p.get(key)is not False:raise ValueError('Forbiddenchoice input '+key)
 ip=base.validate_initial_protocol(root,p['initial_protocol'],full_inputs=False)
 if p.get('source_sha256')!={n:io.sha(root/n)for n in source_names(ip)}or p.get('taskset')!=dict(path=shutil.which('taskset'),sha256=io.sha(shutil.which('taskset'))):raise ValueError('Full source/taskset pins changed')
 if p['objects']!=ip['objects']or p['expected_initial_seal']!=ip['output']+'/prediction-seal.json'or p['side_inputs']!=common.side_plan(root,p['side_inputs']['forward_point_validation']):raise ValueError('Exactinitial/qualifiedSIDE forwardclosure')
 io.local(root,p['output']);return p,ip


def worker(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p,ip=validate(protocol,digest,root=root);base.guard(root);seal,b=base.initial_barrier(root,p,ip);models=common.load_composites(root,b,p['side_inputs']);design=base.obs.load_frozen_design(root);out=io.local(root,p['output']);rows=[];modelrows={m:[]for m in MODELS}
 for row in p['objects']:
  base.guard(root);token=row['object_id'];public=base.codec.load_public_prefix(root,row['prefix'],design=design)
  if public['object_id']!=token:raise ValueError('Exactpublic identity')
  d=out/'objects'/token;d.mkdir();state=dict(completed=False,object_id=token,models={},private_SIDE_outcomes_access=False,acquired_DOWN_observations_or_posteriors_access=False,private_target_or_latent_access=False)
  for name,model in models.items():
   result=policy.design_typed_policy(model,public,design=design);path=d/(name+'.json');io.write_new(path,io.jsonable(result));choices,numeric=common.choices_from_result(result,token);entry=dict(choices=choices,result=io.record(root,path),numerical_controls_passed=numeric,fallback_due_to_numerical_failure=not numeric);state['models'][name]=entry;modelrows[name].append(dict(object_id=token,**entry));io.atomic(d/'result.json',state)
  state['completed']=True;io.atomic(d/'result.json',state);rows.append(dict(object_id=token,choices=state['models']['nonlinear_table']['choices'],result=io.record(root,d/'result.json')));io.atomic(out/'progress.json',dict(completed=False,count=len(rows),total=64))
 validate(protocol,digest,root=root);base.initial_barrier(root,p,ip)
 sealdata=dict(schema='C39-typed-static-policy-choices-seal-v01',completed=True,count=64,model='nonlinear_table',model_executions=128,protocol=dict(path=protocol,sha256=digest),extension_design=policy.DESIGN_RECORD,initial_prediction_seal=seal,forward_point_validation=p['side_inputs']['forward_point_validation'],candidate_side_archive_sha256=p['side_inputs']['candidate_side_archive']['sha256'],linear_side_archive_sha256=p['side_inputs']['linear_side_archive']['sha256'],action_catalog=[a.record()for a in policy.ACTIONS],choices=rows,model_choices=modelrows,private_SIDE_outcomes_access=False,acquired_DOWN_observations_or_posteriors_access=False,private_target_or_latent_access=False,all_choices_sealed_before_SIDE_noise=True,all_policy_numerical_controls_passed=all(r['numerical_controls_passed']for rs in modelrows.values()for r in rs),source_sha256=p['source_sha256'],raw_local=True)
 io.write_new(out/'choices-seal.json',sealdata);common.validate_choices(root,io.record(root,out/'choices-seal.json'),ids=[r['object_id']for r in p['objects']],initial_seal=seal,side_inputs=p['side_inputs'])


class Stop(BaseException):pass

def run(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();start=time.monotonic();p,ip=validate(protocol,digest,root=root);base.guard(root);seal,b=base.initial_barrier(root,p,ip);out=io.local(root,p['output'])
 if out.exists():raise FileExistsError('No overwrite/retry')
 out.mkdir(parents=True);(out/'objects').mkdir();state=dict(completed=False,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),private_SIDE_outcomes_access=False,acquired_DOWN_observations_or_posteriors_access=False,private_target_or_latent_access=False);proc=None;log=None
 def save():state['seconds']=time.monotonic()-start;io.atomic(out/'execution.json',state)
 def stop(*args):raise Stop('Owned C39policycontroller terminated')
 old={s:signal.signal(s,stop)for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
 try:
  log=(out/'worker.log').open('xb');cmd=[p['taskset']['path'],'-c','30,31',sys.executable,'-m','experiments.run_campaign_c39_static_policy','worker','--protocol',protocol,'--protocol-sha256',digest];proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state.update(worker_pid=proc.pid,PGID=proc.pid,command=cmd);save()
  while proc.poll()is None:
   base.guard(root)
   if time.monotonic()-start>RESOURCES['global_wall_s']:raise TimeoutError('C39policyglobal300scap')
   rss=io.current_rss(proc.pid)
   if rss is not None and rss>RESOURCES['RSS_bytes']:raise MemoryError('Ownedworker2GiB cap')
   time.sleep(.2)
  code=proc.wait();state['exit_code']=code
  if code!=0:raise RuntimeError('C39policyworkerexit '+str(code))
  rec=io.record(root,out/'choices-seal.json');common.validate_choices(root,rec,ids=[r['object_id']for r in p['objects']],initial_seal=seal,side_inputs=p['side_inputs']);validate(protocol,digest,root=root);state.update(completed=True,choices_seal=rec)
 except BaseException as e:state.update(failure=repr(e),traceback=traceback.format_exc())
 finally:
  if proc is not None:state['exit_code']=io.cleanup_owned(proc,3);state['owned_worker_reaped']=True
  if log:log.close()
  save()
  for s,h in old.items():signal.signal(s,h)
 return state


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run','worker'));a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--initial-protocol');a.add_argument('--initial-protocol-sha256');a.add_argument('--forward-validation');v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output,initial_protocol=v.initial_protocol,initial_protocol_sha256=v.initial_protocol_sha256,forward_validation=v.forward_validation))
 elif v.mode=='worker':worker(v.protocol,v.protocol_sha256)
 else:r=run(v.protocol,v.protocol_sha256);print(json.dumps(r));sys.exit(0 if r['completed']else 1)
