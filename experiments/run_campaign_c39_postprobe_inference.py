"""C39 PUBLIC prefix+ONE typedSIDE acquisition, both hypothetical models.

No private loader/evaluator or other-action payload enters a worker. All64
object/SIDE pairs (128 model executions, including numerical failures) seal
before private canonical targets/material/support labels may be evaluated.
"""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,time,traceback
import numpy as np
from experiments import run_campaign_c38_initial_inference_v03 as io
from experiments import run_campaign_c38_controlled_observations as prefix_codec
from experiments import campaign_c38_controlled_observations as obs
from experiments.campaign_c39_postprobe_inference import PROBE_KEYS,infer_public_probe
from experiments.campaign_c39_side_observations import decode as decode_probe
from experiments import campaign_c39_public_runtime as common
from experiments import run_campaign_c39_static_policy as choice_runtime
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_campaign_c39_postprobe_inference.py'
RESOURCES={**io.RESOURCES,'worker_slots':[[i,i+1] for i in range(6,30,2)],'max_workers':12}
ACQUISITION_MANIFEST_KEYS={'schema','completed','count','records','extension_design','choices_seal','action_id','force_vector_N','same_original_identities_and_prefixes','new_SIDE_noise_only','no_new_DOWN_noise','private_labels_or_targets_included','is_RGBD_perception','is_temporal_forecast'}


def sources():return sorted(set(io.source_names())|set(choice_runtime.source_names(dict(source_sha256={})))|{SOURCE,'experiments/run_campaign_c38_postprobe_inference.py','experiments/campaign_c39_postprobe_inference.py','experiments/campaign_c39_side_observations.py','experiments/campaign_c39_side_point_forward.py','experiments/campaign_c39_public_runtime.py','experiments/campaign_c39_typed_static_policy.py','experiments/run_campaign_c39_static_policy.py','experiments/run_campaign_c38_static_policy_v02.py','experiments/campaign_c38_static_policy.py','experiments/campaign_c38_exact_posterior_memo.py','experiments/campaign_c38_postprobe_memo.py','tests/test_campaign_c39_postprobe_inference.py','tests/test_run_campaign_c39_postprobe_inference.py'})


def initial_ready(root,protocol_record):
 ip=io.validate(protocol_record['path'],protocol_record['sha256'],root=root)
 out=io.local(root,ip['output']);execution_record=io.record(root,out/'execution.json');execution=io.read_json(root,execution_record)
 if execution.get('completed')is not True or execution.get('technical_failure')is not False or execution.get('all_owned_workers_reaped')is not True or len(execution.get('records',[]))!=64 or execution.get('protocol')!=protocol_record:raise ValueError('Complete/reaped64 initialexecution required')
 pids=[execution['pid']]+[r['pid']for r in execution['records']if isinstance(r.get('pid'),int)]
 if any(Path('/proc',str(pid)).exists()for pid in pids):raise ValueError('Initialcontroller/allworkers must be absent')
 seal_record=execution['prediction_seal'];seal=io.read_json(root,seal_record)
 if seal.get('completed')is not True or seal.get('count')!=64 or seal.get('model_executions')!=128 or seal.get('private_truth_access')is not False or seal.get('protocol_sha256')!=protocol_record['sha256']or seal.get('all_predictions_sealed_before_private_evaluation')is not True:raise ValueError('Complete64x2 initialprediction seal required')
 byid={r['object_id']:r for r in execution['records']}
 if len(byid)!=64 or {r['object_id']for r in seal['objects']}!=set(byid):raise ValueError('Exact64 initialidentity join')
 for r in seal['objects']:
  if r['result']!=byid[r['object_id']]['result']:raise ValueError('Initialresult seal mismatch')
  io.verify_worker_output(root,ip,byid[r['object_id']])
 barrier_record=execution['model_input_barrier'];b=io.read_json(root,barrier_record)
 if b.get('completed')is not True or b.get('protocol_sha256')!=protocol_record['sha256']or b.get('candidate')!=ip['model_inputs']['candidate_archive']or b.get('public_manifest')!=ip['model_inputs']['public_manifest']or b.get('private_truth_access')is not False:raise ValueError('Exact initialpublic modelinput barrier required')
 io.checked(root,b['linear_library'])
 return ip,dict(initial_prediction_seal=seal_record,initial_execution=execution_record,input_barrier=barrier_record,candidate_archive=b['candidate'],linear_library=b['linear_library'],public_manifest=b['public_manifest'])


def public_jobs(root,manifest_record,inputs):
 path=io.checked(root,manifest_record);m=json.loads(path.read_text())
 if set(m)!=ACQUISITION_MANIFEST_KEYS:raise ValueError('Exact sanitizedSIDE manifest allowlist')
 if m.get('schema')!='C39-public-acquired-SIDE-probes-v01'or m.get('completed')is not True or m.get('count')!=64 or m.get('extension_design')!=common.DESIGN or m.get('action_id')!='middle_side_2N'or m.get('force_vector_N')!=[0.,2.,0.]:raise ValueError('Complete64 typedSIDEmanifest required')
 for key in ('same_original_identities_and_prefixes','new_SIDE_noise_only','no_new_DOWN_noise'):
  if m.get(key)is not True:raise ValueError('OnlynewSIDEacquisitions')
 for key in ('private_labels_or_targets_included','is_RGBD_perception','is_temporal_forecast'):
  if m.get(key)is not False:raise ValueError('No labels/future/perception inputs')
 prefixes={r['object_id']:r['prefix']for r in io.public_plan(root,inputs['public_manifest'])};ids=sorted(prefixes)
 if len(m.get('records',[]))!=64 or [r.get('object_id')for r in m['records']]!=ids:raise ValueError('All64 exact sorted original acquiredidentities')
 side=common.side_plan(root,inputs['side_inputs']['forward_point_validation'])
 if side!=inputs['side_inputs']:raise ValueError('SamequalifiedSIDE libraries')
 choices=common.validate_choices(root,m['choices_seal'],ids=ids,initial_seal=inputs['initial_prediction_seal'],side_inputs=side)
 # Policy runtime complete/source closure is validated before SIDE arrays.
 cp=choices['protocol']
 choice_p,_=choice_runtime.validate(cp['path'],cp['sha256'],root=root);ce=io.read_json(root,io.record(root,io.local(root,choice_p['output']+'/execution.json')))
 if ce.get('completed')is not True or ce.get('owned_worker_reaped')is not True or ce.get('choices_seal')!=m['choices_seal']:raise ValueError('Choiceworker completed and reaped before SIDE inference')
 if any(Path('/proc',str(ce[k])).exists()for k in ('pid','worker_pid')):raise ValueError('Choiceprocesses must be absent')
 jobs=[]
 for row in m['records']:
  token=row['object_id'];name=token+'__side2.npz'
  if set(row)!={'object_id','acquisition'}or set(row['acquisition'])!={'path','sha256'}or row['acquisition']['path']!=name:raise ValueError('ExactSIDE publicacquisition basename')
  q=(path.parent/name).resolve()
  if q.parent!=path.parent.resolve():raise ValueError('SIDEacquisition pathescape')
  rec=dict(path=q.relative_to(Path(root).resolve()).as_posix(),sha256=row['acquisition']['sha256']);io.checked(root,rec);jobs.append(dict(id=token+'__side2',object_id=token,action_id='middle_side_2N',force_world_N=[0.,2.,0.],prefix=prefixes[token],acquisition=rec))
 return jobs,m['choices_seal']


def prepare(protocol,output,*,initial_protocol,initial_protocol_sha256,acquisitions_manifest,forward_validation,root=ROOT):
 root=Path(root).resolve();dest=io.local(root,protocol);out=io.local(root,output)
 if dest.exists()or out.exists():raise FileExistsError('Freshexclusive postprobeprotocol/output required')
 initial_record=dict(path=initial_protocol,sha256=initial_protocol_sha256);ip,inputs=initial_ready(root,initial_record);inputs['side_inputs']=common.side_plan(root,io.record(root,io.local(root,forward_validation)))
 acquisitions=io.record(root,io.local(root,acquisitions_manifest));jobs,choices=public_jobs(root,acquisitions,inputs)
 p=dict(schema='C39-public-SIDE-postprobe-inference-v01',status='FROZEN_BEFORE_POSTPROBE_INFERENCE',output=output,initial_protocol=initial_record,inputs={**inputs,'acquisitions_manifest':acquisitions,'choices_seal':choices},source_sha256={n:io.sha(root/n)for n in sources()},runtime=io.runtime(),resources=RESOURCES,taskset=ip['taskset'],jobs=jobs,models=list(io.MODELS),retries=0,private_target_or_latent_access=False,posteriors='Fresh54-coordinate likelihood from uniformlogE/equalBC prior; no reuse of initial posterior weights. One probe perpolicy; same held canonical+1 target.',all128_SIDE_predictions_before_private_evaluation=True,raw_local=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest,*,root=ROOT,full_inputs=True):
 root=Path(root).resolve();p=io.read_json(root,dict(path=protocol,sha256=digest))
 if p.get('schema')!='C39-public-SIDE-postprobe-inference-v01'or p.get('status')!='FROZEN_BEFORE_POSTPROBE_INFERENCE'or p.get('resources')!=RESOURCES or p.get('runtime')!=io.runtime()or p.get('source_sha256')!={n:io.sha(root/n)for n in sources()}or p.get('models')!=list(io.MODELS)or p.get('retries')!=0 or p.get('private_target_or_latent_access')is not False or p.get('all128_SIDE_predictions_before_private_evaluation')is not True:raise ValueError('Frozenpostprobe resource/source/privacy protocol required')
 io.local(root,p['output'])
 if full_inputs:
  ip,inputs=initial_ready(root,p['initial_protocol']);inputs['side_inputs']=common.side_plan(root,p['inputs']['side_inputs']['forward_point_validation']);jobs,choices=public_jobs(root,p['inputs']['acquisitions_manifest'],inputs)
  if p['inputs']!={**inputs,'acquisitions_manifest':p['inputs']['acquisitions_manifest'],'choices_seal':choices}or p['jobs']!=jobs or p['taskset']!=ip['taskset']:raise ValueError('Exact full64SIDEpublicjob/inputclosure required')
 return p


def worker(protocol,digest,job_id,slot,*,root=ROOT):
 root=Path(root).resolve();p=validate(protocol,digest,root=root,full_inputs=False)
 if type(slot)is not int or not 0<=slot<len(RESOURCES['worker_slots']):raise ValueError('DeclaredCPUslot required')
 io.guard(root,RESOURCES['worker_slots'][slot]);matches=[j for j in p['jobs']if j['id']==job_id]
 if len(matches)!=1:raise ValueError('Exact declared ONE object/action job required')
 job=matches[0];design=obs.load_frozen_design(root);prefix=prefix_codec.load_public_prefix(root,job['prefix'],design=design)
 if prefix['object_id']!=job['object_id']:raise ValueError('Prefix identity binding')
 # Only ONE acquiredprobe file, never the counterfactual otheraction or private.
 with np.load(io.checked(root,job['acquisition']),allow_pickle=False)as z:
  if set(z.files)!=PROBE_KEYS:raise ValueError('ExactpublicONEprobe members')
  acquired=decode_probe(prefix,{k:z[k].copy()for k in PROBE_KEYS},design=design)
 if acquired['action_id']!=job['action_id']or not np.array_equal(acquired['force_vector_N'],job['force_world_N']):raise ValueError('Acquiredprobe action binding')
 models=common.load_composites(root,dict(candidate=p['inputs']['candidate_archive'],linear_library=p['inputs']['linear_library']),p['inputs']['side_inputs']);out=io.local(root,p['output'])/'objects'/job_id;out.mkdir(parents=True,exist_ok=False)
 state=dict(completed=False,pid=os.getpid(),slot=slot,object_id=job['object_id'],action_id=job['action_id'],force_world_N=job['force_world_N'],protocol_sha256=digest,models={},private_target_or_latent_access=False,raw_local=True);start=time.monotonic();io.atomic(out/'result.json',state)
 for name,model in models.items():
  io.guard(root,RESOURCES['worker_slots'][slot]);d=out/name;d.mkdir();result=infer_public_probe(model,prefix,acquired,design=design)
  if result.get('object_id',job['object_id'])!=job['object_id']or result.get('acquired_probe_action_id',job['action_id'])!=job['action_id']or not np.array_equal(result.get('acquired_probe_force_world_N',job['force_world_N']),job['force_world_N']):raise ValueError('Postresult object/action binding')
  io.save_full_result(d,result);files={n:io.record(root,d/n)for n in ('posterior-arrays.npz','posterior-structure.json','numerics.json')}
  state['models'][name]=dict(execution_completed=True,completed=result.get('completed')is True,numerical_controls_passed=result.get('numerical_controls_passed')is True,files=files);state['seconds']=time.monotonic()-start;io.atomic(out/'result.json',state)
 state.update(completed=True,all_models_numerical_passed=all(r['completed']and r['numerical_controls_passed']for r in state['models'].values()),seconds=time.monotonic()-start);io.atomic(out/'result.json',state);return state


def output_result(root,row,digest):
 a=io.read_json(root,row['result'])
 if a.get('completed')is not True or a.get('pid')!=row['pid']or a.get('object_id')!=row['object_id']or a.get('action_id')!=row['action_id']or a.get('force_world_N')!=row['force_world_N']or a.get('protocol_sha256')!=digest or set(a.get('models',{}))!=set(io.MODELS)or a.get('private_target_or_latent_access')is not False:raise ValueError('Complete boundpostprobe result required')
 for v in a['models'].values():
  if v.get('execution_completed')is not True or set(v['files'])!={'posterior-arrays.npz','posterior-structure.json','numerics.json'}:raise ValueError('Everypostmodel output retained')
  for f in v['files'].values():io.checked(root,f)
 return a


def run(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p=validate(protocol,digest,root=root);io.guard(root,RESOURCES['controller_cores']);out=io.local(root,p['output'])
 if out.exists():raise FileExistsError('No postinference overwrite/retry')
 out.mkdir(parents=True);(out/'objects').mkdir();(out/'logs').mkdir();start=time.monotonic();live={};stopped=False
 state=dict(completed=False,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),records=[],queued_job_ids=[j['id']for j in p['jobs']],live=[],technical_failure=False,private_target_or_latent_access=False,raw_local=True,retries=0)
 jobs={j['id']:j for j in p['jobs']}
 def save():state.update(seconds=time.monotonic()-start,live=[dict(pid=x['process'].pid,job_id=x['row']['job_id'],slot=k)for k,x in live.items()]);io.atomic(out/'execution.json',state)
 def stop(*args):raise KeyboardInterrupt('Owned C39SIDEcontroller terminated')
 old_handlers={s:signal.signal(s,stop)for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
 save()
 try:
  while state['queued_job_ids']or live:
   io.guard(root,RESOURCES['controller_cores'])
   if time.monotonic()-start>3600:raise TimeoutError('Globalpostprobe3600scap')
   if not stopped:
    for slot in range(RESOURCES['max_workers']):
     if slot in live or not state['queued_job_ids']:continue
     job_id=state['queued_job_ids'].pop(0);j=jobs[job_id];log=(out/'logs'/(job_id+'.log')).open('xb');cmd=[p['taskset']['path'],'-c',','.join(map(str,RESOURCES['worker_slots'][slot])),sys.executable,'-m','experiments.run_campaign_c39_postprobe_inference','worker','--protocol',protocol,'--protocol-sha256',digest,'--job-id',job_id,'--slot',str(slot)]
     try:proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     except BaseException:log.close();state['queued_job_ids'].insert(0,job_id);raise
     row=dict(job_id=job_id,object_id=j['object_id'],action_id=j['action_id'],force_world_N=j['force_world_N'],pid=proc.pid,PGID=proc.pid,slot=slot,command=cmd,start_monotonic=time.monotonic());live[slot]=dict(process=proc,row=row,log=log);save()
   for slot,x in list(live.items()):
    proc=x['process'];row=x['row'];code=proc.poll();reason=None
    if code is None:
     if time.monotonic()-row['start_monotonic']>300:reason='300s perpostprobe jobcap'
     else:
      rss=io.current_rss(proc.pid)
      if rss is None:
       try:code=proc.wait(timeout=.05)
       except subprocess.TimeoutExpired:code=None
      elif rss>RESOURCES['worker_RSS_bytes']:reason='2GiBpostworkerRSScap'
     if reason is not None:code=io.cleanup_owned(proc,3)
    if code is None:continue
    proc.wait();x['log'].close();row.update(exit_code=code,reaped=True,seconds=time.monotonic()-row['start_monotonic']);del live[slot]
    try:
     if reason is not None or code!=0:raise RuntimeError(reason or 'Technicalpostworkerexit'+str(code))
     row['result']=io.record(root,out/'objects'/row['job_id']/'result.json');r=output_result(root,row,digest);row.update(completed=True,all_models_numerical_passed=r['all_models_numerical_passed'])
    except Exception as error:row.update(completed=False,failure=repr(error));stopped=True;state['technical_failure']=True
    state['records'].append(row);save()
   if stopped and not live:break
   time.sleep(.2)
  if not state['technical_failure']and len(state['records'])==64 and not state['queued_job_ids']:
   for row in state['records']:output_result(root,row,digest)
   validate(protocol,digest,root=root)
   seal=dict(schema='C39-public-SIDE-postprobe-prediction-seal-v01',completed=True,count=64,model_executions=128,objects=sorted([dict(object_id=r['object_id'],action_id=r['action_id'],result=r['result'])for r in state['records']],key=lambda r:r['object_id']),choices_seal=p['inputs']['choices_seal'],initial_prediction_seal=p['inputs']['initial_prediction_seal'],protocol=dict(path=protocol,sha256=digest),private_target_or_latent_access=False,all_postpredictions_sealed_before_private_evaluation=True,all_numerical_controls_passed=all(r['all_models_numerical_passed']for r in state['records']),raw_local=True)
   io.write_new(out/'prediction-seal.json',seal);state.update(completed=True,prediction_seal=io.record(root,out/'prediction-seal.json'))
 except BaseException as error:state.update(technical_failure=True,failure=dict(type=type(error).__name__,message=str(error)),traceback=traceback.format_exc());save();raise
 finally:
  state['live_before_owned_cleanup']=[x['process'].pid for x in live.values()];save()
  for slot,x in list(live.items()):
   code=io.cleanup_owned(x['process'],3);x['log'].close();x['row'].update(completed=False,exit_code=code,reaped=True,failure='Ownedcleanup no retry');state['records'].append(x['row']);del live[slot]
  state['all_owned_workers_reaped']=not live;save()
  for sig,handler in old_handlers.items():signal.signal(sig,handler)
 return state

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run','worker'));a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--initial-protocol');a.add_argument('--initial-protocol-sha256');a.add_argument('--acquisitions-manifest');a.add_argument('--forward-validation');a.add_argument('--job-id');a.add_argument('--slot',type=int);v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output,initial_protocol=v.initial_protocol,initial_protocol_sha256=v.initial_protocol_sha256,acquisitions_manifest=v.acquisitions_manifest,forward_validation=v.forward_validation))
 elif v.mode=='worker':worker(v.protocol,v.protocol_sha256,v.job_id,v.slot)
 else:
  s=run(v.protocol,v.protocol_sha256);print(json.dumps({'completed':s['completed']}));sys.exit(0 if s['completed'] else 1)
