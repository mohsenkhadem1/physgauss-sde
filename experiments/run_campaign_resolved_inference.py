"""Separate numerical repair of SAME public prefix/SIDE posteriors, no truth API.

All C39 workers must be sealed/reaped before prepare or model/measurement reads.
No observations, policies, forces, priors or posterior moments are regenerated.
Old outputs stay immutable; new quantiles are not posthoc uncertainty margins.
"""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,time,traceback
import numpy as np
from experiments import run_campaign_c38_initial_inference_v03 as io
from experiments import run_campaign_c39_postprobe_inference as oldside
from experiments import campaign_resolved_prefix_inference as prefix
from experiments import campaign_resolved_SIDE_inference as side
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_campaign_resolved_inference.py'
RESOURCES={**oldside.RESOURCES,'worker_wall_s':600,'global_wall_s':1800}
KINDS=('prefix','SIDE')
FILES={'posterior-arrays.npz','posterior-structure.json','numerics.json'}
CORE_KEYS={'log_normalizer','normalizer_error_estimate','scaled_normalizer','normalizer_relative_error_estimate','category_probabilities','material_CDF','material_quantile_details','material_equal_tail_Pa','canonical_mean_displacement_m','canonical_mean_positions_m','canonical_coordinate_variance_m2','canonical_mean_coordinate_variance_m2','canonical_moment_numerics','canonical_variance_integration_resolution_estimate_m2','canonical_variance_resolution_scope'}


def sources():
 return sorted(set(oldside.sources())|{SOURCE,'experiments/campaign_resolved_prefix_inference.py','experiments/campaign_resolved_SIDE_inference.py','experiments/campaign_resolved_quantiles.py','tests/test_campaign_resolved_inference.py','tests/test_run_campaign_resolved_inference.py'})


def completed_inputs(root,initial_record,SIDE_record):
 """Metadata/hash barrier only: no NPZ member/measurement access."""
 ip,b=oldside.initial_ready(root,initial_record)
 sp=oldside.validate(SIDE_record['path'],SIDE_record['sha256'],root=root)
 if sp['initial_protocol']!=initial_record:raise ValueError('Exact original initial/SIDE lineage required')
 out=io.local(root,sp['output']);er=io.record(root,out/'execution.json');execution=io.read_json(root,er)
 if execution.get('completed')is not True or execution.get('technical_failure')is not False or execution.get('all_owned_workers_reaped')is not True or execution.get('protocol')!=SIDE_record or len(execution.get('records',[]))!=64 or execution.get('queued_job_ids')!=[]or execution.get('live')!=[]:raise ValueError('All64 originalSIDE jobs terminal/reaped before numerical repair')
 pids=[execution['pid']]+[r['pid']for r in execution['records']]
 if any(type(pid)is not int or Path('/proc',str(pid)).exists()for pid in pids):raise ValueError('All C39 controller/workers must be absent')
 sr=execution['prediction_seal'];seal=io.read_json(root,sr)
 if seal.get('schema')!='C39-public-SIDE-postprobe-prediction-seal-v01'or seal.get('completed')is not True or seal.get('count')!=64 or seal.get('model_executions')!=128 or seal.get('protocol')!=SIDE_record or seal.get('private_target_or_latent_access')is not False or seal.get('all_postpredictions_sealed_before_private_evaluation')is not True or seal.get('initial_prediction_seal')!=b['initial_prediction_seal']:raise ValueError('Exact complete oldSIDE128 seal required')
 initial=io.read_json(root,b['initial_prediction_seal']);initialrows={r['object_id']:r['result']for r in initial['objects']};siderows={r['object_id']:r['result']for r in seal['objects']};byid={r['object_id']:r for r in execution['records']};jobs=[]
 if len(seal.get('objects',[]))!=64 or any(r.get('action_id')!='middle_side_2N'for r in seal['objects'])or len(siderows)!=64 or len(byid)!=64 or set(siderows)!=set(initialrows)or set(byid)!=set(siderows):raise ValueError('Exact same64 original prefix/SIDE identities required')
 for r in execution['records']:
  if r.get('completed')is not True or r.get('reaped')is not True or r.get('exit_code')!=0 or r['result']!=siderows[r['object_id']]:raise ValueError('Every originalSIDE execution/hash must be retained')
  oldside.output_result(root,r,SIDE_record['sha256'])
 oldjobs={r['object_id']:r for r in sp['jobs']}
 if set(oldjobs)!=set(initialrows):raise ValueError('Samepublic identity inputs required')
 for kind in KINDS:
  for token in sorted(initialrows):
   j=oldjobs[token];job=dict(id=token+'__'+kind,kind=kind,object_id=token,prefix=j['prefix'],old_result=(initialrows if kind=='prefix'else siderows)[token])
   if kind=='SIDE':job['acquisition']=j['acquisition']
   jobs.append(job)
 return dict(initial_protocol=initial_record,SIDE_protocol=SIDE_record,initial_prediction_seal=b['initial_prediction_seal'],SIDE_prediction_seal=sr,SIDE_execution=er,model_inputs=sp['inputs'],taskset=ip['taskset'],jobs=jobs)


def prepare(protocol,output,*,initial_protocol,initial_protocol_sha256,SIDE_protocol,SIDE_protocol_sha256,root=ROOT):
 root=Path(root).resolve();dest=io.local(root,protocol);out=io.local(root,output)
 if dest.exists()or out.exists():raise FileExistsError('Fresh exclusive resolved protocol/output; no resume/retry')
 inputs=completed_inputs(root,dict(path=initial_protocol,sha256=initial_protocol_sha256),dict(path=SIDE_protocol,sha256=SIDE_protocol_sha256))
 p=dict(schema='RESOLVED-public-prefix-SIDE-inference-v01',status='FROZEN_AFTER_OLD_SEALS_BEFORE_RESOLVED_CALLS',output=output,inputs=inputs,source_sha256={n:io.sha(root/n)for n in sources()},runtime=io.runtime(),resources=RESOURCES,models=list(io.MODELS),retries=0,private_target_or_latent_access=False,new_observations_policies_or_population=False,all256_predictions_sealed_before_separate_private_analysis=True,repair='Same posterior law; coordinate quantiles require min(10nm,1e-8*range) and numericalCDF p±1e-6. NO posthoc interval widening.',raw_local=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest,*,root=ROOT,full_inputs=True):
 root=Path(root).resolve();p=io.read_json(root,dict(path=protocol,sha256=digest))
 if p.get('schema')!='RESOLVED-public-prefix-SIDE-inference-v01'or p.get('status')!='FROZEN_AFTER_OLD_SEALS_BEFORE_RESOLVED_CALLS'or p.get('resources')!=RESOURCES or p.get('runtime')!=io.runtime()or p.get('source_sha256')!={n:io.sha(root/n)for n in sources()}or p.get('models')!=list(io.MODELS)or p.get('retries')!=0 or p.get('private_target_or_latent_access')is not False or p.get('new_observations_policies_or_population')is not False or p.get('all256_predictions_sealed_before_separate_private_analysis')is not True:raise ValueError('Exact resolved source/runtime/resource/privacy contract')
 io.local(root,p['output']);i=p['inputs']
 if full_inputs and i!=completed_inputs(root,i['initial_protocol'],i['SIDE_protocol']):raise ValueError('Exact old sealed input/model/job closure')
 return p


def load_saved_result(root,files):
 """Only already sealed model posterior outputs, never truth arrays."""
 if set(files)!=FILES:raise ValueError('Exact old model bundle')
 for r in files.values():io.checked(root,r)
 tree=io.read_json(root,files['posterior-structure.json'])
 with np.load(io.checked(root,files['posterior-arrays.npz']),allow_pickle=False)as z:
  used=set()
  def unpack(v):
   if isinstance(v,dict)and set(v)=={'array','shape','dtype'}:
    a=z[v['array']].copy();used.add(v['array'])
    if list(a.shape)!=v['shape']or str(a.dtype)!=v['dtype']:raise ValueError('Savedarray structure/dtype mismatch')
    return a
   if isinstance(v,dict):return{k:unpack(x)for k,x in v.items()}
   if isinstance(v,list):return[unpack(x)for x in v]
   return v
  result=unpack(tree)
  if used!=set(z.files):raise ValueError('Exact saved posterior array-member closure')
 return result


def core_identity(old,new):
 """Only quantile/cache/endpoint diagnostics may differ; report every core miss."""
 misses=[]
 def compare(a,b,path):
  if isinstance(a,np.ndarray)or isinstance(b,np.ndarray):
   if not isinstance(a,np.ndarray)or not isinstance(b,np.ndarray)or a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():misses.append(path)
  elif isinstance(a,dict)and isinstance(b,dict):
   if set(a)!=set(b):misses.append(path+'/keys')
   for k in set(a)&set(b):compare(a[k],b[k],path+'/'+str(k))
  elif isinstance(a,(list,tuple))and isinstance(b,(list,tuple)):
   if len(a)!=len(b):misses.append(path+'/length')
   for i,(x,y)in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
  elif isinstance(a,(int,float,np.number))and isinstance(b,(int,float,np.number)):
   if np.float64(a).tobytes()!=np.float64(b).tobytes():misses.append(path)
  elif a!=b:misses.append(path)
 for name in prefix.METHODS:
  a=old.get('methods',{}).get(name,{});b=new.get('methods',{}).get(name,{})
  for key in sorted(CORE_KEYS):
   if key not in a or key not in b:misses.append(name+'/'+key+'/missing')
   else:compare(a[key],b[key],name+'/'+key)
 for key in ('object_id','categories','material_CDF_grid_Pa','canonical_point_ids','canonical_interval_point_ids','equal_tail_probabilities','gates'):
  if key not in old or key not in new:misses.append(key+'/missing')
  else:compare(old[key],new[key],key)
 return dict(passed=not misses,bit_mismatch_paths=misses,scope='Normalizer/category/material/allmoments bitidentity; excludes targetquantile values/brackets,cache counts and quantile-dependent endpointCDF diagnostics. A numerical failure is retained, never relabelled calibrated.')


def worker(protocol,digest,job_id,slot,*,root=ROOT):
 root=Path(root).resolve();p=validate(protocol,digest,root=root,full_inputs=False)
 if type(slot)is not int or not 0<=slot<12:raise ValueError('Declared CPU2 slot')
 io.guard(root,RESOURCES['worker_slots'][slot]);matches=[j for j in p['inputs']['jobs']if j['id']==job_id]
 if len(matches)!=1:raise ValueError('Exactly one declared prefix/SIDE job')
 j=matches[0];design=oldside.obs.load_frozen_design(root);public=oldside.prefix_codec.load_public_prefix(root,j['prefix'],design=design)
 if public['object_id']!=j['object_id']:raise ValueError('Exact public identity')
 acquired=None
 if j['kind']=='SIDE':
  with np.load(io.checked(root,j['acquisition']),allow_pickle=False)as z:
   if set(z.files)!=oldside.PROBE_KEYS:raise ValueError('Exactly ONE typed SIDE public acquisition')
   acquired=oldside.decode_probe(public,{k:z[k].copy()for k in oldside.PROBE_KEYS},design=design)
  if acquired['action_id']!='middle_side_2N':raise ValueError('SIDE only; no old DOWN outcomes')
 models=oldside.common.load_composites(root,dict(candidate=p['inputs']['model_inputs']['candidate_archive'],linear_library=p['inputs']['model_inputs']['linear_library']),p['inputs']['model_inputs']['side_inputs'])
 old=io.read_json(root,j['old_result'])
 if old.get('completed')is not True or old.get('object_id')!=j['object_id']or set(old.get('models',{}))!=set(io.MODELS):raise ValueError('Exact old result comparison identity')
 out=io.local(root,p['output'])/'objects'/job_id;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();state=dict(completed=False,pid=os.getpid(),slot=slot,object_id=j['object_id'],kind=j['kind'],protocol_sha256=digest,old_result=j['old_result'],models={},private_target_or_latent_access=False,not_new_population_test=True,raw_local=True)
 io.atomic(out/'result.json',state)
 for name,model in models.items():
  io.guard(root,RESOURCES['worker_slots'][slot]);d=out/name;d.mkdir()
  result=prefix.infer_public_prefix(model.initial_candidate,public,design=design)if j['kind']=='prefix'else side.infer_public_probe(model,public,acquired,design=design)
  if result.get('object_id',j['object_id'])!=j['object_id']:raise ValueError('Resolved result identity')
  io.save_full_result(d,result);files={n:io.record(root,d/n)for n in FILES};comparison=core_identity(load_saved_result(root,old['models'][name]['files']),load_saved_result(root,files));io.write_new(d/'core-bit-identity.json',comparison)
  state['models'][name]=dict(execution_completed=True,completed=result.get('completed')is True,numerical_controls_passed=result.get('numerical_controls_passed')is True,files=files,core_bit_identity=io.record(root,d/'core-bit-identity.json'),core_bit_identity_passed=comparison['passed']);state['seconds']=time.monotonic()-start;io.atomic(out/'result.json',state)
 state.update(completed=True,all_models_numerical_passed=all(r['completed']and r['numerical_controls_passed']and r['core_bit_identity_passed']for r in state['models'].values()),seconds=time.monotonic()-start);io.atomic(out/'result.json',state);return state


def output_result(root,row,digest):
 a=io.read_json(root,row['result'])
 if a.get('completed')is not True or a.get('pid')!=row['pid']or a.get('object_id')!=row['object_id']or a.get('kind')!=row['kind']or a.get('protocol_sha256')!=digest or set(a.get('models',{}))!=set(io.MODELS)or a.get('private_target_or_latent_access')is not False:raise ValueError('Exact complete resolved bundle identity')
 for v in a['models'].values():
  if v.get('execution_completed')is not True or set(v['files'])!=FILES:raise ValueError('All model outputs retained, including numerical failures')
  for f in v['files'].values():io.checked(root,f)
  c=io.read_json(root,v['core_bit_identity'])
  if c.get('passed')!=v.get('core_bit_identity_passed'):raise ValueError('Core comparison proof binding')
 return a


def run(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p=validate(protocol,digest,root=root);io.guard(root,RESOURCES['controller_cores']);out=io.local(root,p['output'])
 if out.exists():raise FileExistsError('No postinference overwrite/retry')
 out.mkdir(parents=True);(out/'objects').mkdir();(out/'logs').mkdir();start=time.monotonic();live={};stopped=False
 state=dict(completed=False,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),records=[],queued_job_ids=[j['id']for j in p['inputs']['jobs']],live=[],technical_failure=False,private_target_or_latent_access=False,raw_local=True,retries=0)
 jobs={j['id']:j for j in p['inputs']['jobs']}
 def save():state.update(seconds=time.monotonic()-start,live=[dict(pid=x['process'].pid,job_id=x['row']['job_id'],slot=k)for k,x in live.items()]);io.atomic(out/'execution.json',state)
 def stop(*args):raise KeyboardInterrupt('Owned resolved controller terminated')
 old_handlers={s:signal.signal(s,stop)for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
 save()
 try:
  while state['queued_job_ids']or live:
   io.guard(root,RESOURCES['controller_cores'])
   if time.monotonic()-start>RESOURCES['global_wall_s']:raise TimeoutError('Resolved global1800s cap')
   if not stopped:
    for slot in range(RESOURCES['max_workers']):
     if slot in live or not state['queued_job_ids']:continue
     job_id=state['queued_job_ids'].pop(0);j=jobs[job_id];log=(out/'logs'/(job_id+'.log')).open('xb');cmd=[p['inputs']['taskset']['path'],'-c',','.join(map(str,RESOURCES['worker_slots'][slot])),sys.executable,'-m','experiments.run_campaign_resolved_inference','worker','--protocol',protocol,'--protocol-sha256',digest,'--job-id',job_id,'--slot',str(slot)]
     try:proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     except BaseException:log.close();state['queued_job_ids'].insert(0,job_id);raise
     row=dict(job_id=job_id,object_id=j['object_id'],kind=j['kind'],pid=proc.pid,PGID=proc.pid,slot=slot,command=cmd,start_monotonic=time.monotonic());live[slot]=dict(process=proc,row=row,log=log);save()
   for slot,x in list(live.items()):
    proc=x['process'];row=x['row'];code=proc.poll();reason=None
    if code is None:
     if time.monotonic()-row['start_monotonic']>RESOURCES['worker_wall_s']:reason='600s per resolved job cap'
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
  if not state['technical_failure']and len(state['records'])==128 and not state['queued_job_ids']:
   for row in state['records']:output_result(root,row,digest)
   validate(protocol,digest,root=root)
   seal=dict(schema='RESOLVED-prefix-SIDE-prediction-seal-v01',completed=True,count=128,model_executions=256,objects=sorted([dict(object_id=r['object_id'],kind=r['kind'],result=r['result'])for r in state['records']],key=lambda r:(r['kind'],r['object_id'])),initial_prediction_seal=p['inputs']['initial_prediction_seal'],old_SIDE_prediction_seal=p['inputs']['SIDE_prediction_seal'],not_new_population_test=True,quantile_refinement_not_interval_widening=True,protocol=dict(path=protocol,sha256=digest),private_target_or_latent_access=False,all_postpredictions_sealed_before_private_evaluation=True,all_numerical_controls_passed=all(r['all_models_numerical_passed']for r in state['records']),raw_local=True)
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
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run','worker'));a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--initial-protocol');a.add_argument('--initial-protocol-sha256');a.add_argument('--SIDE-protocol');a.add_argument('--SIDE-protocol-sha256');a.add_argument('--job-id');a.add_argument('--slot',type=int);v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output,initial_protocol=v.initial_protocol,initial_protocol_sha256=v.initial_protocol_sha256,SIDE_protocol=v.SIDE_protocol,SIDE_protocol_sha256=v.SIDE_protocol_sha256))
 elif v.mode=='worker':worker(v.protocol,v.protocol_sha256,v.job_id,v.slot)
 else:
  s=run(v.protocol,v.protocol_sha256);print(json.dumps({'completed':s['completed']}));sys.exit(0 if s['completed']else 1)
