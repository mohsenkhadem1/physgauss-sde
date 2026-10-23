"""Exact-support-tail, repair-FAILED-arms continuation; successful v01 arms reused exactly.

Metadata prepare requires every v01 arm sealed/reaped. Only failed model arms
are recomputed, each from the SAME public observations and original prior. No
truth API, policy/noise generation, interval inflation or physical-model change.
"""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,time,traceback
import numpy as np
from experiments import run_campaign_resolved_inference as old
from experiments import campaign_resolved_prefix_inference_v03 as prefix
from experiments import campaign_resolved_SIDE_inference_v03 as side
io=old.io
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_campaign_resolved_repair_v03.py'
RESOURCES={**old.RESOURCES,'global_wall_s':900,'max_workers':2,'worker_slots':[[6,7],[8,9]]}
HELPER='experiments/campaign_resolved_quantiles_v03.py'
HELPER_SHA='e5a9779c74224d4e3ddb809a9c09fdcfbd2341b41bd11f58fffc261b5154b547'


def sources():return sorted(set(old.sources())|{SOURCE,HELPER,'experiments/campaign_resolved_prefix_inference_v03.py','experiments/campaign_resolved_SIDE_inference_v03.py','tests/test_run_campaign_resolved_repair_v03.py','tests/test_campaign_resolved_quantiles_v03.py','tests/test_campaign_resolved_inference_v03.py'})


def v02_lineage(root):
 base='runs/gaussian-campaign-v01/resolved-guard-repair-v02/'
 refs={n:io.record(root,io.local(root,base+n))for n in ('execution.json','terminal-executions.json','prediction-seal.json')}
 e=io.read_json(root,refs['execution.json']);seal=io.read_json(root,refs['prediction-seal.json'])
 if e.get('completed')is not True or e.get('all_owned_workers_reaped')is not True or e.get('technical_failure')is not False or e.get('live')!=[]:raise ValueError('v02 must be fully sealed and reaped')
 if any(Path('/proc',str(pid)).exists()for pid in [e['pid']]+[r['pid']for r in e['records']]):raise ValueError('All v02 PIDs absent')
 if seal.get('model_executions')!=256 or seal.get('denominators',{}).get('merged_numerical_negative_arms')!=2:raise ValueError('Preserve exact two v02 numerical negatives')
 negatives=[r for r in seal['model_records']if not(r['completed']and r['numerical_controls_passed']and r['core_bit_identity_passed'])]
 if len(negatives)!=2 or any(r['kind']!='prefix'or r['model']!='linear_rest_tangent' for r in negatives):raise ValueError('Only two linear PREFIX support-tail failures')
 for r in negatives:
  for f in r['files'].values():io.checked(root,f)
 return dict(refs=refs,negative_model_records=negatives,private_truth_access=False)


def reuse_plan(root,original):
 """No array members read: exact all256 old hashes and PID closure first."""
 p=old.validate(original['path'],original['sha256'],root=root);out=io.local(root,p['output']);er=io.record(root,out/'execution.json');e=io.read_json(root,er)
 if e.get('completed')is not True or e.get('technical_failure')is not False or e.get('all_owned_workers_reaped')is not True or e.get('protocol')!=original or e.get('queued_job_ids')!=[]or e.get('live')!=[]or len(e.get('records',[]))!=128:raise ValueError('ALLv01 jobs sealed/reaped before v02 reuse/repair')
 pids=[e['pid']]+[r['pid']for r in e['records']]
 if any(type(pid)is not int or Path('/proc',str(pid)).exists()for pid in pids):raise ValueError('All129 v01 PIDs must be absent')
 sr=e['prediction_seal'];s=io.read_json(root,sr)
 if s.get('schema')!='RESOLVED-prefix-SIDE-prediction-seal-v01'or s.get('completed')is not True or s.get('count')!=128 or s.get('model_executions')!=256 or s.get('protocol')!=original or s.get('private_target_or_latent_access')is not False or s.get('not_new_population_test')is not True:raise ValueError('Exact v01 256model seal required')
 expected={(j['kind'],j['object_id']):j for j in p['inputs']['jobs']};sealed={(r['kind'],r['object_id']):r for r in s['objects']};rows={(r['kind'],r['object_id']):r for r in e['records']}
 if len(expected)!=128 or len(s['objects'])!=128 or len(sealed)!=128 or set(expected)!=set(sealed)or set(rows)!=set(expected):raise ValueError('Exact128 prefix/SIDE job join')
 reused=[];jobs=[];den=dict(total_model_arms=256,v01_completed_numeric_core_pass=0,v01_failed_model_arms=0,v01_core_bit_mismatches=0)
 for key in sorted(expected):
  j=expected[key];row=rows[key]
  if row.get('completed')is not True or row.get('reaped')is not True or row.get('exit_code')!=0 or row['result']!=sealed[key]['result']:raise ValueError('Every v01 execution result retained')
  result=old.output_result(root,row,original['sha256'])
  for name in io.MODELS:
   v=result['models'][name];proof=io.read_json(root,v['core_bit_identity'])
   if v.get('core_bit_identity_passed')is not True or proof.get('passed')is not True:den['v01_core_bit_mismatches']+=1;raise ValueError('Unexplained corebit mismatch is not this approved guard-only repair')
   entry=dict(object_id=j['object_id'],kind=j['kind'],model=name,files=v['files'],core_bit_identity=v['core_bit_identity'],completed=v.get('completed')is True,numerical_controls_passed=v.get('numerical_controls_passed')is True,core_bit_identity_passed=True,source_result=row['result'],source_protocol=original)
   if entry['completed']and entry['numerical_controls_passed']:
    den['v01_completed_numeric_core_pass']+=1;reused.append(dict(entry,origin='REUSED_v01'))
   else:
    den['v01_failed_model_arms']+=1;jobs.append(dict(id=j['id']+'__'+name,object_id=j['object_id'],kind=j['kind'],model=name,public_job=j,v01_failed_model=entry))
 if len(reused)+len(jobs)!=256:raise ValueError('No dropping v01 model arms')
 return dict(v02_lineage=v02_lineage(root),original_protocol=original,original_execution=er,original_prediction_seal=sr,original_PID_list=pids,model_inputs=p['inputs']['model_inputs'],taskset=p['inputs']['taskset'],reused_models=reused,jobs=jobs,denominators=den)


def prepare(protocol,output,*,original_protocol,original_protocol_sha256,root=ROOT):
 root=Path(root).resolve();dest=io.local(root,protocol)
 if dest.exists()or io.local(root,output).exists():raise FileExistsError('Fresh exclusive guard-repair protocol/output')
 io.checked(root,dict(path=HELPER,sha256=HELPER_SHA));plan=reuse_plan(root,dict(path=original_protocol,sha256=original_protocol_sha256))
 p=dict(schema='RESOLVED-failed-model-repair-execution-v03',status='FROZEN_AFTER_ALL_V01_SEALS_BEFORE_REPAIR',output=output,inputs=plan,source_sha256={n:io.sha(root/n)for n in sources()},runtime=io.runtime(),resources=RESOURCES,models=list(io.MODELS),retries=0,private_target_or_latent_access=False,not_new_population_test=True,repair_only_failed_model_arms=True,reuse_successful_v01_by_hash=True,helper=dict(path=HELPER,sha256=HELPER_SHA),rule='Use exact zero/one CDF ONLY strictly outside rational polynomial support; equality/interior delegate unchanged v02. Root controls and posterior unchanged.',raw_local=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest,*,root=ROOT,full_inputs=True):
 root=Path(root).resolve();p=io.read_json(root,dict(path=protocol,sha256=digest))
 if p.get('schema')!='RESOLVED-failed-model-repair-execution-v03'or p.get('status')!='FROZEN_AFTER_ALL_V01_SEALS_BEFORE_REPAIR'or p.get('source_sha256')!={n:io.sha(root/n)for n in sources()}or p.get('runtime')!=io.runtime()or p.get('resources')!=RESOURCES or p.get('models')!=list(io.MODELS)or p.get('retries')!=0 or p.get('private_target_or_latent_access')is not False or p.get('not_new_population_test')is not True or p.get('repair_only_failed_model_arms')is not True or p.get('reuse_successful_v01_by_hash')is not True or p.get('helper')!=dict(path=HELPER,sha256=HELPER_SHA):raise ValueError('Exact immutable v02 guard-only reuse/repair contract')
 io.local(root,p['output']);r=p['inputs']['original_protocol']
 if full_inputs and p['inputs']!=reuse_plan(root,r):raise ValueError('Full old256 hash-bound model reuse/failed job closure')
 return p


def worker(protocol,digest,job_id,slot,*,root=ROOT):
 root=Path(root).resolve();p=validate(protocol,digest,root=root,full_inputs=False)
 if type(slot)is not int or not 0<=slot<2:raise ValueError('Declared CPU2 repair slot')
 io.guard(root,RESOURCES['worker_slots'][slot]);jobs=[j for j in p['inputs']['jobs']if j['id']==job_id]
 if len(jobs)!=1:raise ValueError('Only declared failed SINGLEmodel arm can run')
 j=jobs[0];public_job=j['public_job'];design=old.oldside.obs.load_frozen_design(root);public=old.oldside.prefix_codec.load_public_prefix(root,public_job['prefix'],design=design)
 if public['object_id']!=j['object_id']:raise ValueError('Same acquired public identity')
 acquired=None
 if j['kind']=='SIDE':
  with np.load(io.checked(root,public_job['acquisition']),allow_pickle=False)as z:
   if set(z.files)!=old.oldside.PROBE_KEYS:raise ValueError('Strict ONE typed SIDE acquisition')
   acquired=old.oldside.decode_probe(public,{k:z[k].copy()for k in z.files},design=design)
  if acquired['action_id']!='middle_side_2N':raise ValueError('SIDE only; no DOWN outcomes')
 inputs=p['inputs']['model_inputs'];models=old.oldside.common.load_composites(root,dict(candidate=inputs['candidate_archive'],linear_library=inputs['linear_library']),inputs['side_inputs']);model=models[j['model']]
 out=io.local(root,p['output'])/'objects'/job_id;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();state=dict(completed=False,pid=os.getpid(),slot=slot,object_id=j['object_id'],kind=j['kind'],model=j['model'],protocol_sha256=digest,private_target_or_latent_access=False,origin='REPAIRED_v03',v01_failed_model=j['v01_failed_model'],raw_local=True);io.atomic(out/'result.json',state)
 result=prefix.infer_public_prefix(model.initial_candidate,public,design=design)if j['kind']=='prefix'else side.infer_public_probe(model,public,acquired,design=design)
 if result.get('object_id',j['object_id'])!=j['object_id']:raise ValueError('Repair result identity')
 io.save_full_result(out,result);files={n:io.record(root,out/n)for n in old.FILES};original=io.read_json(root,public_job['old_result']);comparison=old.core_identity(old.load_saved_result(root,original['models'][j['model']]['files']),old.load_saved_result(root,files));io.write_new(out/'core-bit-identity.json',comparison)
 state.update(completed=True,model_completed=result.get('completed')is True,numerical_controls_passed=result.get('numerical_controls_passed')is True,files=files,core_bit_identity=io.record(root,out/'core-bit-identity.json'),core_bit_identity_passed=comparison['passed'],seconds=time.monotonic()-start);state['all_models_numerical_passed']=state['model_completed']and state['numerical_controls_passed']and state['core_bit_identity_passed'];io.atomic(out/'result.json',state);return state


def output_result(root,row,digest):
 r=io.read_json(root,row['result'])
 if r.get('completed')is not True or r.get('pid')!=row['pid']or r.get('object_id')!=row['object_id']or r.get('kind')!=row['kind']or r.get('model')!=row['model']or r.get('protocol_sha256')!=digest or r.get('private_target_or_latent_access')is not False or r.get('origin')!='REPAIRED_v03'or set(r.get('files',{}))!=old.FILES:raise ValueError('Complete bound SINGLEmodel repair result')
 for f in r['files'].values():io.checked(root,f)
 c=io.read_json(root,r['core_bit_identity'])
 if c.get('passed')!=r.get('core_bit_identity_passed'):raise ValueError('Repair corebit proof binding')
 return r


def merged_records(root,p,state,digest):
 records=list(p['inputs']['reused_models']);jobs={j['id']:j for j in p['inputs']['jobs']}
 if len(state['records'])!=len(jobs)or {r['job_id']for r in state['records']}!=set(jobs):raise ValueError('Every declared failed arm retained before merge')
 for row in state['records']:
  r=output_result(root,row,digest)
  if row.get('reaped')is not True or row.get('exit_code')!=0:raise ValueError('Reaped repair execution before merge')
  records.append(dict(object_id=r['object_id'],kind=r['kind'],model=r['model'],origin='REPAIRED_v03',files=r['files'],core_bit_identity=r['core_bit_identity'],completed=r['model_completed'],numerical_controls_passed=r['numerical_controls_passed'],core_bit_identity_passed=r['core_bit_identity_passed'],source_result=row['result'],source_protocol=dict(path=state['protocol']['path'],sha256=digest)))
 expected={(j['kind'],j['object_id'],name)for j in old.validate(p['inputs']['original_protocol']['path'],p['inputs']['original_protocol']['sha256'],root=root)['inputs']['jobs']for name in io.MODELS};keys={(r['kind'],r['object_id'],r['model'])for r in records}
 if len(records)!=256 or len(keys)!=256 or keys!=expected:raise ValueError('Exact128jobs×2models merged denominator')
 for r in records:
  for f in r['files'].values():io.checked(root,f)
  io.checked(root,r['source_result']);io.checked(root,r['source_protocol']);proof=io.read_json(root,r['core_bit_identity'])
  if proof.get('passed')!=r['core_bit_identity_passed']:raise ValueError('Merged coreproof hash binding')
 return sorted(records,key=lambda r:(r['kind'],r['object_id'],r['model']))


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
   if time.monotonic()-start>RESOURCES['global_wall_s']:raise TimeoutError('Guard-repair global900s cap')
   if not stopped:
    for slot in range(RESOURCES['max_workers']):
     if slot in live or not state['queued_job_ids']:continue
     job_id=state['queued_job_ids'].pop(0);j=jobs[job_id];log=(out/'logs'/(job_id+'.log')).open('xb');cmd=[p['inputs']['taskset']['path'],'-c',','.join(map(str,RESOURCES['worker_slots'][slot])),sys.executable,'-m','experiments.run_campaign_resolved_repair_v03','worker','--protocol',protocol,'--protocol-sha256',digest,'--job-id',job_id,'--slot',str(slot)]
     try:proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     except BaseException:log.close();state['queued_job_ids'].insert(0,job_id);raise
     row=dict(job_id=job_id,object_id=j['object_id'],kind=j['kind'],model=j['model'],pid=proc.pid,PGID=proc.pid,slot=slot,command=cmd,start_monotonic=time.monotonic());live[slot]=dict(process=proc,row=row,log=log);save()
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
  if not state['technical_failure']and len(state['records'])==len(jobs)and not state['queued_job_ids']:
   validate(protocol,digest,root=root);records=merged_records(root,p,state,digest)
   terminal=dict(completed=True,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),records=state['records'],all_owned_workers_reaped=True,private_target_or_latent_access=False,retries=0)
   io.write_new(out/'terminal-executions.json',terminal)
   seal=dict(schema='RESOLVED-v03-merged-model-seal-v01',completed=True,count=128,model_executions=256,model_records=records,v02_lineage=p['inputs']['v02_lineage'],original_v01_execution=p['inputs']['original_execution'],original_v01_prediction_seal=p['inputs']['original_prediction_seal'],repair_execution=io.record(root,out/'terminal-executions.json'),protocol=dict(path=protocol,sha256=digest),denominators={**p['inputs']['denominators'],'v03_repaired_model_arms':len(jobs),'reused_v01_model_arms':len(p['inputs']['reused_models']),'merged_numerical_negative_arms':sum(not(r['completed']and r['numerical_controls_passed']and r['core_bit_identity_passed'])for r in records)},private_target_or_latent_access=False,not_new_population_test=True,all_merged_predictions_sealed_before_separate_private_analysis=True,all_numerical_controls_passed=all(r['completed']and r['numerical_controls_passed']and r['core_bit_identity_passed']for r in records),raw_local=True)
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
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run','worker'));a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--original-protocol');a.add_argument('--original-protocol-sha256');a.add_argument('--job-id');a.add_argument('--slot',type=int);v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output,original_protocol=v.original_protocol,original_protocol_sha256=v.original_protocol_sha256))
 elif v.mode=='worker':worker(v.protocol,v.protocol_sha256,v.job_id,v.slot)
 else:
  s=run(v.protocol,v.protocol_sha256);print(json.dumps({'completed':s['completed']}));sys.exit(0 if s['completed']else 1)
