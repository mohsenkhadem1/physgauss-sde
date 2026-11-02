"""Bounded, resumable public-only fit/forecast supervisor. Never scores targets.

Durable child workers own stage completion records, so parent restart does not
repeat a running or failed stage. Existing unexplained outputs are not reused.
"""
import argparse, concurrent.futures, fcntl, json, os, signal, subprocess, sys, time
from pathlib import Path
from scripts.compositional_dynamics_v17.prototype import ROOT,sha,write
from scripts.compositional_dynamics_v17.cohort_shared_fit import guard
BASE=ROOT/'runs/compositional-dynamics-v17/fresh-cohort'
PIPE=BASE/'pipeline';RENDER=ROOT/'runs/compositional-dynamics-v17/render-env/bin/python';PYTHON=ROOT/'.venv/bin/python'


def checked(bindings):
 p,r=guard(bindings);f=json.loads(Path(bindings['freeze_path']).read_text())
 if f['source_sha256'].get(str(Path(__file__).relative_to(ROOT)))!=sha(__file__):raise ValueError('Pipeline source not frozen')
 return p,r

def flags(bindings):
 return [item for key,value in bindings.items()for item in ['--'+key.replace('_','-'),str(value)]]

def process_identity(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().split();return None if fields[2]=='Z'else fields[21]
 except FileNotFoundError:return None

def child(request_path):
 request_path=Path(request_path);r=json.loads(request_path.read_text());checked(r['bindings']);status=Path(r['status_path']);start=time.time();log=Path(r['log_path']);log.parent.mkdir(parents=True,exist_ok=True)
 record=dict(stage=r['stage'],request_sha256=sha(request_path),state='running',worker_pid=os.getpid(),worker_start=process_identity(os.getpid()),started_epoch=start)
 write(status,record);process=None
 try:
  remaining=min(r['timeout_seconds'],r['deadline_epoch']-time.time())
  if remaining<=0:raise TimeoutError('Original pipeline deadline exhausted')
  env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
  with log.open('ab')as output:
   process=subprocess.Popen(r['command'],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   record.update(command_pid=process.pid,command_start=process_identity(process.pid));write(status,record)
   try:code=process.wait(timeout=remaining)
   except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    raise TimeoutError('Bounded stage timeout')
  record.update(exit_code=code,completed=code==0)
 except Exception as ex:record.update(completed=False,error=f'{type(ex).__name__}: {ex}')
 finally:
  record.update(state='terminal',finished_epoch=time.time(),seconds=time.time()-start);write(status,record)

def run_stage(stage,command,timeout,bindings,deadline):
 locks=PIPE/'locks';locks.mkdir(parents=True,exist_ok=True)
 with (locks/(stage+'.lock')).open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _run_stage(stage,command,timeout,bindings,deadline)

def _run_stage(stage,command,timeout,bindings,deadline):
 directory=PIPE/'stages'/stage;request=directory/'request.json';status=directory/'status.json'
 if request.exists():
  old=json.loads(request.read_text())
  if old['command']!=[str(x)for x in command]or old['bindings']!=bindings or old['deadline_epoch']!=deadline:raise ValueError('Existing stage request differs')
 else:
  if time.time()>=deadline:raise TimeoutError('Original pipeline deadline exhausted before stage')
  directory.mkdir(parents=True,exist_ok=False);write(request,dict(stage=stage,command=[str(x)for x in command],timeout_seconds=timeout,bindings=bindings,deadline_epoch=deadline,status_path=str(status),log_path=str(directory/'output.log')))
  process=subprocess.Popen([str(PYTHON),'-m','scripts.compositional_dynamics_v17.cohort_pipeline','--child',str(request)],cwd=ROOT,start_new_session=True)
  write(directory/'launch.json',dict(pid=process.pid,start=process_identity(process.pid),request_sha256=sha(request)))
 while True:
  if status.exists():
   record=json.loads(status.read_text())
   if record['request_sha256']!=sha(request):raise ValueError('Child stage request changed')
   if record['state']=='terminal':return record
   pid=record['worker_pid'];stamp=record['worker_start']
  else:
   launch=directory/'launch.json'
   if not launch.exists():raise ValueError('Interrupted launch without recorded ownership; no blind rerun')
   state=json.loads(launch.read_text());pid=state['pid'];stamp=state['start']
  if process_identity(pid)!=stamp:
   # Never redo an interrupted fit; preserve outputs and declare failure.
   result=dict(stage=stage,state='terminal',completed=False,error='Recorded stage worker exited without terminal record',request_sha256=sha(request));write(status,result);return result
  if time.time()>deadline+15:raise TimeoutError('Pipeline original deadline exhausted')
  time.sleep(2)

def public_completion(name,bindings,deadline):
 output=BASE/'public'/name;status=output/'complete.json';source=BASE/'references'/name/'calibration/n4-dt0.00002500/complete.json';generator=ROOT/'reports/compositional-dynamics-v17/fresh-cohort-generation.json'
 while not status.exists():
  checked(bindings)
  if source.exists():
   row=json.loads(source.read_text())
   if row.get('world_id')!=name or row.get('role')!='calibration'or row.get('protocol_sha256')!=bindings['protocol_sha']or row.get('method_freeze_sha256')!=bindings['freeze_sha']or type(row.get('completed'))is not bool:raise ValueError('Calibration identity corrupted')
   command=[RENDER,'-m','scripts.compositional_dynamics_v17.cohort_observations','--world',name,'--protocol',bindings['protocol_path'],'--freeze',bindings['freeze_path'],'--approval',bindings['approval_path'],'--protocol-sha',bindings['protocol_sha'],'--freeze-sha',bindings['freeze_sha']]
   result=run_stage('export-'+name,command,300,bindings,deadline)
   if not status.exists():raise ValueError('Exporter did not produce declared terminal public status: '+str(result))
   break
  if generator.exists():
   report=json.loads(generator.read_text())
   if report.get('protocol_sha256')!=bindings['protocol_sha']or report.get('method_freeze_sha256')!=bindings['freeze_sha']:raise ValueError('Generator identity corrupted')
   if report.get('terminal'):
    if name+'/calibration/n4-dt0.00002500' not in report.get('missing_keys',[]):raise ValueError('Generator terminal lacks calibration and missing identity')
    output.mkdir(parents=True,exist_ok=False);write(status,dict(world_id=name,completed=False,protocol_sha256=bindings['protocol_sha'],method_freeze_sha256=bindings['freeze_sha'],error='Generator terminal with missing declared calibration',generator_terminal_sha256=sha(generator)));break
  if time.time()>=deadline:raise TimeoutError('No terminal calibration within pipeline budget')
  time.sleep(5)
 record=json.loads(status.read_text())
 if record.get('world_id')!=name or record.get('protocol_sha256')!=bindings['protocol_sha']or record.get('method_freeze_sha256')!=bindings['freeze_sha']or type(record.get('completed'))is not bool:raise ValueError('Public completion identity mismatch')
 return record

def native_commit(name,role,bindings,output,error=None):
 folder=output/'forecasts'/role;commit=folder/'prediction-commit.json'
 if commit.exists():
  existing=json.loads(commit.read_text())
  if existing.get('protocol_sha256')!=bindings['protocol_sha']or existing.get('method_freeze_sha256')!=bindings['freeze_sha']or existing.get('world_id')!=name or existing.get('role')!=role:raise ValueError('Existing native outer commit corrupted')
  return existing
 folder.mkdir(parents=True,exist_ok=True);result=dict(world_id=name,role=role,protocol_sha256=bindings['protocol_sha'],method_freeze_sha256=bindings['freeze_sha'],completed=False,terminal=True)
 status=folder/'complete.json'
 if error:result['error']=error
 elif status.exists():
  row=json.loads(status.read_text());plan=PIPE/'native-plans'/f'{name}.json'
  if row.get('plan_sha256')!=sha(plan):raise ValueError('Native per-world replay plan mismatch')
  result['native_complete_sha256']=sha(status)
  if row.get('completed'):
   prediction=folder/'prediction.npz'
   if row.get('prediction_sha256')!=sha(prediction):raise ValueError('Native prediction digest corrupted')
   result.update(completed=True,prediction_sha256=sha(prediction),path=str(prediction.relative_to(ROOT)))
  else:result['error']=row.get('error','Native replay incomplete')
 else:result['error']='Native replay missing terminal output'
 write(commit,result);return result

def gpu_ready(deadline):
 while time.time()<deadline:
  text=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);free,util=map(int,text.strip().split(','))
  if free>=6144 and util<=90:return dict(time_epoch=time.time(),gpu=0,free_MiB=free,utilization_percent=util)
  time.sleep(10)
 raise TimeoutError('GPU0 headroom unavailable within original deadline')

def native_lane(p,bindings,deadline):
 results=[]
 for world in p['native_worlds']:
  name=world['name'];output=ROOT/world['output_directory'];plan=PIPE/'native-plans'/f'{name}.json';error=None
  try:
   status=public_completion(name,bindings,deadline)
   if not status['completed']:raise RuntimeError('Calibration unavailable: '+status.get('error','generator failure'))
   result=run_stage('native-prepare-'+name,[RENDER,'-m','scripts.compositional_dynamics_v17.native_phystwin_cohort_adapter','prepare','--plan',plan,'--cohort-plan',bindings['protocol_path'],'--world',name],60,bindings,deadline)
   if not result['completed']:raise RuntimeError('Native prepare failed')
   if not (PIPE/'stages'/('native-fit-'+name)/'request.json').exists():
    census=gpu_ready(deadline);write(PIPE/'native-plans'/f'{name}-gpu-census-{time.time_ns()}.json',census)
   result=run_stage('native-fit-'+name,[RENDER,'-m','scripts.compositional_dynamics_v17.native_phystwin_cohort_adapter','fit','--plan',plan],3700,bindings,deadline)
   if not result['completed']:raise RuntimeError('Native fit failed or exhausted budget')
  except (RuntimeError,TimeoutError)as ex:error=f'{type(ex).__name__}: {ex}'
  for role in ('forecast-40mm','forecast-80mm'):
   reason=error
   if not reason:
    try:
     result=run_stage('native-forecast-'+name+'-'+role,[RENDER,'-m','scripts.compositional_dynamics_v17.native_phystwin_cohort_adapter','predict','--plan',plan,'--query',role],300,bindings,deadline)
     if not result['completed']:reason='Native replay process failed or timed out'
    except TimeoutError as ex:reason=str(ex)
   results.append(native_commit(name,role,bindings,output,reason))
 return results

def shared_lane(p,recipe,bindings,deadline):
 for name in p['generator_settings']['world_ids']:public_completion(name,bindings,deadline)
 result=run_stage('shared-fit',[PYTHON,'-m','scripts.compositional_dynamics_v17.cohort_shared_fit','--stage','fit',*flags(bindings)],10800,bindings,deadline)
 def one(pair):
  name,role=pair
  try:return run_stage('shared-forecast-'+name+'-'+role,[PYTHON,'-m','scripts.compositional_dynamics_v17.cohort_shared_fit','--stage','forecast','--world',name,'--role',role,*flags(bindings)],600,bindings,deadline)
  except TimeoutError as ex:return dict(completed=False,error=str(ex))
 pairs=[(n,r)for n in p['generator_settings']['world_ids']for r in ('forecast-40mm','forecast-80mm')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:forecasts=list(pool.map(one,pairs))
 # A killed forecast process cannot write its own terminal failure commit.
 for (name,role),record in zip(pairs,forecasts):
  folder=ROOT/recipe['forecast_root']/name/role;commit=folder/'prediction-commit.json'
  if not commit.exists():
   folder.mkdir(parents=True,exist_ok=True);write(commit,dict(world_id=name,role=role,protocol_sha256=bindings['protocol_sha'],method_freeze_sha256=bindings['freeze_sha'],completed=False,terminal=True,error='Forecast process failed/missing commitment',stage_status=record))
 return dict(fit_stage=result,forecasts=forecasts)

def final_census(p,recipe,bindings,result):
 records=[];errors=[]
 for name in p['generator_settings']['world_ids']:
  for method in ('shared','native'):
   root=ROOT/recipe['forecast_root']/name if method=='shared'else ROOT/next(w['output_directory']for w in p['native_worlds']if w['name']==name)/'forecasts'
   for role in ('forecast-40mm','forecast-80mm'):
    folder=root/role;commit=folder/'prediction-commit.json'
    if not commit.exists():
     folder.mkdir(parents=True,exist_ok=True)
     write(commit,dict(world_id=name,role=role,protocol_sha256=bindings['protocol_sha'],method_freeze_sha256=bindings['freeze_sha'],completed=False,terminal=True,error='No terminal prediction commitment before pipeline ended',pipeline_lane=result['lanes'].get(method)))
    row=json.loads(commit.read_text())
    valid=row.get('world_id')==name and row.get('role')==role and row.get('protocol_sha256')==bindings['protocol_sha']and row.get('method_freeze_sha256')==bindings['freeze_sha']and type(row.get('completed'))is bool
    if not valid:errors.append(dict(method=method,world_id=name,role=role,error='Existing commit identity corrupted; preserved unchanged'))
    records.append(dict(method=method,world_id=name,role=role,path=str(commit.relative_to(ROOT)),sha256=sha(commit),identity_valid=valid,completed=row.get('completed')))
 return dict(expected=16,present=len(records),all_identities_valid=not errors,errors=errors,records=records)

def main(bindings):
 p,recipe=checked(bindings);generation=BASE/'private/runtime-input-freeze.json'
 if not generation.exists():raise ValueError('Generator must initialize before pipeline; no premature output creation')
 generation_identity=json.loads(generation.read_text())
 if generation_identity.get('protocol_sha256')!=bindings['protocol_sha']or generation_identity.get('method_freeze_sha256')!=bindings['freeze_sha']:raise ValueError('Generation identity mismatch')
 PIPE.mkdir(parents=True,exist_ok=True);lock=(PIPE/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runtime=PIPE/'runtime.json'
 if runtime.exists():
  record=json.loads(runtime.read_text())
  if record['bindings']!=bindings:raise ValueError('Existing pipeline bindings differ')
 else:
  record=dict(bindings=bindings,started_epoch=time.time(),deadline_epoch=time.time()+18000,scope='Public-only inference and committed forecasts; no scoring, private parameters or forecast targets');write(runtime,record)
 deadline=record['deadline_epoch'];result=dict(protocol_sha256=bindings['protocol_sha'],method_freeze_sha256=bindings['freeze_sha'],runtime_sha256=sha(runtime),terminal=False,lanes={})
 write(PIPE/'complete.json',result)
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
   jobs={pool.submit(shared_lane,p,recipe,bindings,deadline):'shared',pool.submit(native_lane,p,bindings,deadline):'native'}
   for future in concurrent.futures.as_completed(jobs):
    name=jobs[future]
    try:result['lanes'][name]=dict(completed=True,result=future.result())
    except Exception as ex:result['lanes'][name]=dict(completed=False,error=f'{type(ex).__name__}: {ex}')
    write(PIPE/'complete.json',result)
 finally:
  result['forecast_census']=final_census(p,recipe,bindings,result)
  result.update(terminal=True,finished_epoch=time.time());write(PIPE/'complete.json',result)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--child')
 for key in ('protocol-path','freeze-path','approval-path','protocol-sha','freeze-sha'):parser.add_argument('--'+key)
 args=vars(parser.parse_args());request=args.pop('child')
 if request:child(request)
 else:
  if any(v is None for v in args.values()):parser.error('All protocol/freeze/approval bindings required')
  main(args)
