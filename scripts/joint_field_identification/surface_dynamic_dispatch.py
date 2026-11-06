"""Operational-only dynamic queue for unchanged frozen surface SMC jobs.

Pause only verified old surface scheduler parents. Let every running child finish,
then terminate paused parents at boundaries. No old-v01 worker or GPU3 changes.
"""
from pathlib import Path
import os,signal,subprocess,sys,time,json,hashlib,datetime,argparse
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/joint-field-surface-v04/continuous';RUN=ROOT/'runs/joint-field-surface-v04/continuous'
PARENTS=[313446,334328,334327]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def key(j):return f"{j['shape']}-{j['population']}-case{j['case']:03d}-{j['stage']}-n{j['particles']}-s{j['seed']}"
def proc(pid):
 p=Path(f'/proc/{pid}')
 if not p.exists():return None
 try:
  command=(p/'cmdline').read_bytes().split(b'\0');stat=(p/'stat').read_text();state=stat[stat.rfind(')')+2:].split()[0]
  children=[int(v) for v in (p/'task'/str(pid)/'children').read_text().split()]
  return dict(pid=pid,command=[v.decode() for v in command if v],state=state,children=children)
 except FileNotFoundError:return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--transition',action='store_true');args=ap.parse_args()
 manifestpath=REPORT/'manifest.json';manifest=read(manifestpath);digest=sha(manifestpath)
 jobs=manifest['allowed_jobs'];names=[key(j) for j in jobs];assert len(names)==len(set(names))
 deadline=datetime.datetime.fromisoformat(manifest['deadline_utc'].replace('Z','+00:00')).timestamp()
 statepath=REPORT/'dynamic-dispatch.json';old=read(statepath) if statepath.exists() else {}
 if old and old['manifest_sha256']!=digest:raise RuntimeError('Manifest changed')
 # Atomic exclusive process lock. Stale locks require deliberate human/agent review.
 lock=REPORT/'dynamic-dispatch.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 failures=old.get('failures',{});running={};logs={};state=dict(pid=os.getpid(),manifest_sha256=digest,started_utc=now(),failures=failures,source_sha256=sha(__file__),scope='Scheduling-only; no particle/move/batch/seed/evidence changes, one surface worker perGPU0-2.',completed=[],pending=[],running={})
 def status(label):
  state.update(status=label,updated_utc=now(),running={str(g):{k:v for k,v in d.items() if k!='process'} for g,d in running.items()});write(statepath,state)
 def verify():
  assert sha(manifestpath)==digest
  for f,h in manifest['source_sha256'].items():
   if sha(ROOT/f)!=h:raise RuntimeError('Frozen source changed: '+f)
 def completed(name):
  p=REPORT/(name+'.json')
  if not p.exists():return False
  try:d=read(p)
  except json.JSONDecodeError:return False
  if d.get('complete'):
   assert d['protocol']['manifest_sha256']==digest
   return True
  return False
 try:
  verify()
  for shard in range(3):
   p=REPORT/f'supervisor-{shard}.json'
   if p.exists():failures.update(read(p).get('failures',{}))
  if args.transition:
   transition=dict(started_utc=now(),manifest_sha256=digest,before=[],statistical_changes=False,configuration=dict(batch=manifest['batch_size'],moves=manifest['moves']),reason='Remove static surface shard imbalance; keep GPU2 useful. No scheduling decisions depend on prediction metrics.')
   # Validate every target before stopping any parent.
   for pid in PARENTS:
    p=proc(pid)
    if p is not None and p['state']!='Z':
     if 'scripts.joint_field_identification.surface_supervisor' not in p['command']:raise RuntimeError('Refuse unrelated PID '+str(pid))
     transition['before'].append(p)
   for p in transition['before']:os.kill(p['pid'],signal.SIGSTOP)
   time.sleep(.2);paused=[]
   for original in transition['before']:
    p=proc(original['pid']);assert p is not None and p['state'] in ['T','t']
    children=[]
    for child in p['children']:
     c=proc(child)
     if c is None:continue
     if c['state']!='Z' and 'scripts.joint_field_identification.surface_continuous' not in c['command']:raise RuntimeError('Unexpected child; leave paused for review')
     children.append(c)
    metadata=next(read(REPORT/f'supervisor-{shard}.json') for shard in range(3) if read(REPORT/f'supervisor-{shard}.json')['pid']==p['pid'])
    paused.append(dict(parent=p,children=children,supervisor_state=metadata))
   transition['paused']=paused;write(REPORT/'dynamic-transition.json',transition);status('waiting_for_running_children')
   while any(c is not None and c['state']!='Z' for d in paused for c in [proc(x['pid']) for x in d['children']]):
    if time.time()>=deadline:raise TimeoutError('Deadline while waiting; existing workers retain own deadline guard')
    status('waiting_for_running_children');time.sleep(5)
   transition['boundaries']=[]
   for d in paused:
    for child in d['children']:
     command=child['command'];name=None
     if '--shape' in command:
      get=lambda option:command[command.index(option)+1]
      j=dict(shape=get('--shape'),population=get('--population'),case=int(get('--case')),stage=get('--stage'),particles=int(get('--particles')),seed=int(get('--seed')))
      name=key(j);assert name in names
     if name is None:name=d['supervisor_state'].get('current')
     if name is not None and not completed(name):failures[name]=dict(reason='Pretransition worker ended without complete valid report',pid=child['pid'])
     transition['boundaries'].append(dict(parent_pid=d['parent']['pid'],child_pid=child['pid'],job=name,complete=completed(name) if name else None))
    p=proc(d['parent']['pid'])
    if p is not None:
     assert p['state'] in ['T','t'];os.kill(p['pid'],signal.SIGKILL) # only stopped scheduler, child already exited
   for shard in range(3):
    p=REPORT/f'supervisor-{shard}.json'
    if p.exists():
     doc=read(p);doc.update(status='superseded_by_dynamic_dispatch',superseded_utc=now(),dynamic_dispatch_pid=os.getpid());write(p,doc)
   transition['completed_utc']=now();transition['after_config_manifest_sha256']=sha(manifestpath);write(REPORT/'dynamic-transition.json',transition)
  else:
   for pid in PARENTS:
    p=proc(pid)
    if p and p['state']!='Z':raise RuntimeError('Old surface scheduler still alive; transition required')
  while time.time()<deadline:
   verify()
   for gpu,d in list(running.items()):
    code=d['process'].poll()
    if code is None:continue
    logs[gpu].close();del logs[gpu]
    if code or not completed(d['job']):failures[d['job']]=dict(reason='dynamic_worker_failed_or_incomplete',returncode=code,pid=d['pid'],log=d['log'])
    del running[gpu]
   done=[n for n in names if completed(n)];active={d['job'] for d in running.values()};pending=[i for i,n in enumerate(names) if n not in done and n not in failures and n not in active]
   state.update(completed=done,pending=[names[i] for i in pending],failures=failures)
   for gpu in [0,1,2]:
    if gpu in running or not pending:continue
    i=pending.pop(0);j=jobs[i];name=names[i]
    # All finite records exist already; still validate acquisition record readiness.
    record=ROOT/f"reports/joint-field-surface-v04/{j['shape']}/{j['population']}/case-{j['case']:03d}.json"
    if not record.exists():raise RuntimeError('Finite record absent '+str(record))
    if j['stage']!='initial':assert len([r for r in read(record)['rows'] if r['policy']==j['stage'] and r['step']>0])==manifest['action_budget']
    logfile=RUN/'logs'/f'{name}-dynamic.log';logfile.parent.mkdir(parents=True,exist_ok=True);log=logfile.open('a');log.write('\nDYNAMIC DISPATCH '+now()+'\n');log.flush()
    cmd=[sys.executable,'-m','scripts.joint_field_identification.surface_continuous','--shape',j['shape'],'--population',j['population'],'--case',str(j['case']),'--stage',j['stage'],'--particles',str(j['particles']),'--seed',str(j['seed']),'--cells',*map(str,j['cells']),'--device',f'cuda:{gpu}','--batch',str(manifest['batch_size']),'--moves',str(manifest['moves']),'--manifest',str(manifestpath),'--resume']
    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    logs[gpu]=log;running[gpu]=dict(process=child,pid=child.pid,job=name,index=i,device=f'cuda:{gpu}',started_utc=now(),log=str(logfile.relative_to(ROOT)))
   state['pending']=[names[i] for i in pending];status('running' if running else ('complete' if not failures else 'finished_with_failures'))
   if not running:return
   time.sleep(5)
  for gpu,d in running.items():
   d['process'].terminate()
   try:d['process'].wait(timeout=5)
   except subprocess.TimeoutExpired:d['process'].kill();d['process'].wait()
   failures[d['job']]=dict(reason='deadline',pid=d['pid'])
  status('deadline_reached_incomplete')
 except Exception as e:
  state['exception']=dict(type=type(e).__name__,message=str(e));status('blocked_exception');raise
 finally:
  for log in logs.values():log.close()
  lock.unlink(missing_ok=True)
if __name__=='__main__':main()
