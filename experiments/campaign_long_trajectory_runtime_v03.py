"""Durable accepted-state transactions and owned fixed-pipeline supervision.

No CUDA/FE imports. The controller never resumes or retries a failed worker.
"""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
import numpy as np
from experiments.campaign_owned_gpu_budget_v02 import terminate_owned
GPUS=('GPU-5b432ce5-a24a-68ba-150f-682265260a9b','GPU-c49707f6-24fd-38e4-3973-3543ec644ceb','GPU-571378aa-c44e-ffb8-b346-1e409f9821f9')
JOBS=(dict(id='fine40_h32',cells=[40,20,20],steps=32,slot=0),dict(id='fine40_h64',cells=[40,20,20],steps=64,slot=1),dict(id='fine40_h128',cells=[40,20,20],steps=128,slot=2),dict(id='coarse32_h128',cells=[32,16,16],steps=128,slot=0))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path):
 p=Path(path);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)

def atomic_json(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
 fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

def append_ledger(path,row):
 with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def validate_lease(job,gpu,cores):
 if job not in JOBS or gpu!=GPUS[job['slot']] or cores!=[2*job['slot'],2*job['slot']+1]:raise ValueError('Exact frozen job/GPU0..2/CPU lease required; GPU3 forbidden')
 return True

def verify_pins(pins):
 if not pins or any(not Path(p).is_file() or sha(p)!=digest for p,digest in pins.items()):raise ValueError('Frozen pin mismatch')
 return True

def checkpoint(root,u,v,step,time_s,protocol_sha,job_id,accepted_ledger=None):
 """Commit immutable generation, then pointer, then prune to current+previous.

 Transient third generation allows old current+previous to survive interrupted
 writes. Only fully accepted states may call this. No automatic resume action.
 """
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 if type(step)is not int or step<0 or not np.isfinite(time_s) or time_s<0 or u.shape!=v.shape or not np.isfinite(u).all() or not np.isfinite(v).all():raise ValueError('Finite accepted q/v/time required')
 if (root/'COMMIT.json').exists():
  previous=load_checkpoint(root,protocol_sha,job_id)[2]
  if step!=previous['step']+1 or time_s<=previous['time_s']:raise ValueError('Strict accepted-state generation order required')
 elif step!=0:raise ValueError('Initial accepted step0 required')
 directory=root/f'accepted-{step:05d}';directory.mkdir(exist_ok=False)
 with (directory/'state.npz').open('xb') as f:np.savez_compressed(f,u=u,v=v);f.flush();os.fsync(f.fileno())
 metadata=dict(step=step,time_s=float(time_s),protocol_sha256=protocol_sha,job_id=job_id,state=record(directory/'state.npz'),accepted_ledger=accepted_ledger)
 atomic_json(directory/'metadata.json',metadata)
 atomic_json(root/'COMMIT.json',dict(current=record(directory/'metadata.json'),previous=None if step==0 else record(root/f'accepted-{step-1:05d}'/'metadata.json')))
 # Commit readback before removing history; a corrupt state never prunes history.
 load_checkpoint(root,protocol_sha,job_id)
 import shutil
 for old in root.glob('accepted-*'):
  if old.is_dir() and old.name not in {f'accepted-{step:05d}',f'accepted-{step-1:05d}'}:shutil.rmtree(old)
 return metadata

def load_checkpoint(root,protocol_sha,job_id):
 root=Path(root);pointer=json.loads((root/'COMMIT.json').read_text());row=pointer['current']
 if sha(row['path'])!=row['sha256']:raise ValueError('Checkpoint metadata hash mismatch')
 m=json.loads(Path(row['path']).read_text())
 if m['protocol_sha256']!=protocol_sha or m['job_id']!=job_id or sha(m['state']['path'])!=m['state']['sha256']:raise ValueError('Checkpoint source/job/data hash mismatch')
 if pointer['previous']:
  old=pointer['previous']
  if sha(old['path'])!=old['sha256']:raise ValueError('Previous checkpoint metadata hash mismatch')
  pm=json.loads(Path(old['path']).read_text())
  if pm['step']!=m['step']-1 or pm['protocol_sha256']!=protocol_sha or pm['job_id']!=job_id or sha(pm['state']['path'])!=pm['state']['sha256']:raise ValueError('Previous accepted-state hash/lineage mismatch')
 with np.load(m['state']['path'],allow_pickle=False) as f:u,v=f['u'].copy(),f['v'].copy()
 return u,v,m

def supervise(jobs,spawn,terminal,*,global_cap_s,per_job_cap_s,receipt_path,poll_s=.2):
 """First three start together; fourth waits for job0 terminal PASS AND reap.

 Any numerical/technical/resource error cancels queued launch and terminates
 every still-owned peer. Always logs each launched PID and reaped exit code.
 """
 start=time.monotonic();active={};done=[];queued=list(jobs);failed=False;failure=None;previous={}
 def save():atomic_json(receipt_path,dict(terminal=not active and not queued,elapsed_s=time.monotonic()-start,active=[dict(id=k,pid=v[0].pid) for k,v in active.items()],completed=done,queued=[j['id'] for j in queued],failed=failed,failure=failure))
 def stopped(signum,frame):raise InterruptedError('Controller signal '+str(signum))
 for sig in (signal.SIGTERM,signal.SIGINT):previous[sig]=signal.signal(sig,stopped)
 try:
  while queued or active:
   if time.monotonic()-start>global_cap_s:raise TimeoutError('Global wall cap')
   for key,(child,began,j) in list(active.items()):
    if time.monotonic()-began>per_job_cap_s:raise TimeoutError('Per-trajectory cap: '+key)
    code=child.poll()
    if code is None:continue
    child.wait();passed=code==0 and terminal(j)
    done.append(dict(id=key,pid=child.pid,exit_code=code,passed=bool(passed)));del active[key];save()
    if not passed:raise RuntimeError('Worker numerical/resource/technical failure: '+key)
   occupied={j['slot'] for _,_,j in active.values()}
   for j in queued.copy():
    if j['slot'] in occupied:continue
    if j is not jobs[0] and j['slot']==0 and not any(r['id']==jobs[0]['id'] and r['passed'] for r in done):continue
    child=spawn(j)
    if child.pid!=os.getpgid(child.pid):raise RuntimeError('Worker must own isolated process group')
    active[j['id']]=(child,time.monotonic(),j);occupied.add(j['slot']);queued.remove(j);save()
   if active:time.sleep(poll_s)
 except BaseException as e:
  failed=True;failure=dict(type=type(e).__name__,message=str(e));queued.clear()
 finally:
  for sig in previous:signal.signal(sig,signal.SIG_IGN)
  for key,(child,_,j) in list(active.items()):
   code=terminate_owned(child,grace_s=3);done.append(dict(id=key,pid=child.pid,exit_code=code,passed=False,cancelled=True));del active[key]
  save()
  for sig,handler in previous.items():signal.signal(sig,handler)
 return not failed


def watchdog(pid,deadline,step_deadline_path,rss_limit,receipt,poll_s=.25,kill_grace_s=2):
 """Independent guard catches native-blocking steps; only its original parent PID.

 The inherited mask is cleared in the watcher as well as in the worker.
 """
 signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGALRM})
 signal.signal(signal.SIGTERM,signal.SIG_DFL);signal.signal(signal.SIGINT,signal.SIG_DFL)
 stat=Path('/proc')/str(pid);identity=(stat/'stat').read_text().split()[21];peak=0
 while stat.exists():
  try:
   if (stat/'stat').read_text().split()[21]!=identity:return
   rss=int((stat/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
  except FileNotFoundError:return
  peak=max(peak,rss);now=time.monotonic()
  try:step_deadline=float(json.loads(Path(step_deadline_path).read_text())['deadline_monotonic_s'])
  except (FileNotFoundError,ValueError,KeyError):step_deadline=deadline
  reason='RSS_LIMIT' if rss>rss_limit else ('GLOBAL_WORKER_WALL_LIMIT' if now>deadline else ('STEP_WALL_LIMIT' if now>step_deadline else None))
  atomic_json(receipt,dict(observed_peak_RSS_bytes=peak,reason=reason,parent_pid=pid))
  if reason:
   try:os.kill(pid,signal.SIGTERM)
   except ProcessLookupError:return
   time.sleep(kill_grace_s)
   try:
    if (stat/'stat').read_text().split()[21]==identity:os.kill(pid,signal.SIGKILL)
   except FileNotFoundError:pass
   return
  time.sleep(poll_s)
