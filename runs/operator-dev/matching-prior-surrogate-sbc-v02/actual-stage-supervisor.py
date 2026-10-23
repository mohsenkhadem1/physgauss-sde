"""Independent bounded one-shot owner. No scientific values are changed/read."""
import os,json,signal,subprocess,sys,time,hashlib,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OP=Path(__file__).resolve().parent
stage=sys.argv[1]
assert stage in ('generate','inference','evaluation')
protocol='runs/operator-dev/matching-prior-surrogate-sbc-v02/protocol.json';digest='8b641efeda802561471f55f0a4311036a5290ed37030ef610c2c2dd18814aee6'
assert hashlib.sha256((ROOT/protocol).read_bytes()).hexdigest()==digest
receipt=OP/('actual-'+stage+'-receipt.json');assert not receipt.exists()
limit=60 if stage=='generate'else 1800
journal=ROOT/'runs/gaussian-campaign-v01/matching-prior-surrogate-sbc-v02'/stage/'journal.json'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
cmd=['taskset','-c','10,11',str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_campaign_surrogate_sbc_v02',stage,'--protocol',protocol,'--digest',digest]
start=time.monotonic();reason=None;peak=0;owned={};proc=None;code=None

def procinfo(pid):
 try:
  a=Path('/proc',str(pid),'stat').read_text();fields=a[a.rfind(')')+2:].split();return int(fields[1]),fields[19]
 except FileNotFoundError:return None

def stop(*args):raise InterruptedError('Independent owned supervisor interrupted')
old={s:signal.signal(s,stop)for s in (signal.SIGTERM,signal.SIGINT)}
try:
 with (OP/('actual-'+stage+'.log')).open('x')as log:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  print(json.dumps(dict(stage=stage,supervisor_pid=os.getpid(),controller_pid=proc.pid,hard_wall_s=limit)),flush=True)
  while proc.poll() is None:
   elapsed=time.monotonic()-start
   try:rss=next(int(x.split()[1])*1024 for x in Path('/proc',str(proc.pid),'status').read_text().splitlines()if x.startswith('VmRSS:'));peak=max(peak,rss)
   except(FileNotFoundError,StopIteration):rss=0
   if journal.exists():
    try:
     j=json.loads(journal.read_text())
     if j.get('pid')!=proc.pid:raise ValueError('Owned journal controller identity')
     for row in j.get('live',[]):
      pid=row['pid'];info=procinfo(pid)
      if info and info[0]==proc.pid:owned[pid]=info[1]
    except json.JSONDecodeError:pass
   if elapsed>limit:reason='hard wall';break
   if rss>2*1024**3:reason='controller2GiB RSS';break
   time.sleep(.1)
  if reason:raise RuntimeError(reason)
  code=proc.wait()
except BaseException as e:
 reason=reason or repr(e)
finally:
 for s in old:signal.signal(s,signal.SIG_IGN)
 if proc is not None:
  if proc.poll()is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=3)
  code=proc.wait()
  for pid,stamp in owned.items():
   info=procinfo(pid)
   if info and info[1]==stamp:
    try:os.killpg(pid,signal.SIGTERM)
    except ProcessLookupError:pass
  end=time.monotonic()+3
  while any(procinfo(pid)and procinfo(pid)[1]==stamp for pid,stamp in owned.items())and time.monotonic()<end:time.sleep(.05)
  for pid,stamp in owned.items():
   info=procinfo(pid)
   if info and info[1]==stamp:
    try:os.killpg(pid,signal.SIGKILL)
    except ProcessLookupError:pass
  live=[pid for pid,stamp in owned.items()if procinfo(pid)and procinfo(pid)[1]==stamp]
  result=dict(completed=code==0 and reason is None and not live,stage=stage,exit_code=code,reason=reason,supervisor_pid=os.getpid(),controller_pid=proc.pid,controller_reaped=True,controller_PID_absent=not Path('/proc',str(proc.pid)).exists(),owned_worker_PIDs_observed=sorted(owned),all_observed_owned_worker_PIDs_absent=not live,wall_s=time.monotonic()-start,peak_controller_RSS_bytes=peak,hard_wall_s=limit,RSS_limit_bytes=2*1024**3,protocol_sha256=digest,no_GPU=True,cores=[10,11])
  with receipt.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
  print(json.dumps(result),flush=True)
 for s,h in old.items():signal.signal(s,h)
sys.exit(0 if result['completed']else 1)
