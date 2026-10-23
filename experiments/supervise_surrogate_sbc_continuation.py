"""Independent bounded supervisor for evaluation-only SBC continuation."""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,sys,time
from experiments.campaign_owned_proc import proc_identity,proc_rss
ROOT=Path(__file__).resolve().parents[1]


def supervise(command,log_path,receipt_path,*,journal_path=None,wall_s=1800,RSS_bytes=2*1024**3):
 signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGALRM})
 start=time.monotonic();reason=None;peak=0;owned={};proc=None;result={}
 def stop(*args):raise InterruptedError('Independent owned supervisor interrupted')
 def same_process(pid,stamp):
  info=proc_identity(pid);return bool(info and info[1]==stamp)
 previous={s:signal.signal(s,stop)for s in (signal.SIGTERM,signal.SIGINT)}
 try:
  with Path(log_path).open('x')as log:
   proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   print(json.dumps(dict(supervisor_pid=os.getpid(),controller_pid=proc.pid,hard_wall_s=wall_s)),flush=True)
   while proc.poll()is None:
    peak=max(peak,proc_rss(proc.pid))
    if journal_path is not None and Path(journal_path).exists():
     j=json.loads(Path(journal_path).read_text())
     if j.get('pid')!=proc.pid:raise ValueError('Owned journal controller identity')
     for r in j.get('live',[]):
      info=proc_identity(r['pid'])
      if info and info[0]==proc.pid:owned[r['pid']]=info[1]
    if time.monotonic()-start>wall_s:raise TimeoutError('Independent hard wall')
    if peak>RSS_bytes:raise MemoryError('Independent controller RSS cap')
    time.sleep(.1)
   proc.wait()
 except BaseException as e:reason=repr(e)
 finally:
  for s in previous:signal.signal(s,signal.SIG_IGN)
  try:
   if proc is not None:
    if proc.poll()is None:
     try:os.killpg(proc.pid,signal.SIGTERM)
     except ProcessLookupError:pass
     try:proc.wait(timeout=3)
     except subprocess.TimeoutExpired:
      try:os.killpg(proc.pid,signal.SIGKILL)
      except ProcessLookupError:pass
      proc.wait(timeout=3)
    code=proc.wait()
    for pid,stamp in owned.items():
     if same_process(pid,stamp):
      try:os.killpg(pid,signal.SIGTERM)
      except ProcessLookupError:pass
    end=time.monotonic()+3
    while any(same_process(pid,stamp)for pid,stamp in owned.items())and time.monotonic()<end:time.sleep(.05)
    for pid,stamp in owned.items():
     if same_process(pid,stamp):
      try:os.killpg(pid,signal.SIGKILL)
      except ProcessLookupError:pass
    live=[pid for pid,stamp in owned.items()if same_process(pid,stamp)]
    result=dict(completed=code==0 and reason is None and not live,exit_code=code,reason=reason,supervisor_pid=os.getpid(),controller_pid=proc.pid,controller_reaped=True,controller_PID_absent=proc_identity(proc.pid)is None,observed_owned_worker_PIDs=sorted(owned),all_observed_owned_worker_PIDs_absent=not live,wall_s=time.monotonic()-start,peak_controller_RSS_bytes=peak,hard_wall_s=wall_s,RSS_limit_bytes=RSS_bytes)
    with Path(receipt_path).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
  finally:
   for s,h in previous.items():signal.signal(s,h)
 return result


def main(protocol,digest):
 path=ROOT/protocol
 if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen continuation protocol hash')
 p=json.loads(path.read_text());op=path.parent
 if p.get('new_generation_or_inference')is not False:raise ValueError('Evaluation-only continuation')
 command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','taskset','-c','10,11',str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_campaign_surrogate_sbc_evaluation_continuation','run','--protocol',protocol,'--digest',digest]
 r=supervise(command,op/'actual-continuation.log',op/'actual-supervisor-receipt.json',journal_path=ROOT/p['output']/'journal.json',wall_s=1800,RSS_bytes=2*1024**3);print(json.dumps(r));return 0 if r['completed']else 1


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--protocol',required=True);a.add_argument('--digest',required=True);v=a.parse_args();sys.exit(main(v.protocol,v.digest))
