"""Explicit own-process budgets and owned-session cleanup; no GPU allocation."""
from __future__ import annotations
import math,os,signal,subprocess,shutil
from pathlib import Path
MiB=1024**2
GiB=1024**3
LIMITS=dict(gpu_bytes=8*GiB,rss_bytes=12*GiB,disk_bytes=30*GiB,disk_free_reserve_bytes=30*GiB)
class ControllerStop(Exception):pass

def validate_budget(metrics,*,reserve_write_bytes=0,reserve_memory_bytes=0,limits=LIMITS):
 for k in ['gpu_bytes','torch_allocated_bytes','torch_reserved_bytes','rss_bytes','disk_bytes','disk_free_bytes']:
  v=metrics[k]
  if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:raise ValueError('Finite resource measurement required: '+k)
 if reserve_write_bytes<0 or reserve_memory_bytes<0:raise ValueError('Nonnegative allocation reserves required')
 if max(metrics['gpu_bytes'],metrics['torch_allocated_bytes'],metrics['torch_reserved_bytes'])>limits['gpu_bytes']:raise RuntimeError('Own GPU8GiB budget exceeded')
 if metrics['rss_bytes']+reserve_memory_bytes>limits['rss_bytes']:raise RuntimeError('RSS12GiB projected budget exceeded')
 if metrics['disk_bytes']+reserve_write_bytes>limits['disk_bytes']:raise RuntimeError('Own disk30GiB projected usage cap exceeded')
 if metrics['disk_free_bytes']-reserve_write_bytes<limits['disk_free_reserve_bytes']:raise RuntimeError('Filesystem free30GiB reserve would be violated')
 return True

def validate_cpu(cores,actual,environment,intra,interop):
 if len(cores)!=2 or len(set(cores))!=2 or set(cores)!=set(actual):raise RuntimeError('Exactly two pinned permitted CPU cores required')
 if intra!=2 or interop!=2 or any(environment.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise RuntimeError('CPU2 thread contract violated')
 return True

def collect_resources(*,gpu_uuid,pid,roots,torch):
 rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines();own=[r for r in rows if r.split(',')[0].strip()==str(pid)]
 if not own or any(r.split(',')[1].strip()!=gpu_uuid for r in own):raise RuntimeError('Own physicalGPU UUID proof unavailable/wrong')
 try:gpu=sum(float(r.split(',')[2].strip()) for r in own)*MiB+MiB
 except ValueError as e:raise RuntimeError('Own GPU resident memory unavailable') from e
 status=Path('/proc/self/status').read_text().splitlines();rss=int(next(x for x in status if x.startswith('VmRSS:')).split()[1])*1024
 disk=sum(p.stat().st_size for root in roots if root.exists() for p in root.rglob('*') if p.is_file())
 free=min(shutil.disk_usage(root).free for root in roots if root.exists())
 return dict(disk_free_bytes=free,gpu_bytes=gpu,torch_allocated_bytes=torch.cuda.memory_allocated(),torch_reserved_bytes=torch.cuda.memory_reserved(),rss_bytes=rss,disk_bytes=disk,own_GPU_records=own,gpu_rounding_reserve_bytes=MiB)

def terminate_owned(child,grace_s=3):
 """Only a Popen child launched with start_new_session=True is eligible."""
 if child.poll() is not None:return child.wait()
 try:group=os.getpgid(child.pid)
 except ProcessLookupError:return child.wait()
 if group!=child.pid or group==os.getpgrp():raise RuntimeError('Refusing non-owned/nonisolated process-group cleanup')
 try:os.killpg(group,signal.SIGTERM)
 except ProcessLookupError:pass
 try:return child.wait(timeout=grace_s)
 except subprocess.TimeoutExpired:
  try:os.killpg(group,signal.SIGKILL)
  except ProcessLookupError:pass
  return child.wait()

def run_owned_command(command,*,environment,log,timeout,on_child=None):
 """Catch controller TERM/INT before spawn; always drain/reap its own group."""
 child=None;result=dict(timeout=False,stop_signal=None,exit_code=None);previous={}
 def stop(signum,frame):raise ControllerStop(signum)
 for sig in (signal.SIGTERM,signal.SIGINT):previous[sig]=signal.signal(sig,stop)
 try:
  child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
  result['worker_pid']=child.pid
  if on_child:on_child(child.pid)
  result['exit_code']=child.wait(timeout=timeout)
 except (subprocess.TimeoutExpired,ControllerStop) as error:
  for sig in previous:signal.signal(sig,signal.SIG_IGN)
  if isinstance(error,ControllerStop):result['stop_signal']=int(error.args[0])
  else:result['timeout']=True
  if child is not None:result['exit_code']=terminate_owned(child)
 finally:
  try:
   if child is not None and child.poll() is None:terminate_owned(child)
  finally:
   for sig,old in previous.items():signal.signal(sig,old)
 return result
