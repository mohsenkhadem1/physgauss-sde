import os,signal,subprocess,sys,time
import pytest
from experiments.campaign_owned_gpu_budget_v02 import LIMITS,GiB,validate_budget,validate_cpu,terminate_owned

def test_projected_resource_caps_without_gpu():
 m=dict(gpu_bytes=1,torch_allocated_bytes=1,torch_reserved_bytes=1,rss_bytes=1,disk_bytes=1,disk_free_bytes=100*GiB)
 assert validate_budget(m)
 for key,limit in [('gpu_bytes',8*GiB),('torch_allocated_bytes',8*GiB),('torch_reserved_bytes',8*GiB),('rss_bytes',12*GiB),('disk_bytes',30*GiB)]:
  with pytest.raises(RuntimeError):validate_budget({**m,key:limit+1})
 with pytest.raises(RuntimeError):validate_budget(m,reserve_write_bytes=30*GiB)
 with pytest.raises(RuntimeError):validate_budget(m,reserve_memory_bytes=12*GiB)
 with pytest.raises(ValueError):validate_budget({**m,'gpu_bytes':float('nan')})

def test_cpu_thread_and_affinity_contract():
 env={k:'2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}
 assert validate_cpu([2,3],{2,3},env,2,2)
 with pytest.raises(RuntimeError):validate_cpu([2,3],{2,3,4},env,2,2)
 with pytest.raises(RuntimeError):validate_cpu([2,3],{2,3},{**env,'MKL_NUM_THREADS':'8'},2,2)
 with pytest.raises(RuntimeError):validate_cpu([2,3],{2,3},env,4,2)

def test_owned_cleanup_drains_and_reaps_term_then_kill(tmp_path):
 for ignore in [False,True]:
  ready=tmp_path/str(ignore)
  code='import signal,time;from pathlib import Path;'+('signal.signal(signal.SIGTERM,signal.SIG_IGN);' if ignore else '')+f'Path({str(ready)!r}).write_text("ready");time.sleep(60)'
  p=subprocess.Popen([sys.executable,'-c',code],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  try:
   deadline=time.monotonic()+3
   while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
   assert ready.exists();ret=terminate_owned(p,grace_s=.1)
   assert p.poll() is not None and ret==(-signal.SIGKILL if ignore else -signal.SIGTERM)
  finally:
   if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()

def test_controller_SIGTERM_reaps_owned_worker(tmp_path):
 import json
 ready=tmp_path/'childpid';ledger=tmp_path/'ledger.json';log=tmp_path/'child.log'
 code='''import os,sys,json
from pathlib import Path
from experiments.campaign_owned_gpu_budget_v02 import run_owned_command
ready,ledger,log=map(Path,sys.argv[1:])
childcode="import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"
with log.open('w') as f:
 r=run_owned_command([sys.executable,'-c',childcode],environment=os.environ.copy(),log=f,timeout=60,on_child=lambda pid:ready.write_text(str(pid)))
ledger.write_text(json.dumps(r))
'''
 p=subprocess.Popen([sys.executable,'-c',code,str(ready),str(ledger),str(log)],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 try:
  deadline=time.monotonic()+5
  while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
  assert ready.exists();childpid=int(ready.read_text());time.sleep(.1);p.send_signal(signal.SIGTERM);assert p.wait(timeout=6)==0
  row=json.loads(ledger.read_text());assert row['stop_signal']==signal.SIGTERM
  with pytest.raises(ProcessLookupError):os.kill(childpid,0)
 finally:
  if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()


def test_free_filesystem_reserve_even_when_own_files_are_small():
 m=dict(gpu_bytes=1,torch_allocated_bytes=1,torch_reserved_bytes=1,rss_bytes=1,disk_bytes=1,disk_free_bytes=29*GiB)
 with pytest.raises(RuntimeError,match='free30GiB'):validate_budget(m)
 with pytest.raises(RuntimeError,match='free30GiB'):validate_budget({**m,'disk_free_bytes':30*GiB},reserve_write_bytes=1)
 assert validate_budget({**m,'disk_free_bytes':30*GiB+1000},reserve_write_bytes=1000)

def test_collector_reads_minimum_filesystem_free_without_gpu(monkeypatch,tmp_path):
 from types import SimpleNamespace
 import experiments.campaign_owned_gpu_budget_v02 as module
 a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir();(a/'small').write_bytes(b'123')
 monkeypatch.setattr(module.subprocess,'check_output',lambda *args,**kw:f'{os.getpid()}, GPU-test, 12\n')
 monkeypatch.setattr(module.shutil,'disk_usage',lambda root:SimpleNamespace(free=(35 if root==a else 40)*GiB))
 fake=SimpleNamespace(cuda=SimpleNamespace(memory_allocated=lambda:10,memory_reserved=lambda:20))
 row=module.collect_resources(gpu_uuid='GPU-test',pid=os.getpid(),roots=[a,b],torch=fake)
 assert row['disk_free_bytes']==35*GiB and row['disk_bytes']==3
