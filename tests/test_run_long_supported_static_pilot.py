import builtins,inspect,symtable,json,os,signal,subprocess
import pytest
from experiments import run_long_supported_static_pilot as m


def test_worker_global_names_and_new_domain_case_geometry():
 table=symtable.symtable(inspect.getsource(m.worker),'long-worker','exec')
 def check(t):
  for name in t.get_symbols():
   if name.is_referenced() and name.is_global():assert name.get_name() in vars(m) or hasattr(builtins,name.get_name()),name.get_name()
  for child in t.get_children():check(child)
 check(table)
 assert len(m.cases())==2 and all(c['size_m']==[.48,.08,.08] and c['young_Pa']==300000. for c in m.cases())
 with pytest.raises(ValueError):m.build_model((4,2,2),{**m.cases()[0],'world_surface_force_N':[0,0,1]},rule='keast24')


def test_actual_exec_child_unblocks_inherited_mask_and_terminates_owned_watchdog():
 old=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGALRM});child=None
 try:child=subprocess.Popen([str(m.ROOT/'.venv/bin/python'),'-m','experiments.run_long_supported_static_pilot','signal-control'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
 finally:signal.pthread_sigmask(signal.SIG_SETMASK,old)
 try:
  out,err=child.communicate(timeout=10);assert child.returncode==0,err.decode();r=json.loads(out);assert set([2,14,15]).issubset(r['before']) and not set([2,14,15])&set(r['after']);assert r['watchdog_exit_code']==-15 and r['watchdog_terminated']
 finally:
  if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
