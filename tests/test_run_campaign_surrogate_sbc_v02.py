"""Focused startup regression only; no scientific models/data/random draws."""
import ast,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from experiments import run_campaign_surrogate_sbc as old
from experiments import run_campaign_surrogate_sbc_v02 as m


def test_frozen_design_science_and_v01_snapshot_unchanged():
 p=Path('runs/operator-dev/matching-prior-surrogate-sbc-v01/protocol.json');frozen=json.loads(p.read_text())
 assert m.DESIGN==old.DESIGN==frozen['design'] and m.RESOURCES==old.RESOURCES==frozen['resources']
 assert m.DESIGN['seeds']==frozen['design']['seeds']
 for n in ('experiments/run_campaign_surrogate_sbc.py','experiments/campaign_surrogate_sbc.py','experiments/campaign_resolved_quantiles_v04.py','experiments/campaign_cdf_arithmetic.py'):
  assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==frozen['source_sha256'][n]
 # Every scientific lifecycle function except added entry restoration is AST-identical.
 def functions(path):return {n.name:n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}
 a=functions(old.__file__);b=functions(m.__file__)
 for name in ('generate','worker'):
  before=a[name];after=b[name]
  # restore_entry_signals is the only new statement in these scientific paths.
  after.body=[n for n in after.body if not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='restore_entry_signals')]
  assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
 assert 'cwd=ROOT'in Path(m.__file__).read_text()


def test_real_exec_CLI_restores_all_inherited_masks_before_owned_work(tmp_path):
 out=tmp_path/'mask.json'
 code="""import signal,runpy,atexit,json,sys
from pathlib import Path
out=Path(sys.argv[1]);signals={signal.SIGTERM,signal.SIGINT,signal.SIGALRM};before=signal.pthread_sigmask(signal.SIG_BLOCK,set());assert signals<=before
atexit.register(lambda:out.write_text(json.dumps({'before':sorted(int(s)for s in before),'after':sorted(int(s)for s in signal.pthread_sigmask(signal.SIG_BLOCK,set()))})))
sys.argv=['surrogate-sbc-v02','--help'];runpy.run_module('experiments.run_campaign_surrogate_sbc_v02',run_name='__main__')
"""
 def blocked():signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGALRM})
 r=subprocess.run([sys.executable,'-c',code,str(out)],cwd=m.ROOT,preexec_fn=blocked,capture_output=True,timeout=15);assert r.returncode==0,r.stderr
 value=json.loads(out.read_text());assert not({int(signal.SIGTERM),int(signal.SIGINT),int(signal.SIGALRM)}&set(value['after']))


def test_real_exec_worker_entry_unblocks_and_TERM_exits_gracefully_reaped(tmp_path):
 out=tmp_path/'owned';out.mkdir();receipt=tmp_path/'worker-mask.json';sender_pid=None
 code="""import signal,json,sys,os,time
from pathlib import Path
before=signal.pthread_sigmask(signal.SIG_BLOCK,set());assert signal.SIGTERM in before and signal.SIGINT in before
from experiments import run_campaign_surrogate_sbc_v02 as m
path=Path(sys.argv[1])
def fakevalidate(*a,**k):
 after=signal.pthread_sigmask(signal.SIG_BLOCK,set());assert not({signal.SIGTERM,signal.SIGINT,signal.SIGALRM}&after)
 def stop(*a):
  value=json.loads(path.read_text());value['graceful_TERM']=True;path.write_text(json.dumps(value));raise SystemExit(0)
 signal.signal(signal.SIGTERM,stop);path.write_text(json.dumps({'pid':os.getpid(),'before':sorted(int(s)for s in before),'after':sorted(int(s)for s in after),'cwd':os.getcwd()}))
 while True:time.sleep(.02)
m.validate=fakevalidate;m.worker('NO_ACTUAL_PROTOCOL','NO_ACTUAL_DIGEST','inference',0,0)
"""
 # A sender is test-owned, separate from the controller's scientific workers.
 sender_code="""import json,sys,time,signal,os
from pathlib import Path
p=Path(sys.argv[1]);end=time.monotonic()+12
while time.monotonic()<end:
 if p.exists():os.kill(json.loads(p.read_text())['pid'],signal.SIGTERM);raise SystemExit(0)
 time.sleep(.02)
raise RuntimeError('No restored worker appeared')
"""
 sender=subprocess.Popen([sys.executable,'-c',sender_code,str(receipt)],start_new_session=True)
 try:
  records=m.owned_jobs(lambda i,s:[sys.executable,'-c',code,str(receipt)],[0],1,out,worker_wall_s=12,global_wall_s=15,RSS_bytes=2*1024**3)
  sender.wait(timeout=3);assert sender.returncode==0
 finally:
  if sender.poll()is None:sender.terminate();sender.wait(timeout=3)
 value=json.loads(receipt.read_text());assert value['graceful_TERM']and value['cwd']==str(m.ROOT)
 assert records[0]['exit_code']==0 and records[0]['reaped']and not Path('/proc',str(records[0]['pid'])).exists()
 journal=json.loads((out/'journal.json').read_text());assert journal['completed']and journal['all_owned_workers_reaped']and not journal['live']
