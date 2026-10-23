import os,sys,time,json,subprocess,signal,hashlib
from pathlib import Path
root=Path('/home/bayespc2/vscode/guassianSDE');op=root/'runs/operator-dev/resolved-private-numerical-evaluation-v01'
receipt=op/'actual-execution-receipt.json'
assert not receipt.exists()
assert not (root/'runs/gaussian-campaign-v01/resolved-private-numerical-evaluation-v01').exists()
protocol=op/'protocol.json';digest=hashlib.sha256(protocol.read_bytes()).hexdigest()
assert digest=='40f6dfe7ba43b990843abc7d6d46bd7f8f4ea08bb32e63ee815d9399cbf3e068'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
cmd=['taskset','-c','34,35',str(root/'.venv/bin/python'),'-m','experiments.run_campaign_resolved_numerical_evaluation','run','--protocol',str(protocol.relative_to(root)),'--protocol-sha256',digest]
start=time.monotonic();reason=None;peak=0
with (op/'actual-execution.log').open('x') as log:
 p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  while p.poll() is None:
   elapsed=time.monotonic()-start
   try:
    rows=Path('/proc',str(p.pid),'status').read_text().splitlines();rss=next(int(x.split()[1])*1024 for x in rows if x.startswith('VmRSS:'));peak=max(peak,rss)
   except (FileNotFoundError,StopIteration):rss=0
   if elapsed>180:reason='180s hard wall'
   elif rss>2*1024**3:reason='2GiB RSS'
   if reason:break
   time.sleep(.2)
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=3)
  code=p.wait()
r=dict(completed=code==0,exit_code=code,supervisor_pid=os.getpid(),worker_pid=p.pid,worker_reaped=True,worker_PID_absent=not Path('/proc',str(p.pid)).exists(),wall_s=time.monotonic()-start,peak_RSS_bytes=peak,reason=reason,cores=[34,35],threads=2,no_GPU=True,hard_wall_s=180,RSS_limit_bytes=2*1024**3,protocol_sha256=digest)
with receipt.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
sys.exit(code)
