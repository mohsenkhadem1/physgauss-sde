"""Hard600s owned-process launcher for the independent controls/static stages."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from experiments.campaign_newton_vbd_static_v02 import ROOT,ENV,PROTOCOL_ROOT,OUTPUT_ROOT,sha,write

def run(path,digest):
 p=ROOT/path
 if sha(p)!=digest:raise ValueError('Changedstageprotocol')
 d=json.loads(p.read_text());mode=d['mode'];folder=ROOT/PROTOCOL_ROOT/mode;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'execution.json').exists():raise FileExistsError('Preserveexecution')
 if mode=='static':
  controls=ROOT/'runs/gaussian-campaign-v01/newton-vbd-static-v01/controls/summary.json'
  s=json.loads(controls.read_text())
  if s.get('completed') is not True or s.get('passed') is not True:raise RuntimeError('IndependentAPI/freefallcontrolsincomplete/failed')
 command=[str(ENV/'bin/python'),'-u','-m','experiments.campaign_newton_vbd_static_v02','run','--protocol',path,'--protocol-sha256',digest]
 before=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout;gpu0=before.splitlines()[0].split(',')[1].strip();apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
 if any(r.split(',')[1].strip()==gpu0 for r in apps.splitlines() if ',' in r):raise RuntimeError('GPU0notidle')
 ledger=dict(mode=mode,protocol=dict(path=path,sha256=digest),controller_source_sha256=sha(__file__),controller_pid=os.getpid(),hard_cap_s=600,terminal=False,completed=False,command=command,prelaunch_gpu=before)
 if mode=='static':ledger['controls_summary_sha256']=sha(controls)
 write(folder/'execution.json',ledger);start=time.monotonic()
 with (folder/'worker.log').open('x') as log:
  child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MAX_JOBS':'2'})
  ledger['worker_pid']=child.pid;write(folder/'execution.json',ledger)
  try:code=child.wait(timeout=600)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=child.returncode;ledger['timeout']=True
 result=ROOT/OUTPUT_ROOT/mode/'summary.json';s=json.loads(result.read_text()) if result.exists() else {};ledger.update(terminal=True,completed=code==0 and s.get('completed') is True,passed=s.get('passed',False),exit_code=code,elapsed_s=time.monotonic()-start)
 if result.exists():ledger['summary_sha256']=sha(result)
 write(folder/'execution.json',ledger);return ledger
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True);a=p.parse_args();r=run(a.protocol,a.protocol_sha256);print(json.dumps(r));raise SystemExit(0 if r['completed'] else 1)
