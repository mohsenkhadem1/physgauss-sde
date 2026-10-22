"""Owned-process bounded launcher for the frozen standalone Newton probe."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from experiments.campaign_newton_vbd_probe import ROOT,ENV,sha,write,PROTOCOL

def run(path,digest):
 d=json.loads((ROOT/path).read_text())
 if sha(ROOT/path)!=digest or d['hard_cap_s']!=600:raise ValueError('Changed innerprobe/deadline')
 folder=ROOT/'runs/operator-dev/newton-vbd-probe-v01'
 if (folder/'execution.json').exists():raise FileExistsError('Preserve execution')
 before=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
 gpu0=before.splitlines()[0].split(',')[1].strip();apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
 if any(r.split(',')[1].strip()==gpu0 for r in apps.splitlines() if ',' in r):raise RuntimeError('GPU0 notidle')
 command=[str(ENV/'bin/python'),'-u','-m','experiments.campaign_newton_vbd_probe','run','--protocol',path,'--protocol-sha256',digest]
 ledger=dict(protocol=dict(path=path,sha256=digest),controller_pid=os.getpid(),controller_source_sha256=sha(__file__),command=command,terminal=False,completed=False,prelaunch_gpu=before,hard_cap_s=600)
 write(folder/'execution.json',ledger);start=time.monotonic()
 with (folder/'worker.log').open('x') as log:
  child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MAX_JOBS':'2','OMNI_KIT_ACCEPT_EULA':'YES'})
  ledger['worker_pid']=child.pid;write(folder/'execution.json',ledger)
  try:code=child.wait(timeout=600)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=child.returncode;ledger['timeout']=True
 p=ROOT/'runs/gaussian-campaign-v01/newton-vbd-probe-v01/summary.json';s=json.loads(p.read_text()) if p.exists() else {}
 ledger.update(terminal=True,completed=code==0 and s.get('completed') is True and s.get('passed') is True,exit_code=code,elapsed_s=time.monotonic()-start)
 if p.exists():ledger['summary_sha256']=sha(p)
 write(folder/'execution.json',ledger);return ledger
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',default=PROTOCOL);p.add_argument('--protocol-sha256',required=True);a=p.parse_args();r=run(a.protocol,a.protocol_sha256);print(json.dumps(r));raise SystemExit(0 if r['completed'] else 1)
