"""Owned GPU worker with independent 600-second deadline and durable terminal receipt."""
import argparse,json,os,signal,subprocess,time
from experiments.campaign_newton_contact_controls import ROOT,ENV,OP,OUT,sha,write,GPU


def run(path,digest):
    if sha(ROOT/path)!=digest:raise ValueError('Changed protocol')
    if (OP/'execution.json').exists():raise FileExistsError('Preserve execution')
    d=json.loads((ROOT/path).read_text())
    if d['wall_cap_s']!=600:raise ValueError('Changed deadline')
    gpus=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
    gpu0=next(r.split(',')[1].strip() for r in gpus.splitlines() if r.split(',')[0].strip()=='1')
    if gpu0!=GPU:raise ValueError('PhysicalGPU1 UUID changed')
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
    if any(r.split(',')[1].strip()==gpu0 for r in apps.splitlines() if ',' in r):raise RuntimeError('GPU1 not idle')
    command=[str(ENV/'bin/python'),'-u','-m','experiments.campaign_newton_contact_controls','run','--protocol',path,'--protocol-sha256',digest]
    receipt=dict(protocol=dict(path=path,sha256=digest),controller_pid=os.getpid(),terminal=False,completed=False,wall_cap_s=600,command=command)
    begin=time.monotonic();write(OP/'execution.json',receipt)
    with (OP/'worker.log').open('x') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MAX_JOBS':'2'})
        receipt['worker_pid']=child.pid;write(OP/'execution.json',receipt)
        try:code=child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=child.returncode;receipt['timeout']=True
    summary=OUT/'summary.json';s=json.loads(summary.read_text()) if summary.exists() else {}
    receipt.update(terminal=True,completed=code==0 and s.get('completed') is True,passed=s.get('passed',False),exit_code=code,elapsed_s=time.monotonic()-begin)
    if summary.exists():receipt['summary_sha256']=sha(summary)
    write(OP/'execution.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True);a=p.parse_args()
    r=run(a.protocol,a.protocol_sha256);print(json.dumps(r));raise SystemExit(0 if r['completed'] else 1)
