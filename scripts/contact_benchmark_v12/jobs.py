"""Persistent locked queue; native GPU and independent prefix-fit CPU workers."""
import sys,json,fcntl,subprocess,time,argparse,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.contact_benchmark_v12.common import *
PYTHON=str(ROOT/'runs/phystwin-env/bin/python')

def create():
    protocol=json.loads((REPORT/'benchmark-protocol.json').read_text());jobs=[]
    for cohort in ['development','regression','untouched_confirmation']:
        group=protocol[cohort]
        for spec in group['SoftVTBench']+[dict(case=c) for c in group['PhysTwin']]:
            for validation in ([True] if cohort=='regression' else [False,True]):
                cmd=[PYTHON,'scripts/contact_benchmark_v12/base_runner.py','--device','cpu']
                for k,v in spec.items():cmd+=['--'+k,str(v)]
                if validation:cmd+=['--validation']
                jobs.append(dict(index=len(jobs),kind='base',cohort=cohort,spec=spec,validation=validation,command=cmd,state='pending'))
    for spec in protocol['untouched_confirmation']['SoftVTBench']:
        cmd=[PYTHON,'scripts/contact_benchmark_v12/native_runner.py','--tag','confirmation','--epochs','100']
        for k,v in spec.items():cmd+=['--'+k,str(v)]
        jobs.append(dict(index=len(jobs),kind='native',cohort='untouched_confirmation',spec=spec,command=cmd,state='pending'))
    for j in jobs:write_json(BASE/'jobs'/f'{j["index"]:03d}.json',j)
    write_json(REPORT/'execution-plan.json',dict(jobs=jobs,created_utc=datetime.now(timezone.utc).isoformat()))
    print('Created',len(jobs),'jobs')

def worker(name,kind):
    check_deadline();(BASE/'jobs').mkdir(exist_ok=True,parents=True)
    while True:
        check_deadline();selected=None
        with open(BASE/'queue.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for path in sorted((BASE/'jobs').glob('*.json')):
                j=json.loads(path.read_text())
                if j['state']=='pending' and j['kind']==kind:
                    j.update(state='running',worker=name,started_utc=datetime.now(timezone.utc).isoformat());write_json(path,j);selected=(path,j);break
        if selected is None:break
        path,j=selected;log=path.with_suffix('.log');start=time.time()
        with open(log,'w') as output:
            try:
                p=subprocess.run(j['command'],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=2400)
                code=p.returncode
            except subprocess.TimeoutExpired:code=124
        j.update(state='complete' if code==0 else 'failed',returncode=code,elapsed_seconds=time.time()-start,log=str(log),finished_utc=datetime.now(timezone.utc).isoformat());write_json(path,j)
        print(name,j['index'],j['state'],round(j['elapsed_seconds']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--create',action='store_true');p.add_argument('--worker');p.add_argument('--kind',choices=['base','native']);a=p.parse_args()
    create() if a.create else worker(a.worker,a.kind)
