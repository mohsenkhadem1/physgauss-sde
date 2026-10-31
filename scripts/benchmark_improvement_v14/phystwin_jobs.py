"""Three GPU workers, sixteen independent prefix fits, fixed absolute deadline."""
import concurrent.futures
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.benchmark_improvement_v14.common import BASE,ROOT,REPORT,read,write,remaining,sha
from scripts.benchmark_improvement_v14.phystwin_production import NAMES,freeze,validate_fit
QUEUE=BASE/'phystwin-jobs';LOCK=threading.Lock()


def claim(device):
    with LOCK:
        for path in sorted(QUEUE.glob('*.json')):
            job=read(path)
            if job['state']!='pending':continue
            job.update(state='running',device=device,started_unix=time.time());write(path,job)
            return path,job
    return None


def worker(device):
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    while remaining()>10:
        selected=claim(device)
        if selected is None:return
        path,job=selected;start=time.monotonic()
        command=[str(ROOT/'runs/phystwin-env/bin/python'),'scripts/benchmark_improvement_v14/phystwin_production.py',
                 '--name',job['name'],'--tag',job['tag'],'--device',device]
        with open(path.with_suffix('.log'),'w') as log:
            try:code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(3020,remaining())).returncode
            except subprocess.TimeoutExpired:code=124
        if code==0:validate_fit(job['name'],job['tag'])
        job.update(state='complete' if code==0 else 'failed',returncode=code,elapsed_seconds=time.monotonic()-start)
        write(path,job);print(device,job['name'],job['tag'],job['state'],flush=True)


if __name__=='__main__':
    frozen=freeze();QUEUE.mkdir(parents=True,exist_ok=True)
    for name in NAMES:
        for tag in ['short','full']:
            path=QUEUE/(name+'-'+tag+'.json')
            if not path.exists():write(path,dict(name=name,tag=tag,state='pending',freeze_sha256=frozen))
            elif read(path)['freeze_sha256']!=frozen:raise ValueError('Queue freeze changed')
    write(REPORT/'phystwin-jobs-plan.json',dict(names=list(NAMES),tags=['short','full'],jobs=16,devices=['cuda:0','cuda:1','cuda:2'],
          freeze_sha256=frozen,parent_timeout_seconds=3020,source_sha256=sha(Path(__file__))))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(worker,device) for device in ['cuda:0','cuda:1','cuda:2']]
        for future in futures:future.result()
