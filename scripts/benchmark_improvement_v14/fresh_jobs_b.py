"""Bounded input export and isolated reference-fit queue for forty fresh episodes."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.benchmark_improvement_v14.common import BASE, ROOT, REPORT, remaining, write, read, sha

BASE = BASE / 'robust'
REPORT = REPORT / 'robust'
PYTHON = str(ROOT/'runs/phystwin-env/bin/python')
QUEUE = BASE/'fresh-jobs'
LOCK = threading.Lock()


def specs():
    return [dict(subset=subset,task=task,episode=episode)
            for episode in range(40,50) for task in [1,2]
            for subset in ['object-soft','spatial-soft']]


def prepare():
    from scripts.prefix_ridge_switch_v13.inputs import load_pair,save_payload,save_scene
    cases=[]
    for spec in specs():
        remaining();name=f"{spec['subset']}-task{spec['task']}-demo{spec['episode']}"
        folder=BASE/'inputs'/name
        if (folder/'export.json').exists():
            export=read(folder/'export.json')
            if export['spec']!=spec or any(sha(folder/key)!=digest for key,digest in export['files'].items()):
                raise ValueError('Input export cache mismatch: '+name)
        else:
            full,short=load_pair(spec)
            save_payload(folder/'full.npz',full);save_payload(folder/'short.npz',short)
            save_scene(folder/'scene.npz',full)
            write(folder/'export.json',dict(spec=spec,files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
                access='Object states prefix only; complete prescribed controls; initial static geometry'))
        cases.append(dict(name=name,spec=spec))
        print('exported',name,flush=True)
    write(REPORT/'fresh-reference-plan.json',dict(cases=cases,frames=36,maximum_iterations=100,
         fit_timeout_seconds=2400,checkpoint_continuation_seconds=1800,
         unchanged_reference='v12 current field-enabled contact FEM, independent full and short prefixes',
         source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/benchmark_improvement_v14/fresh_fit_b.py']}))
    QUEUE.mkdir(parents=True,exist_ok=True)
    for case in cases:
        for tag in ['full','short']:
            path=QUEUE/(case['name']+'-'+tag+'.json')
            if not path.exists():write(path,dict(**case,tag=tag,state='pending'))


def claim(device):
    with LOCK:
        for path in sorted(QUEUE.glob('*.json')):
            job=read(path)
            if job['state']!='pending':continue
            job.update(state='running',device=device,started_unix=time.time())
            write(path,job);return path,job
    return None


def worker(device):
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    while remaining()>60:
        selected=claim(device)
        if selected is None:return
        path,job=selected
        command=[PYTHON,'scripts/benchmark_improvement_v14/fresh_fit_b.py','--name',job['name'],
                 '--tag',job['tag'],'--device',device]
        start=time.monotonic()
        with open(path.with_suffix('.log'),'w') as log:
            try:code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                    timeout=min(2400,remaining())).returncode
            except subprocess.TimeoutExpired:code=124
        job['fit_returncode']=code
        if code==124 and remaining()>60:
            out=BASE/'fresh-base/current'/job['tag']/job['name']
            if (out/'best.npz').exists():
                with open(path.with_suffix('.continuation.log'),'w') as log:
                    try:code=subprocess.run(command+['--continuation'],cwd=ROOT,env=env,stdout=log,
                        stderr=subprocess.STDOUT,timeout=min(1800,remaining())).returncode
                    except subprocess.TimeoutExpired:code=124
                job['continuation_returncode']=code
        job.update(state='complete' if code==0 else 'failed',returncode=code,
                   elapsed_seconds=time.monotonic()-start)
        write(path,job);print(device,job['name'],job['tag'],job['state'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    p.add_argument('--workers',action='store_true');p.add_argument('--cpu-workers',type=int,default=6)
    a=p.parse_args()
    if a.prepare:prepare()
    if a.workers:
        devices=['cuda:0','cuda:1','cuda:2']+['cpu']*a.cpu_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures=[pool.submit(worker,d) for d in devices]
            for future in futures:future.result()
