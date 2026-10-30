"""Separate fresh-cohort queue; never edits original209-job queue/finalizer.

Uses unchanged base/native runners and declared timeout-checkpoint policy.
Indices1000+ avoid collision with original timeout-record directory names.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.contact_benchmark_v12.common import BASE,ROOT,REPORT,check_deadline,write_json,sha256

QUEUE=BASE/'feasibility-jobs'
LOCK=BASE/'feasibility-queue.lock'
PROTOCOL=REPORT/'feasibility-protocol.json'
PYTHON=str(ROOT/'runs/phystwin-env/bin/python')


def now():return datetime.now(timezone.utc).isoformat()


def validate_inputs(group):
    """Presence-only checks: do not read fresh target geometry or scores."""
    import h5py
    missing=[]
    for spec in group['SoftVTBench']:
        family='object' if spec['subset']=='object-soft' else 'spatial'
        folder=ROOT/f'runs/external-benchmark-v11/raw/SoftVTBench/{spec["subset"]}/libero_{family}/libero_{family}_task{spec["task"]}/replayed_demos'
        paths=list(folder.glob('*.hdf5'))
        if len(paths)!=1:missing.append(str(folder));continue
        with h5py.File(paths[0]) as h:
            if f'data/demo_{spec["episode"]}' not in h:missing.append(f'{paths[0]}:demo_{spec["episode"]}')
    for case in group['PhysTwin']:
        folder=ROOT/'runs/third_party/PhysTwin/data/different_types'/case
        for name in ['final_data.pkl','split.json','gt_track_3d.pkl']:
            if not(folder/name).exists():missing.append(str(folder/name))
    if missing:raise FileNotFoundError('Fresh inputs not ready: '+', '.join(missing))


def create():
    check_deadline()
    if not PROTOCOL.exists():raise FileNotFoundError(PROTOCOL)
    protocol=json.loads(PROTOCOL.read_text());group=protocol['fresh_confirmation'];validate_inputs(group)
    if len(group['SoftVTBench'])!=40 or len(group['PhysTwin'])!=2:raise ValueError('Expected prespecified40 SoftVT +2 PhysTwin cohort')
    jobs=[]
    for spec in group['SoftVTBench']+[dict(case=c) for c in group['PhysTwin']]:
        for validation in [False,True]:
            cmd=[PYTHON,'scripts/contact_benchmark_v12/base_runner.py','--device','cpu']
            for key,value in spec.items():cmd+=['--'+key,str(value)]
            if validation:cmd.append('--validation')
            jobs.append(dict(index=1000+len(jobs),kind='base',cohort='fresh_confirmation',spec=spec,validation=validation,command=cmd,state='pending'))
    for spec in group['SoftVTBench']:
        cmd=[PYTHON,'scripts/contact_benchmark_v12/native_runner.py','--tag','confirmation','--epochs','100']
        for key,value in spec.items():cmd+=['--'+key,str(value)]
        jobs.append(dict(index=1000+len(jobs),kind='native',cohort='fresh_confirmation',spec=spec,command=cmd,state='pending'))
    QUEUE.mkdir(parents=True,exist_ok=True)
    with open(LOCK,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if list(QUEUE.glob('*.json')):raise RuntimeError('Queue already exists; refusing to overwrite any job state')
        for job in jobs:write_json(QUEUE/f'{job["index"]:04d}.json',job)
        write_json(REPORT/'feasibility-execution-plan.json',dict(created_utc=now(),protocol_sha256=sha256(PROTOCOL),queue=str(QUEUE.relative_to(ROOT)),timeout_seconds=2400,timeout_continuation_seconds=1800,timeout_policy='Unchanged timeout_repair.repair; snapshots1000+ preserve original queue records',runners={name:sha256(ROOT/'scripts/contact_benchmark_v12'/name) for name in ['base_runner.py','native_runner.py','timeout_repair.py']},jobs=jobs))
    status()


def claim(name,kind,repair=False):
    if not QUEUE.exists():raise FileNotFoundError('Create validated queue first')
    selected=None
    with open(LOCK,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for path in sorted(QUEUE.glob('*.json')):
            job=json.loads(path.read_text())
            eligible=(job['kind']=='base' and job['state']=='failed' and job.get('returncode')==124 and not job.get('timeout_repair_attempted') and not job.get('repair_claimed')) if repair else (job['kind']==kind and job['state']=='pending')
            if eligible:
                if repair:job.update(state='repairing',repair_claimed=True,repair_worker=name)
                else:job.update(state='running',worker=name,started_utc=now())
                write_json(path,job);selected=(path,job);break
    return selected


def worker(name,kind):
    check_deadline()
    while True:
        check_deadline();selected=claim(name,kind)
        if selected is None:break
        path,job=selected;log=path.with_suffix('.log');start=time.monotonic()
        with open(log,'w') as output:
            try:code=subprocess.run(job['command'],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=2400).returncode
            except subprocess.TimeoutExpired:code=124
        job.update(state='complete' if code==0 else 'failed',returncode=code,elapsed_seconds=time.monotonic()-start,log=str(log),finished_utc=now())
        write_json(path,job);print(name,job['index'],job['state'],round(job['elapsed_seconds']),flush=True)
    status()


def repairs(watch=False):
    from scripts.contact_benchmark_v12.timeout_repair import repair
    while True:
        check_deadline();selected=claim('feasibility-timeout-repair','base',True)
        if selected:
            path,job=selected
            try:repair(path,job)
            except Exception as error:
                job.update(state='failed',timeout_repair_attempted=True,repair_validation_error=str(error));write_json(path,job)
                print('REPAIR_REJECTED',job['index'],str(error),flush=True)
            continue
        live=any(j['kind']=='base' and j['state'] in ['pending','running','repairing'] for j in read_jobs())
        if not watch or not live:break
        time.sleep(20)
    status()


def read_jobs():return [json.loads(p.read_text()) for p in sorted(QUEUE.glob('*.json'))]


def status():
    jobs=read_jobs();report=dict(updated_utc=now(),total=len(jobs),states=dict(Counter(j['state'] for j in jobs)),by_kind={kind:dict(Counter(j['state'] for j in jobs if j['kind']==kind)) for kind in ['base','native']},running=[dict(index=j['index'],kind=j['kind'],spec=j['spec'],worker=j.get('worker'),started_utc=j.get('started_utc')) for j in jobs if j['state'] in ['running','repairing']])
    write_json(BASE/'feasibility-progress.json',report);print(json.dumps(report,indent=2),flush=True);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();m=parser.add_mutually_exclusive_group(required=True)
    m.add_argument('--create',action='store_true');m.add_argument('--worker');m.add_argument('--repair',action='store_true');m.add_argument('--status',action='store_true')
    parser.add_argument('--kind',choices=['base','native']);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if args.create:create()
    elif args.worker:
        if not args.kind:parser.error('--worker requires --kind')
        worker(args.worker,args.kind)
    elif args.repair:repairs(args.watch)
    else:status()
