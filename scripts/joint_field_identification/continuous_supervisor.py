"""Durable deadline-limited queue for frozen continuous-identification jobs.

No predictions or scores are consulted for scheduling. Jobs are prioritized by
declared cohort then fixed index. All failures/incomplete jobs remain recorded.
"""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
import numpy as np
from . import bank
from .continuous_campaign import job_key,BASE,REPORT,sha

DEADLINE='2026-11-07T07:37:29Z'


def name(job):
    return f"{job['shape']}-{job['population']}-case{job['case']:03d}-{job['stage']}-n{job['particles']}-s{job['seed']}"


def atomic_json(path,doc):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(doc,indent=2));tmp.replace(path)


def make_manifest(path):
    path=Path(path)
    if path.exists():raise FileExistsError('Frozen manifest exists; refusing overwrite')
    jobs=[]
    for case in range(32):
        for stage in ['initial','active','fixed']:
            for shape in ['block','step']:
                jobs.append(job_key(shape,'continuous-coarse',case,stage,1024,81001,[8,4,4]))
    main_count=len(jobs)
    for particles in [1024,2048]:
        for case in [0,1]:
            for population in ['matched','offgrid-fine']:
                for stage in ['initial','active','fixed']:
                    for seed in [71001,71002]:
                        for shape in ['block','step']:
                            jobs.append(job_key(shape,population,case,stage,particles,seed,[8,4,4]))
    sources=[bank.ROOT/'scripts/joint_field_identification'/f for f in
             ['bank.py','mechanics.py','inference.py','continuous.py','campaign.py',
              'continuous_campaign.py','continuous_supervisor.py']]
    doc=dict(version='continuous-joint-fields-v01',created_utc=datetime.now(timezone.utc).isoformat(),
             deadline_utc=DEADLINE,bounds=bank.BOUNDS.tolist(),moves=4,batch_size=16,
             main_jobs=main_count,diagnostic_jobs=len(jobs)-main_count,allowed_jobs=jobs,
             source_sha256={str(p.relative_to(bank.ROOT)):sha(p) for p in sources},
             main_cohort='32 independently generated continuous-uniform truths per shape, same8x4x4 finite mechanics.',
             main_seed=81001,diagnostic_seeds=[71001,71002],diagnostic_counts=[1024,2048],
             action_budget=4,
             acquisition='Replay initial and final active/fixed evidence from the frozen finite-dictionary planner. No continuous policy superiority claim.',
             inference='Uniform in declared theta coordinates; exact force elimination conditional on theta, approximate continuous posterior by SMC.',
             convergence='Compare independent seeds and particle counts for first2matched/offgrid cases per shape; retain all differences. No post-resampling ESS convergence claim.',
             no_truth_selection=True,raw_data_local=True)
    main_protocol=bank.ROOT/'reports/joint-field-identification-v01/protocol.json'
    if main_protocol.exists():
        doc['main_protocol_sha256']=sha(main_protocol)
        doc['main_protocol_path']=str(main_protocol.relative_to(bank.ROOT))
    atomic_json(path,doc);return doc


def ready(job,budget):
    data=BASE/job['shape']/f"{job['population']}-truth.npz"
    if not data.exists():return False,'truth archive not generated'
    try:
        with np.load(data) as a:
            if len(a['y'])<=job['case']:return False,'case absent'
    except (ValueError,OSError,EOFError,zipfile.BadZipFile):return False,'truth archive still being written'
    if job['stage']=='initial':return True,''
    record=bank.ROOT/f"reports/joint-field-identification-v01/{job['shape']}/{job['population']}/case-{job['case']:03d}.json"
    if not record.exists():return False,'finite acquisition record not complete'
    try:doc=json.loads(record.read_text())
    except (ValueError,OSError):return False,'finite acquisition record still being written'
    rows=[r for r in doc.get('rows',[]) if r.get('policy')==job['stage'] and r.get('step',0)>0]
    if sorted(r['step'] for r in rows)!=list(range(1,budget+1)):return False,'acquisition budget incomplete'
    return True,''


def supervise(manifest,device,shard,shards,poll=5.):
    manifest=Path(manifest);doc=json.loads(manifest.read_text());digest=sha(manifest)
    deadline=datetime.fromisoformat(doc['deadline_utc'].replace('Z','+00:00')).timestamp()
    if device not in ['cuda:0','cuda:1','cuda:2']:raise ValueError('Only GPUs0-2 authorized')
    if not 0<=shard<shards:raise ValueError('Invalid shard')
    jobs=[(i,j) for i,j in enumerate(doc['allowed_jobs']) if i%shards==shard]
    statuspath=REPORT/f'supervisor-{shard}.json'
    previous=json.loads(statuspath.read_text()) if statuspath.exists() else {}
    if previous and previous.get('manifest_sha256')!=digest:raise ValueError('Supervisor manifest changed')
    failures=previous.get('failures',{})
    state=dict(pid=os.getpid(),device=device,shard=shard,shards=shards,manifest_sha256=digest,
               deadline_utc=doc['deadline_utc'],failures=failures,started_utc=datetime.now(timezone.utc).isoformat())
    logroot=BASE/'continuous/logs';logroot.mkdir(parents=True,exist_ok=True)
    def write(status,current=None,completed=None,pending=None):
        state.update(status=status,current=current,heartbeat_utc=datetime.now(timezone.utc).isoformat())
        if completed is not None:state['completed']=completed
        if pending is not None:state['pending']=pending
        atomic_json(statuspath,state)
    while True:
        if time.time()>=deadline:
            write('deadline_reached_incomplete');return
        for filename,expected in doc['source_sha256'].items():
            if sha(bank.ROOT/filename)!=expected:
                write('blocked_source_changed',filename);raise RuntimeError(f'Frozen source changed: {filename}')
        pending=[];completed=[];eligible=[]
        for index,job in jobs:
            key=name(job);report=REPORT/(key+'.json')
            if report.exists():
                try:
                    old=json.loads(report.read_text())
                    if old.get('complete'):
                        if old['protocol']['manifest_sha256']!=digest:raise ValueError('Completed job uses another manifest')
                        completed.append(key);continue
                except json.JSONDecodeError:pass
            if key in failures:continue
            available,reason=ready(job,doc['action_budget'])
            pending.append(dict(name=key,index=index,reason=reason))
            if available:eligible.append((index,job))
        if not pending:
            write('complete' if not failures else 'finished_with_failures',completed=completed,pending=[]);return
        if not eligible:
            write('waiting_for_finite_records',completed=completed,pending=pending)
            time.sleep(min(poll,max(0.,deadline-time.time())));continue
        index,job=min(eligible,key=lambda v:v[0]);key=name(job)
        cmd=[sys.executable,'-m','scripts.joint_field_identification.continuous_campaign',
             '--shape',job['shape'],'--population',job['population'],'--case',str(job['case']),
             '--stage',job['stage'],'--particles',str(job['particles']),'--seed',str(job['seed']),
             '--cells',*map(str,job['cells']),'--device',device,'--batch',str(doc['batch_size']),
             '--moves',str(doc['moves']),'--manifest',str(manifest),'--resume']
        write('running',key,completed,pending)
        with (logroot/(key+'.log')).open('a') as log:
            log.write('\nSUPERVISOR START '+datetime.now(timezone.utc).isoformat()+'\n');log.flush()
            child=subprocess.Popen(cmd,cwd=bank.ROOT,stdout=log,stderr=subprocess.STDOUT,
                                   env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            state['child_pid']=child.pid
            while child.poll() is None:
                if time.time()>=deadline:
                    child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    failures[key]=dict(reason='deadline',returncode=child.returncode)
                    write('deadline_reached_incomplete',key);return
                write('running',key)
                time.sleep(min(poll,max(0.,deadline-time.time())))
        state.pop('child_pid',None)
        if child.returncode:
            failures[key]=dict(reason='worker_failed',returncode=child.returncode,log=str(logroot/(key+'.log')))
            write('worker_failed',key)
        else:
            report=REPORT/(key+'.json')
            if not report.exists() or not json.loads(report.read_text()).get('complete'):
                failures[key]=dict(reason='worker_returned_without_complete_report',returncode=0)
        print(json.dumps(dict(shard=shard,job=key,returncode=child.returncode)),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default=str(REPORT/'manifest.json'))
    ap.add_argument('--create-manifest',action='store_true');ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=2)
    args=ap.parse_args()
    if args.create_manifest:
        doc=make_manifest(args.manifest);print(json.dumps(dict(main_jobs=doc['main_jobs'],diagnostic_jobs=doc['diagnostic_jobs'],manifest=args.manifest)));return
    supervise(args.manifest,args.device,args.shard,args.shards)


if __name__=='__main__':main()
