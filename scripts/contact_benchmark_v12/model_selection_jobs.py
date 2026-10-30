"""Isolated296-job CPU queue for the one fixed exploratory selection study."""
import argparse,fcntl,json,subprocess,time,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.contact_benchmark_v12.common import BASE,ROOT,REPORT,check_deadline,write_json,sha256
from scripts.contact_benchmark_v12.data import load,predictor_payload
from scripts.contact_benchmark_v12.fieldfree_runner import check_freeze,settings,OUTPUT,PROTOCOL,FREEZE,terminal_failure

QUEUE=BASE/'model-selection-jobs';LOCK=BASE/'model-selection-queue.lock'
PLAN=REPORT/'model-selection-execution-plan.json';PYTHON=str(ROOT/'runs/phystwin-env/bin/python')


def now():return datetime.now(timezone.utc).isoformat()


def planned_jobs():
    old=json.loads((REPORT/'benchmark-protocol.json').read_text());fresh=json.loads((REPORT/'feasibility-protocol.json').read_text());jobs=[];seen=set()
    for cohort in ['development','regression','untouched_confirmation','fresh_confirmation']:
        group=(fresh if cohort=='fresh_confirmation' else old)[cohort]
        for spec in group['SoftVTBench']+[dict(case=c) for c in group['PhysTwin']]:
            name=spec.get('case') or f"{spec['subset']}-task{spec['task']}-demo{spec['episode']}"
            if name in seen:raise ValueError('Duplicate model-selection case: '+name)
            seen.add(name)
            for validation in [False,True]:
                command=[PYTHON,'scripts/contact_benchmark_v12/fieldfree_runner.py','--device','cpu']
                for key,value in spec.items():command+=['--'+key,str(value)]
                if validation:command.append('--validation')
                jobs.append(dict(index=2000+len(jobs),kind='field_free',cohort=cohort,spec=spec,validation=validation,command=command,state='pending'))
    assert len(seen)==148 and len(jobs)==296
    return jobs


def create():
    check_deadline();check_freeze();jobs=planned_jobs();QUEUE.mkdir(parents=True,exist_ok=True)
    with open(LOCK,'a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        if list(QUEUE.glob('*.json')):raise RuntimeError('Refusing to overwrite an existing model-selection queue')
        for job in jobs:write_json(QUEUE/f"{job['index']:04d}.json",job)
        write_json(PLAN,dict(created_utc=now(),protocol_sha256=sha256(PROTOCOL),freeze_sha256=sha256(FREEZE),
            cohort_protocol_sha256={name:sha256(REPORT/name) for name in ['benchmark-protocol.json','feasibility-protocol.json']},
            jobs=jobs,timeout_seconds=2400,timeout_continuation_seconds=1800,timeout_policy='Validate finite prefix checkpoint, preserve originals, one bounded unchanged-parameter prediction continuation'))
    status()


def claim(worker,repair=False):
    check_freeze()
    if not QUEUE.exists():raise FileNotFoundError('Create queue after source/protocol freeze first')
    with open(LOCK,'a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        for path in sorted(QUEUE.glob('*.json')):
            job=json.loads(path.read_text())
            eligible=job['state']=='pending' if not repair else job['state']=='failed' and job.get('returncode')==124 and not job.get('timeout_repair_attempted') and not job.get('repair_claimed')
            if eligible:
                if repair:job.update(state='repairing',repair_claimed=True,repair_worker=worker)
                else:job.update(state='running',worker=worker,started_utc=now())
                write_json(path,job);return path,job
    return None


def worker(name):
    while True:
        check_deadline();selected=claim(name)
        if selected is None:break
        path,job=selected;start=time.monotonic();log=path.with_suffix('.log')
        with open(log,'w') as handle:
            try:code=subprocess.run(job['command'],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,timeout=2400).returncode
            except subprocess.TimeoutExpired:code=124
        job.update(state='complete' if code==0 else 'failed',returncode=code,elapsed_seconds=time.monotonic()-start,log=str(log),finished_utc=now())
        if code not in [0,124]:
            record=terminal_failure(job['spec'],job['validation'],f'Fitting process exited with code {code}; see {log}',job['index'])
            job['terminal_failure_artifact']=record
        write_json(path,job);print(name,job['index'],job['state'],flush=True)
    status()


def continue_timeout(path,job):
    """Same bounded checkpoint policy in isolated field-free paths; no refit."""
    check_deadline();check_freeze();data=predictor_payload(load(job['spec'],job['validation']));tag='validation' if job['validation'] else 'full';out=OUTPUT/tag/data['name']
    saved=np.load(out/'best.npz');progress=json.loads((out/'progress.json').read_text());best=progress['best']
    np.testing.assert_array_equal(saved['nodes'],data['nodes']);np.testing.assert_array_equal(saved['theta'],best['theta'])
    np.testing.assert_array_equal(saved['selected_frames'],np.unique(np.linspace(0,data['train_end']-1,36,dtype=int)))
    assert progress['fit_end']==data['train_end'] and np.isfinite(saved['theta']).all() and np.isfinite(best['loss']) and best['loss']<1000
    assert saved['theta'][2+len(data['attachment_weights'])+settings(data)['materials']]==0.
    snapshot=BASE/'model-selection-timeout-records'/str(job['index']);snapshot.mkdir(parents=True,exist_ok=True)
    write_json(snapshot/'original-job.json',job)
    for file in ['best.npz','progress.json']:shutil.copy2(out/file,snapshot/file)
    checkpoint_sha=sha256(out/'best.npz');job['timeout_repair_attempted']=True
    if (out/'fit.json').exists():
        fit=json.loads((out/'fit.json').read_text());np.testing.assert_array_equal(fit['best']['theta'],saved['theta']);shutil.copy2(out/'fit.json',snapshot/'completed-fit.json')
        phase='prediction after completed fit'
    else:
        fit=dict(state='budget_exhausted_valid_checkpoint',name=data['name'],dataset=data['dataset'],fit_end=data['train_end'],selected_frames=saved['selected_frames'].tolist(),best=best,
                 optimizer=dict(success=False,message='2400second job budget exhausted; unchanged valid prefix checkpoint retained',iterations=None),history=[best],failed_trials=progress.get('failures',[]),elapsed_seconds=job['elapsed_seconds'],**settings(data),timeout_checkpoint_sha256=checkpoint_sha,interpretation='Unfinished optimizer, no convergence claim or future-error selection')
        write_json(out/'fit.json',fit);phase='unfinished optimization checkpoint'
    job['repair_phase']=phase;write_json(path,job);start=time.monotonic();log=snapshot/'continuation.log'
    with open(log,'w') as handle:
        try:code=subprocess.run(job['command'],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,timeout=1800).returncode
        except subprocess.TimeoutExpired:code=124
    assert sha256(out/'best.npz')==checkpoint_sha
    job.update(state='complete' if code==0 else 'failed',repair_returncode=code,repair_elapsed_seconds=time.monotonic()-start,repair_log=str(log),repair_checkpoint_sha256=checkpoint_sha,optimizer_converged=fit['optimizer']['success'])
    if code!=0:
        job['terminal_failure_artifact']=terminal_failure(job['spec'],job['validation'],f'Bounded checkpoint continuation exited with code {code}; see {log}',job['index'])
    write_json(path,job)


def repairs(watch=False):
    while True:
        check_deadline();selected=claim('field-free-timeout-continuation',True)
        if selected:
            path,job=selected
            try:continue_timeout(path,job)
            except Exception as error:
                job.update(state='failed',timeout_repair_attempted=True,repair_validation_error=str(error))
                job['terminal_failure_artifact']=terminal_failure(job['spec'],job['validation'],'Timeout continuation unavailable: '+str(error),job['index'])
                write_json(path,job)
            continue
        live=any(j['state'] in ['pending','running','repairing'] for j in read_jobs())
        if not watch or not live:break
        time.sleep(20)
    status()


def read_jobs():return [json.loads(p.read_text()) for p in sorted(QUEUE.glob('*.json'))]


def status():
    jobs=read_jobs();value=dict(updated_utc=now(),total=len(jobs),states=dict(Counter(j['state'] for j in jobs)),running=[dict(index=j['index'],spec=j['spec'],validation=j['validation'],worker=j.get('worker')) for j in jobs if j['state'] in ['running','repairing']])
    write_json(BASE/'model-selection-progress.json',value);print(json.dumps(value,indent=2));return value

if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--create',action='store_true');mode.add_argument('--worker');mode.add_argument('--repair',action='store_true');mode.add_argument('--status',action='store_true');parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if args.create:create()
    elif args.worker:worker(args.worker)
    elif args.repair:repairs(args.watch)
    else:status()
