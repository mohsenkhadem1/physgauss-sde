"""Parallel fixed-recipe calibrated probes; one unchanged Isaac worker per GPU.

The worker source is frozen. This coordinator only schedules known episode IDs,
validates completed cases and preserves interruptions/failures. No resampling,
physics modification, image interpolation or acceptance-threshold change.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

WORKER_SHA256='b33d66ac9057885dcb4afeeee419dcc672afe91f888b0f73c1d30f961ca813c0'
ISAAC_PYTHON='/home/bayespc2/vscode/deformable worldmodel/.venv/bin/python'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stamp():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+f'.tmp-{os.getpid()}-{threading.get_ident()}')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def choose_cases(plan,split,episode_ids=None,limit=16):
    cases={c['episode_id']:c for c in plan['episodes']}
    if episode_ids is None:
        if split!='train':raise ValueError('Non-TRAIN pools require explicit predeclared episode IDs')
        selected=[c for c in plan['episodes'] if c['split']==split][:limit]
    else:
        if len(episode_ids)!=len(set(episode_ids)):raise ValueError('Duplicate episode IDs')
        if any(e not in cases or cases[e]['split']!=split for e in episode_ids):raise ValueError('Unknown episode or split mismatch')
        selected=[cases[e] for e in episode_ids]
    if not selected:raise ValueError('No selected cases')
    return selected


def process_info(pid):
    path=Path('/proc')/str(pid)
    try:
        status=(path/'stat').read_text();state=status.rsplit(')',1)[1].strip().split()[0]
        command=(path/'cmdline').read_bytes().split(b'\0');command=[x.decode(errors='replace') for x in command if x]
        return {'pid':pid,'state':state,'command':command}
    except FileNotFoundError:return None


def takeover_driver(pid,pool,timeout_s=1200):
    """Only explicit caller authorization may stop an existing serial driver."""
    pool=Path(pool).resolve();info=process_info(pid)
    if info is None or info['state']=='Z':return {'driver_pid':pid,'already_exited':True}
    command=info['command']
    if not any(Path(x).name=='isaac_calibrated_probes_v02.py' for x in command) or '--episode' in command:
        raise ValueError('Takeover PID is not the serial probe DRIVER')
    if '--output' not in command or Path(command[command.index('--output')+1]).resolve()!=pool:
        raise ValueError('Takeover driver belongs to another output pool')
    ledger_path=pool/'coordinator-history'/f'takeover-{stamp()}.json';ledger_path.parent.mkdir(parents=True,exist_ok=True)
    ledger={'driver_before':info,'started_utc':utc(),'policy':'Signal DRIVER PID only; child finishes unchanged; no child kill','states':[]}
    atomic_json(ledger_path,ledger)
    os.kill(pid,signal.SIGSTOP);ledger['driver_stopped_utc']=utc();atomic_json(ledger_path,ledger)
    try:
        children=[int(x) for x in (Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()]
        episodes=[]
        for child in children:
            child_info=process_info(child)
            if child_info is None:continue
            args=child_info['command']
            if child_info['state']!='Z' and ('--episode' not in args or not any(Path(x).name=='isaac_calibrated_probes_v02.py' for x in args)):
                raise ValueError('Unexpected live child; refusing takeover')
            if '--episode' in args:episodes.append(args[args.index('--episode')+1])
            ledger['states'].append(child_info)
        atomic_json(ledger_path,ledger)
        started=time.monotonic()
        while any((p:=process_info(child)) is not None and p['state']!='Z' for child in children):
            if time.monotonic()-started>timeout_s:raise TimeoutError('Current worker did not finish at case boundary')
            time.sleep(1)
        # SIGTERM is pending while stopped; SIGCONT delivers it BEFORE the
        # Python driver can resume its loop. No process-group signals are used.
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        started=time.monotonic()
        while (p:=process_info(pid)) is not None and p['state']!='Z':
            if time.monotonic()-started>15:raise TimeoutError('Serial driver did not exit after SIGTERM')
            time.sleep(.1)
        ledger.update(completed_utc=utc(),completed_child_episodes=episodes,driver_after=process_info(pid))
        atomic_json(ledger_path,ledger);return ledger
    except BaseException:
        # A failed preflight must not leave a healthy serial driver suspended.
        if (p:=process_info(pid)) is not None and p['state']!='Z':os.kill(pid,signal.SIGCONT)
        ledger['error']='Takeover failed; live driver resumed';atomic_json(ledger_path,ledger)
        raise


def archive_interrupted_case(pool,eid):
    pool=Path(pool);destination=pool/'rejected-attempts'/f'coordinator-interruption-{stamp()}'/eid
    destination.mkdir(parents=True)
    moved=[]
    for rel in [f'private/{eid}',f'public/{eid}',f'outcomes/{eid}',f'logs/{eid}.log']:
        source=pool/rel
        if source.exists():
            target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.move(source,target);moved.append(rel)
    atomic_json(destination/'reason.json',{'episode_id':eid,'reason':'Explicit resume of process-interrupted same recipe; no failure.json and no outcome-based resampling','moved':moved,'archived_utc':utc()})
    return str(destination)


def validate_completed(pool,eid):
    from isaac_calibrated_probes_v02 import validate_case
    audit=validate_case(pool,eid)
    atomic_json(Path(pool)/'private'/eid/'validation.json',audit)
    return audit


def run_fixed_queue(cases,gpus,invoke,stop_event=None,on_result=None):
    """Pure scheduling core: each worker thread owns exactly one GPU."""
    if len(gpus)!=len(set(gpus)) or not gpus:raise ValueError('Distinct nonempty GPU list required')
    stop_event=threading.Event() if stop_event is None else stop_event
    pending=queue.Queue();[pending.put(case) for case in cases]
    results=[];mutex=threading.Lock()
    def loop(gpu):
        while not stop_event.is_set():
            try:case=pending.get_nowait()
            except queue.Empty:break
            try:result=invoke(case,gpu)
            except BaseException as error:result={'episode_id':case['episode_id'],'gpu':gpu,'passed':False,'error':repr(error)}
            with mutex:
                results.append(result)
                if on_result:on_result(result,list(results))
            pending.task_done()
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures=[executor.submit(loop,gpu) for gpu in gpus]
        for future in futures:future.result()
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',type=int,required=True);p.add_argument('--split',choices=['train','validation','test'],default='train')
    p.add_argument('--episodes',nargs='+');p.add_argument('--limit',type=int,default=16)
    p.add_argument('--isaac-python',default=ISAAC_PYTHON);p.add_argument('--takeover-driver-pid',type=int)
    p.add_argument('--worker-script',type=Path,default=Path(__file__).with_name('isaac_calibrated_probes_v02.py'))
    p.add_argument('--worker-sha256',default=WORKER_SHA256)
    p.add_argument('--archive-interrupted',action='store_true');p.add_argument('--case-timeout-s',type=float,default=1200)
    a=p.parse_args()
    if not a.gpus or len(a.gpus)!=len(set(a.gpus)):p.error('Distinct nonempty GPU list required')
    worker=a.worker_script.resolve();worker_hash=a.worker_sha256
    if worker.name not in ['isaac_calibrated_probes_v02.py','isaac_calibrated_probes_v03.py'] or sha(worker)!=worker_hash:
        raise ValueError('Frozen Isaac worker fingerprint/version differs')
    plan=json.loads((a.dataset/'plan.json').read_text());cases=choose_cases(plan,a.split,a.episodes,a.limit)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'logs').mkdir(exist_ok=True)
    if a.takeover_driver_pid:takeover_driver(a.takeover_driver_pid,a.output,a.case_timeout_s)
    locks=[]
    # Same legacy lock excludes its serial driver; second lock distinguishes
    # this scheduler. Both remain held until all active children finish.
    for name in ['.probe-generation.lock','.probe-pool-coordinator.lock']:
        handle=(a.output/name).open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
    history=a.output/'coordinator-history';history.mkdir(exist_ok=True);run_id=stamp()
    if (a.output/'execution.json').exists():shutil.copy2(a.output/'execution.json',history/f'execution-before-{run_id}.json')
    ledger={'coordinator_pid':os.getpid(),'started_utc':utc(),'dataset':str(a.dataset.resolve()),'output':str(a.output.resolve()),
        'source_sha256':{'worker':worker_hash,'coordinator':sha(__file__)},'dataset_plan_sha256':sha(a.dataset/'plan.json'),
        'gpus':a.gpus,'split':a.split,'episode_ids':[c['episode_id'] for c in cases],
        'recipe_fingerprints':{c['episode_id']:hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest() for c in cases},
        'selection':'Fixed requested IDs; no outcome-based resampling','cases':[],'completed':False}
    ledger_path=history/f'coordinator-{run_id}.json';atomic_json(ledger_path,ledger)
    stop=threading.Event()
    def request_stop(signum,frame):
        stop.set();print(json.dumps({'stop_requested':signum,'policy':'Active workers finish; no new cases start'}),flush=True)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def invoke(case,gpu):
        eid=case['episode_id'];started=time.monotonic();private=a.output/'private'/eid
        claims=a.output/'claims';claims.mkdir(exist_ok=True)
        with (claims/(eid+'.lock')).open('a+') as claim:
            fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
            has_artifacts=any((a.output/rel).exists() for rel in [f'private/{eid}',f'public/{eid}',f'outcomes/{eid}',f'logs/{eid}.log'])
            if has_artifacts:
                if (private/'summary.json').exists():
                    if json.loads((private/'truth/recipe.json').read_text())!=case:raise ValueError('Existing case belongs to another frozen recipe')
                    validate_completed(a.output,eid)
                    return {'episode_id':eid,'gpu':gpu,'passed':True,'status':'resumed_verified','elapsed_s':time.monotonic()-started}
                if (private/'failure.json').exists():raise ValueError('Preserved failed case cannot be silently retried')
                if not a.archive_interrupted:raise ValueError('Partial case requires explicit --archive-interrupted')
                archive_interrupted_case(a.output,eid)
            command=[a.isaac_python,'-u',str(worker),'--dataset',str(a.dataset.resolve()),'--output',str(a.output.resolve()),'--episode',eid,'--gpu',str(gpu)]
            if sha(worker)!=worker_hash:raise RuntimeError('Frozen worker changed')
            with (a.output/'logs'/f'{eid}.log').open('x') as logfile:
                process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=logfile,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES'},start_new_session=True)
                atomic_json(claims/(eid+'.json'),{'episode_id':eid,'gpu':gpu,'worker_pid':process.pid,'command':command,'started_utc':utc(),'coordinator_pid':os.getpid(),'worker_sha256':worker_hash})
                try:code=process.wait(timeout=a.case_timeout_s)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise TimeoutError('Worker exceeded declared timeout; partial case preserved')
            if code:raise RuntimeError(f'Worker exited{code}; failure/partial data preserved')
            if json.loads((private/'truth/recipe.json').read_text())!=case:raise ValueError('Worker recipe differs from fixed queue')
            if sha(worker)!=worker_hash:raise RuntimeError('Frozen worker changed during simulation')
            validate_completed(a.output,eid)
            return {'episode_id':eid,'gpu':gpu,'passed':True,'status':'generated_verified','elapsed_s':time.monotonic()-started}
    def result_callback(result,results):
        ledger['cases']=results;ledger['updated_utc']=utc();atomic_json(ledger_path,ledger)
        atomic_json(a.output/'execution.json',{'cases':results,'completed':False,'coordinator_ledger':str(ledger_path)})
        print(json.dumps(result),flush=True)
    results=run_fixed_queue(cases,a.gpus,invoke,stop,result_callback)
    completed=len(results)==len(cases) and all(r['passed'] for r in results)
    ledger.update(cases=results,completed=completed,finished_utc=utc(),stop_requested=stop.is_set())
    atomic_json(ledger_path,ledger);atomic_json(a.output/'execution.json',{'cases':results,'completed':completed,'coordinator_ledger':str(ledger_path)})
    return not completed


if __name__=='__main__':raise SystemExit(main())
