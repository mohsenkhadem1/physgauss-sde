"""Bounded one-shot capture controller. Never parses truth or response arrays.

No inference/calibration analysis, retry/resume, redrawing, role mixing or GPU.
Root must invoke run only after the plan and all source pins are reviewed/frozen.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
try:
    import prepare_point_rotation_prior as inputs
    from campaign_hypothesis_batch import resource_sample,resource_capacity
except ImportError:
    from experiments import prepare_point_rotation_prior as inputs
    from experiments.campaign_hypothesis_batch import resource_sample,resource_capacity

worker=inputs.worker;ROOT=inputs.ROOT


def read(path):
    p=Path(path)
    if 'private-truth' in p.parts or p.suffix in ('.npz','.npy','.pt'):
        raise ValueError('Controller may read metadata only, never private truth or response arrays')
    return json.loads(p.read_text())


def eligible_launches(records,*,stopped,live_count,capacity):
    """Pure admission rule: failure stops NEW work; existing workers drain."""
    if stopped:return []
    return [row for row in records if row['state']=='queued'][:max(0,capacity-live_count)]


def atomic(path,value):
    temp=Path(str(path)+'.tmp');inputs.write(temp,value);temp.replace(path)


def validate_plan(path,digest):
    plan=read(worker.checked(dict(path=path,sha256=digest)))
    if (plan.get('schema')!=inputs.SCHEMA or plan.get('cohort_version')!='point-rotation-prior-v02'
            or plan.get('status')!='ALL_INPUTS_FROZEN_NO_FINITE_LAUNCH'):raise ValueError('Exact frozen cohort required')
    if plan['source_sha256']!=inputs.sources():raise ValueError('Frozen source map changed')
    expected=inputs.identities()
    if len(plan['objects'])!=48 or len(plan['cells'])!=96:raise ValueError('Retain all48objects/96cells')
    for actual,identity in zip(plan['objects'],expected):
        if any(actual[k]!=v for k,v in identity.items()):raise ValueError('Seed/object/role assignment changed')
        worker.checked(actual['realized_hypothesis']);worker.checked(actual['retained_state']) # hashes only
    canonical=[]
    for obj in plan['objects']:
        for label,program in zip(('A','B'),inputs.PROGRAM_IDS):
            canonical.append((obj['object_id']+'__'+label,obj,program))
    for row,(cell_id,obj,program) in zip(plan['cells'],canonical):
        protocol=worker.validate_protocol(row['protocol']['path'],row['protocol']['sha256'])
        if (row['id']!=cell_id or row['object_id']!=obj['object_id'] or row['role']!=obj['role'] or row['program_id']!=program
                or row['output']!=plan['output_prefix']+'/cells/'+cell_id
                or protocol['cell_id']!=cell_id or protocol['evaluation_role']!=row['role'] or protocol['program_id']!=program
                or protocol['output']!=row['output'] or protocol['integration_hz']!=120
                or protocol['settings']!=worker.SETTINGS or protocol['public_hardware']!=plan['public_hardware']
                or protocol['realized_hypothesis']!=obj['realized_hypothesis'] or protocol['input_design']!=plan['public_manifest']):
            raise ValueError('Canonical cell/role/hardware/truth/clock differs')
    if plan['roles']!={'CALIBRATION':list(range(16)),'RESERVED_EVALUATION':list(range(16,48))}:raise ValueError('Role barrier changed')
    if plan['resources']!=dict(maximum_workers=8,threads_per_worker=2,gpu_used=False,memory_reserve_GiB=8,memory_per_worker_GiB=1):raise ValueError('Resource scope changed')
    public=read(worker.checked(plan['public_manifest']))
    if public.get('true_material_or_support_location_present') is not False:raise ValueError('Public information role changed')
    for row in plan['cells']:
        protocol=read(worker.checked(row['protocol']))
        if protocol['known_commands']!=public['known_commands'][row['program_id']]:raise ValueError('Commands must be identical across truth/roles')
    return plan


def completed(row):
    """Metadata and digest verification only, including RESERVED_EVALUATION."""
    directory=worker.resolve(row['output']);execution=read(directory/'execution.json');summary=read(directory/'summary.json')
    if (not execution.get('terminal') or not execution.get('completed') or not summary.get('completed')
            or Path('/proc',str(execution['pid'])).exists() or (directory/'failure.json').exists()
            or summary['frames']!=61 or summary['steps']!=360):raise ValueError('Worker terminal capture contract failed')
    if read(directory/'protocol.json')!=read(worker.checked(row['protocol'])):raise ValueError('Capture protocol differs')
    for name,key in [('trajectory.npz','trajectory_sha256'),('steps.jsonl','steps_sha256')]:
        if worker.sha(directory/name)!=summary[key]:raise ValueError('Terminal artifact digest differs')
    return True


def poll_finished(live,stopped):
    for pid,(process,row,log) in list(live.items()):
        if (worker.resolve(row['output'])/'failure.json').exists():stopped=True
        code=process.poll()
        if code is None:continue
        process.wait();log.close();del live[pid]
        try:
            if code!=0 or not completed(row):raise ValueError('Worker unsuccessful')
            row.update(state='completed',completed=True,exit_code=code,terminal_pid_missing=not Path('/proc',str(pid)).exists())
        except BaseException as error:
            stopped=True;row.update(state='failed',completed=False,exit_code=code,error=str(error))
    return stopped


def run(plan_path,plan_sha):
    plan=validate_plan(plan_path,plan_sha);prefix=worker.resolve(plan['output_prefix'])
    # Entire run is one-shot: no reconstruction from partial index, resume or retries.
    if prefix.exists():raise ValueError('Fresh response prefix required; prior failures/queued slots preserved, never retry')
    prefix.mkdir(parents=True);logs=prefix/'worker-logs';logs.mkdir(mode=0o700)
    records=[dict(**row,state='queued',completed=False) for row in plan['cells']]
    index=dict(schema='point-rotation-cohort-result-index-v0.2',plan=dict(path=plan_path,sha256=plan_sha),records=records,
        pid=os.getpid(),terminal=False,completed=False,source_sha256=plan['source_sha256'],resource_samples=[])
    path=prefix/'result-index.json';atomic(path,index);live={};stopped=False;error=None
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':''}
    try:
        while True:
            stopped=poll_finished(live,stopped)
            sample=resource_sample();sample['time_unix_s']=time.time();index['resource_samples'].append(sample)
            index['resource_samples']=index['resource_samples'][-512:];cap=resource_capacity(sample,plan['resources'])
            for row in eligible_launches(records,stopped=stopped,live_count=len(live),capacity=cap):
                if inputs.sources()!=plan['source_sha256']:raise ValueError('Frozen sources changed before launch')
                worker.validate_protocol(row['protocol']['path'],row['protocol']['sha256'])
                if worker.resolve(row['output']).exists():raise ValueError('Existing cell output: never retry')
                log=(logs/(row['id']+'.log')).open('x');os.chmod(log.name,0o600)
                command=[sys.executable,str(worker.SOURCE),'--protocol',row['protocol']['path'],'--protocol-sha256',row['protocol']['sha256']]
                try:process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                except BaseException:log.close();raise
                row.update(state='running',pid=process.pid);live[process.pid]=(process,row,log)
            index.update(stopped_on_failure=stopped,live_count=len(live),completed_count=sum(r['completed'] for r in records));atomic(path,index)
            if not live and (stopped or all(r['completed'] for r in records)):break
            if not live and cap==0:raise RuntimeError('Insufficient memory: all queued records preserved')
            time.sleep(2)
    except BaseException as exc:
        stopped=True;error=repr(exc)
    finally:
        # Drain only owned workers; never kill unrelated jobs or drop queued records.
        for process,row,log in list(live.values()):
            code=process.wait();log.close()
            try:
                if code!=0 or not completed(row):raise ValueError('Worker terminal failure while draining')
                row.update(state='completed',completed=True,exit_code=code)
            except BaseException as exc:row.update(state='failed',completed=False,exit_code=code,error=str(exc))
        index.update(terminal=True,completed=all(r['completed'] for r in records),live_count=0,
            completed_count=sum(r['completed'] for r in records),stopped_on_failure=stopped,controller_error=error)
        atomic(path,index)
    return index


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();result=run(args.plan,args.plan_sha256)
    print(json.dumps({k:result[k] for k in ['terminal','completed','completed_count','stopped_on_failure']}))
    if not result['completed']:raise SystemExit(1)
