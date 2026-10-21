"""Durable, finite metadata-gated CPU-only postprocessing; no GPU job ownership.

One attempt per stage. Waits inspect only terminal execution metadata. Existing
frozen child validators own all prediction/truth gates. Raw outputs stay local.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def resolve(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=str(name):raise ValueError('Canonical repository-relative path required')
    result=(ROOT/p).resolve()
    if not result.is_relative_to(ROOT):raise ValueError('Repository escape')
    return result


def checked(rec):
    p=resolve(rec['path'])
    if sha(p)!=rec['sha256']:raise ValueError('Pinned postprocess artifact changed: '+rec['path'])
    return p


def record(path):return dict(path=Path(path).resolve().relative_to(ROOT).as_posix(),sha256=sha(path))

def write_atomic(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def validate(protocol_path,digest):
    p=read(checked(dict(path=protocol_path,sha256=digest)))
    if (p.get('schema')!='point-rotation-durable-postprocess-v0.2' or p.get('wall_limit_s')!=3600
        or p.get('poll_interval_s')!=10 or p.get('CPU_threads')!=2 or p.get('GPU_jobs') is not False
        or len(p['primary_source_sha256'])!=48):raise ValueError('Exact finite CPU-only controller declaration required')
    own=Path(__file__).relative_to(ROOT).as_posix()
    if p['source_sha256'].get(own)!=sha(__file__):raise ValueError('Controller source must be frozen before launch')
    for mapping in [p['primary_source_sha256'],p['source_sha256']]:
        for name,value in mapping.items():checked(dict(path=name,sha256=value))
    for rec in p['artifacts'].values():checked(rec)
    execution=read(checked(p['artifacts']['primary_execution']))
    if execution['source_sha256']!=p['primary_source_sha256']:raise ValueError('All48 primary source pins must match')
    return p


def metadata_complete(value,kind,p):
    indices=list(range(16,48))
    if value.get('terminal') is not True:return False
    if value.get('completed') is not True or value.get('failures')!=[]:raise ValueError(kind+' terminal failure retained; no retry')
    if kind=='primary':
        if (value.get('role')!='RESERVED_EVALUATION' or value.get('planned_object_indices')!=indices
            or value.get('completed_object_indices')!=indices or value.get('unlaunched_object_indices')!=[]
            or value.get('method_sha256')!=p['artifacts']['method']['sha256']
            or value.get('execution_protocol_sha256')!=p['artifacts']['primary_execution']['sha256']):raise ValueError('Exact32 primary terminal metadata required')
    elif kind=='auxiliary':
        if (value.get('planned_indices')!=indices or value.get('completed_indices')!=indices or value.get('queued_indices')!=[]
            or value.get('protocol')!=p['artifacts']['benchmark_protocol'] or value.get('release')!=p['artifacts']['release']):raise ValueError('Exact32 auxiliary terminal metadata required')
    else:raise ValueError('Unknown metadata gate')
    return True


def run(protocol_path,digest):
    p=validate(protocol_path,digest);destination=resolve(p['controller_output']);destination.mkdir(parents=True,exist_ok=True)
    lock=destination/'controller.lock';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
    status_path=destination/'status.json';started=time.monotonic();deadline=started+p['wall_limit_s']
    status=dict(schema='point-rotation-postprocess-status-v0.2',pid=os.getpid(),protocol=dict(path=protocol_path,sha256=digest),
        terminal=False,completed=False,stage='startup',stages=[],failures=[],CPU_threads=2,GPU_used=False,
        metadata_poll_read_errors=0,original_GPU_jobs_owned_or_reaped=False)
    def save():
        status['elapsed_s']=time.monotonic()-started;write_atomic(status_path,status)
    def remaining():
        left=deadline-time.monotonic()
        if left<=0:raise TimeoutError('One-hour finite postprocess wall limit reached')
        return left
    def wait_metadata(kind,path):
        status['stage']='wait_'+kind;save()
        while True:
            remaining()
            try:value=read(resolve(path))
            except FileNotFoundError:value=None
            except json.JSONDecodeError as error:
                # GPU writers replace JSON by truncate/write. This is a poll
                # race, never a failed stage retry; count and retain the error.
                status['metadata_poll_read_errors']+=1;status['last_metadata_poll_error']=str(error);value=None
            if value is not None:
                key='completed_object_indices' if kind=='primary' else 'completed_indices'
                status['last_'+kind+'_metadata']=dict(terminal=value.get('terminal'),completed=value.get('completed'),
                    completed_indices=value.get(key,[]),pid=value.get('pid'))
                if metadata_complete(value,kind,p):
                    status[kind+'_terminal_metadata']=record(resolve(path));save();return value
            save();time.sleep(min(p['poll_interval_s'],remaining()))
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1')
    python=str(ROOT/'.venv/bin/python');active=None
    def stage(name,args,artifact=None):
        nonlocal active
        validate(protocol_path,digest);remaining();status['stage']=name
        log=destination/(name+'.log');entry=dict(name=name,command=[python,*args],log=str(log.relative_to(ROOT)),started_elapsed_s=time.monotonic()-started)
        status['stages'].append(entry);save()
        with log.open('xb') as stream:
            active=subprocess.Popen([python,*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            entry['pid']=active.pid;status['active_child_pid']=active.pid;save()
            try:code=active.wait(timeout=remaining())
            except BaseException:
                os.killpg(active.pid,signal.SIGTERM)
                try:active.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
                raise
        active=None;status['active_child_pid']=None;entry.update(exit_code=code,log_sha256=sha(log),finished_elapsed_s=time.monotonic()-started)
        if code!=0:save();raise RuntimeError(name+' failed; see '+str(log.relative_to(ROOT)))
        if artifact is not None:
            path=resolve(artifact)
            if not path.exists():raise ValueError(name+' exit0 without expected output')
            entry['artifact']=record(path)
        save();return entry.get('artifact')
    a=p['artifacts']
    try:
        save();wait_metadata('primary',p['primary_status'])
        primary=stage('primary_aggregate',['-m','experiments.summarize_point_rotation_analysis','evaluate-reserved','--output',p['primary_output'],
            '--release',a['release']['path'],'--release-sha256',a['release']['sha256']],p['primary_output']+'/reserved-evaluation.json')
        value=read(checked(primary))
        if value.get('completed') is not True or value.get('objects')!=32 or value.get('schema')!='point-rotation-reserved-view-evaluation-v0.2':raise ValueError('Complete32 primary output contract failed')
        status['primary_sealed_evaluation']=primary;save()
        stage('reserved_material_audit',['-m','experiments.campaign_point_rotation_material_audit','run','--recipe',a['material_recipe']['path'],
            '--recipe-sha256',a['material_recipe']['sha256'],'--role','RESERVED_EVALUATION','--calibration',a['calibration']['path'],'--calibration-sha256',a['calibration']['sha256'],
            '--release',a['release']['path'],'--release-sha256',a['release']['sha256'],'--primary',primary['path'],'--primary-sha256',primary['sha256']],p['material_summary'])
        stage('reserved_videos',['-m','experiments.render_point_rotation_comparison','run','--recipe',a['video_recipe']['path'],'--recipe-sha256',a['video_recipe']['sha256'],
            '--evaluation',primary['path'],'--evaluation-sha256',primary['sha256']],p['video_manifest'])
        auxiliary=wait_metadata('auxiliary',p['auxiliary_status'])
        all_states=auxiliary.get('all_states_freeze')
        if all_states is None or checked(all_states)!=resolve(p['auxiliary_all_states']):raise ValueError('Auxiliary terminal must bind exact all-states freeze')
        freeze=read(checked(all_states))
        if (freeze.get('completed') is not True or freeze.get('object_indices')!=list(range(16,48)) or freeze.get('protocol')!=a['benchmark_protocol']
            or freeze.get('release')!=a['release']):raise ValueError('All32 auxiliary states must be frozen')
        stage('matched_compare_help',['-m','experiments.run_point_old_matched_benchmark','compare','--help'])
        stage('matched_comparison',['-m','experiments.run_point_old_matched_benchmark','compare','--protocol',a['benchmark_protocol']['path'],'--protocol-sha256',a['benchmark_protocol']['sha256'],
            '--all-states',all_states['path'],'--all-states-sha256',all_states['sha256'],'--primary-evaluation',primary['path'],'--primary-evaluation-sha256',primary['sha256']],p['matched_comparison'])
        validate(protocol_path,digest);status.update(terminal=True,completed=True,stage='completed');save();return status
    except BaseException as exc:
        status['failures'].append(dict(stage=status['stage'],error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(limit=8)))
        status.update(terminal=True,completed=False);save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);parser.add_argument('--protocol-sha256',required=True)
    args=parser.parse_args();result=run(args.protocol,args.protocol_sha256);print(json.dumps(dict(completed=result['completed'],pid=result['pid'])))
