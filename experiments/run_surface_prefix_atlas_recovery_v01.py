"""Explicit five-point recovery of a failed supervisor; old outputs immutable.

Only104,105,106,108,112 are newly solved. All108 completed predecessor points
are reused by exact byte/context hashes. Physics/atlas gates are unchanged.
"""
import argparse,ast,json,os,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_surface_prefix_atlas_feasibility_v01 as old
ROOT=old.ROOT;LOCAL=ROOT/'runs/operator-dev/surface-prefix-atlas-recovery-v01'
SOURCE='experiments/run_surface_prefix_atlas_recovery_v01.py';TEST='tests/test_surface_prefix_atlas_recovery_v01.py'
OLD_PROTOCOL_SHA='3f9147741d0edaf0c32d48744b1fa5c0678cbb6f04f936c56e012a434bad32cf'
MISSING=[104,105,106,108,112]
LIMITS=dict(worker_CPU=17,controller_CPU=16,threads=1,GPU=False,worker_RSS_cap_bytes=4*1024**3,aggregate_RSS_cap_bytes=4*1024**3,worker_wall_cap_s=300,global_wall_cap_s=300,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=2*1024**3)
sha=old.sha;record=old.record;write=old.write;array_sha=old.array_sha


def sources():
    seen=set();pending=[SOURCE,TEST]
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:modules=[node.module or '']+[(node.module+'.'+v.name)for v in node.names if node.module]
            for module in modules:
                path=module.replace('.','/')+'.py'
                if (ROOT/path).is_file()and path not in seen:pending.append(path)
    return {name:sha(ROOT/name)for name in sorted(seen)}


def predecessor_gate():
    p=old.validate(OLD_PROTOCOL_SHA);base=old.LOCAL;e=json.loads((base/'execution.json').read_text())
    if not e.get('terminal')or e.get('completed')or e.get('failure',{}).get('type')!='StopIteration'or len(e['workers'])!=8:raise ValueError('Exact stopped supervisory failure prerequisite')
    old.absent(e['controller_pid'])
    if sorted(w['slot']for w in e['workers'])!=list(range(8))or len({w['pid']for w in e['workers']})!=8:raise ValueError('Eight unique predecessor slots/PIDs')
    paths=[base/name for name in('protocol.json','execution.json','release.json')];completed={}
    for w in e['workers']:
        if not w['reaped']or type(w['exit_code'])is not int:raise ValueError('All interrupted/successful children must be reaped')
        old.absent(w['pid']);directory=base/f'worker-{w["slot"]}';summary=directory/'summary.json';paths.append(summary);a=json.loads(summary.read_text())
        if a['slot']!=w['slot']or a['pid']!=w['pid']or a['protocol_SHA256']!=OLD_PROTOCOL_SHA:raise ValueError('Old worker ownership/context')
        for row in a['points']:
            index=row['index'];metadata=directory/f'point-{index:03d}.json';paths.append(metadata)
            if json.loads(metadata.read_text())!=row or index%8!=w['slot']or row['protocol_SHA256']!=OLD_PROTOCOL_SHA or row['unit_SHA256']!=array_sha(p['points_unit_LOCAL'][index]):raise ValueError('Original point record/unit/context mapping')
            if row['completed']and row['passed']:
                raw=ROOT/row['raw']['path']
                if sha(raw)!=row['raw']['sha256']or index in completed:raise ValueError('Original rawSHA/unique point')
                paths.append(raw);completed[index]=row
    if sorted(set(range(113))-set(completed))!=MISSING or len(completed)!=108:raise ValueError('Exactly108 reused and5 missing; no implicit retries')
    return p,completed,{str(path.relative_to(ROOT)):sha(path)for path in paths}


def guard(role):
    if role not in('worker','controller')or set(os.sched_getaffinity(0))!={LIMITS[role+'_CPU']}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact role/CPU/threads/noGPU guard')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>LIMITS['worker_RSS_cap_bytes']:raise MemoryError('Own4GiB RSS')
    available=int(next(v for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:')).split()[1])*1024
    if available<LIMITS['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<LIMITS['disk_reserve_bytes']:raise MemoryError('30GiB RAM/disk reserves')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>LIMITS['maximum_writes_bytes']:raise OSError('Own2GiB writes cap')


def owned_RSS(child,*,read_status=None):
    """Only a reaped/exiting child may have absent proc/RSS; others fail closed."""
    read_status=(lambda:Path('/proc',str(child.pid),'status').read_text())if read_status is None else read_status
    try:
        status=read_status();rss=next(v for v in status.splitlines()if v.startswith('VmRSS:'))
        return int(rss.split()[1])*1024
    except(FileNotFoundError,StopIteration):
        if child.poll()is not None:return 0
        if 'status'in locals():
            state=next((v.split()[1]for v in status.splitlines()if v.startswith('State:')),None)
            if state in('Z','X'):return 0
        raise RuntimeError('Live owned process has unresolved RSS; fail closed')


def prepare():
    guard('controller');_,_,origin=predecessor_gate()
    p=dict(schema='surface-prefix-atlas-explicit-recovery-v01',status='FROZEN_ROOT_RELEASE_REQUIRED',source_SHA256=sources(),resources=LIMITS,origin_SHA256=origin,original_protocol_SHA256=OLD_PROTOCOL_SHA,new_point_indices=MISSING,reused_point_count=108,total_points=113,physics='Identical frozen old.solve_point,80bulkquadrature,originalallphysicalgates',validation='Identical frozen old.atlas_checks; all3levels,32offgrid,12likelihoodcases',implicit_retries=False,old_namespace_writes=False)
    if (LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text());oldp,rows,origin=predecessor_gate()
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=LIMITS or p['origin_SHA256']!=origin or p['original_protocol_SHA256']!=OLD_PROTOCOL_SHA or p['new_point_indices']!=MISSING or p['reused_point_count']!=108:raise ValueError('Exact continuation/source/origin/resources')
    return p,oldp,rows


def worker(digest):
    guard('worker');p,oldp,_=validate(digest);directory=LOCAL/'new-points-LOCAL'
    if directory.exists():raise FileExistsError('No recovery retry')
    directory.mkdir();start=time.monotonic();a=dict(completed=False,passed=False,worker_pid=os.getpid(),protocol_SHA256=digest,points=[])
    def save():a.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'worker-result-LOCAL.json',a)
    def deadline(*_):raise TimeoutError('Actual300s recovery deadline')
    def terminated(*_):raise InterruptedError('Explicit supervisor SIGTERM cleanup')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,terminated);signal.alarm(300)
    try:
        from experiments.campaign_surface_p2_static_v02 import SparseStaticForward,point_query_matrix
        from experiments.campaign_surface_p2_force_v03 import load_catalog
        from experiments.campaign_surface_p2_study_design_v01 import observation_points
        fwd=SparseStaticForward(4);force=load_catalog(4)['normal_x180'];Q=point_query_matrix(fwd,observation_points());save()
        for index in MISSING:
            guard('worker');begin=time.monotonic();row=dict(index=index,completed=False,passed=False,protocol_SHA256=digest,original_protocol_SHA256=OLD_PROTOCOL_SHA,unit_SHA256=array_sha(oldp['points_unit_LOCAL'][index]))
            try:
                arrays,diag=old.solve_point(fwd,force,Q,oldp['points_unit_LOCAL'][index]);guard('worker');target=directory/f'point-{index:03d}-LOCAL.npz';np.savez_compressed(target,**arrays);row.update(raw=record(target),diagnostics=diag);guard('worker');row.update(completed=True,passed=diag['passed'])
            except Exception as error:
                row['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
                if isinstance(error,(MemoryError,TimeoutError,InterruptedError)):raise
            finally:row['wall_s']=time.monotonic()-begin;a['points'].append(row);write(directory/f'point-{index:03d}.json',row);save()
        a.update(completed=True,passed=all(r['passed']for r in a['points']));save()
    except BaseException as error:a['failure']=dict(type=type(error).__name__,message=str(error));save();raise
    finally:signal.alarm(0);save()
    return a


def summarize(digest):
    p,oldp,rows=validate(digest);a=json.loads((LOCAL/'worker-result-LOCAL.json').read_text())
    if not a['completed']or a['protocol_SHA256']!=digest or [r['index']for r in a['points']]!=MISSING:raise ValueError('Five explicit terminal recovery records')
    for row in a['points']:
        metadata=LOCAL/'new-points-LOCAL'/f'point-{row["index"]:03d}.json'
        if row.get('protocol_SHA256')!=digest or row.get('original_protocol_SHA256')!=OLD_PROTOCOL_SHA or json.loads(metadata.read_text())!=row or (row['passed']and not row['completed']):raise ValueError('Exact new point metadata/protocol/completion context')
        if row['unit_SHA256']!=array_sha(oldp['points_unit_LOCAL'][row['index']]):raise ValueError('New point unit context')
        rows[row['index']]=row
    result=dict(completed=True,qualified=False,reused_points=108,new_points=5,all113_retained=True,physical_passed=sum(r['passed']for r in rows.values()),protocol_SHA256=digest,original_failed_protocol_SHA256=OLD_PROTOCOL_SHA,original_failure_preserved=True)
    if result['physical_passed']==113:
        raw=[]
        for index in range(113):
            row=rows[index];path=ROOT/row['raw']['path']
            if sha(path)!=row['raw']['sha256']:raise ValueError('Every old/new raw SHA')
            with np.load(path,allow_pickle=False)as z:
                if not np.array_equal(z['unit'],oldp['points_unit_LOCAL'][index]):raise ValueError('Raw original coordinates')
                raw.append((z['delta_m'].copy(),z['linear_response_m'].copy(),z['response_m'].copy()))
        check,atlas=old.atlas_checks(json.loads(old.DESIGN.read_text()),np.array([r[0]for r in raw[:81]]).reshape(9,9,1,24),np.array([r[1]for r in raw[81:]]),np.array([r[2]for r in raw[81:]]),fingerprint=digest)
        target=LOCAL/'prefix-atlas-LOCAL.npz'
        if target.exists():raise FileExistsError('No atlas overwrite')
        np.savez_compressed(target,delta=atlas.delta,axis=atlas.axis);result.update(qualified=check['passed'],validation=check,atlas=record(target),atlas_fingerprint=atlas.fingerprint)
    write(LOCAL/'summary-LOCAL.json',result);return result


def worker_command(digest,*,guard_only=False):
    return ['taskset','-c','17',sys.executable,'-u','-m','experiments.run_surface_prefix_atlas_recovery_v01','worker-guard'if guard_only else'worker','--sha',digest]


def run(digest):
    guard('controller');p,_,_=validate(digest);release=json.loads((LOCAL/'release.json').read_text())
    if release.get('approved')is not True or release.get('protocol_SHA256')!=digest or release.get('source_SHA256')!=p['source_SHA256']:raise ValueError('Concrete root-approved release')
    if (LOCAL/'execution.json').exists()or (LOCAL/'new-points-LOCAL').exists():raise FileExistsError('One explicit recovery run only')
    start=time.monotonic();child=None;a=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'))
    def stopped(*_):raise InterruptedError('Explicit controller stop')
    handlers={sig:signal.signal(sig,stopped)for sig in(signal.SIGTERM,signal.SIGINT)}
    try:
        with (LOCAL/'worker.log').open('xb')as log:
            child=subprocess.Popen(worker_command(digest),cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);a['worker_pid']=child.pid;write(LOCAL/'execution.json',a)
            while child.poll()is None:
                guard('controller');rss=owned_RSS(child)+resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                a['maximum_observed_aggregate_RSS_bytes']=max(a.get('maximum_observed_aggregate_RSS_bytes',0),rss)
                if rss>LIMITS['aggregate_RSS_cap_bytes']or time.monotonic()-start>300:raise TimeoutError('Independent owned aggregate RSS/300s cap')
                time.sleep(.1)
            a['worker_exit_code']=child.wait()
            if child.returncode!=0:raise RuntimeError('Recovery worker abnormal exit')
            old.absent(child.pid);result=summarize(digest);a.update(completed=True,qualified=result['qualified'],summary=record(LOCAL/'summary-LOCAL.json'))
    except BaseException as error:a['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        if child is not None:a['cleanup_exit_code']=old.cost.cleanup(child);old.absent(child.pid)
        for sig,handler in handlers.items():signal.signal(sig,handler)
        a.update(terminal=True,wall_s=time.monotonic()-start);write(LOCAL/'execution.json',a)
    return a


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','run','worker-guard','controller-guard']);parser.add_argument('--sha');args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare()))
    elif args.action=='worker':a=worker(args.sha);print(json.dumps(dict(completed=a['completed'],passed=a['passed'])))
    elif args.action=='run':a=run(args.sha);print(json.dumps(dict(completed=a['completed'],qualified=a.get('qualified',False))));raise SystemExit(0 if a['completed'] else 1)
    elif args.action=='worker-guard':guard('worker');print('guard passed')
    else:guard('controller');print('guard passed')
