"""Validation-only completion; zero new PDE/subprocess calls.

All108 original plus5 explicitly recovered passed fields are hash-bound.
Both supervisory negatives remain immutable, including their failure records.
"""
import argparse,ast,json,os,resource,signal,time
from pathlib import Path
import numpy as np
from experiments import run_surface_prefix_atlas_recovery_v01 as recovery
from experiments import run_surface_prefix_atlas_feasibility_v01 as prefix
ROOT=prefix.ROOT;LOCAL=ROOT/'runs/operator-dev/surface-prefix-atlas-validation-v01'
SOURCE='experiments/validate_surface_prefix_atlas_v01.py';TEST='tests/test_validate_surface_prefix_atlas_v01.py'
LIMITS=dict(CPU=17,threads=1,GPU=False,RSS_cap_bytes=4*1024**3,wall_cap_s=300,no_subprocess=True,no_PDE_calls=True)
sha=prefix.sha;record=prefix.record;write=prefix.write;array_sha=prefix.array_sha


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


def inputs():
    base=recovery.LOCAL;digest=sha(base/'protocol.json');p,oldp,rows=recovery.validate(digest)
    e=json.loads((base/'execution.json').read_text());w=json.loads((base/'worker-result-LOCAL.json').read_text())
    if not e.get('terminal')or e.get('completed')or e.get('cleanup_exit_code')!=0 or e.get('protocol_SHA256')!=digest or e.get('failure',{}).get('type')!='RuntimeError'or 'unresolved RSS'not in e['failure']['message']:raise ValueError('Exact retained recovery supervisory negative/exit0 prerequisite')
    prefix.absent(e['controller_pid']);prefix.absent(e['worker_pid'])
    if w.get('completed')is not True or w.get('passed')is not True or w['worker_pid']!=e['worker_pid']or w['protocol_SHA256']!=digest or [r['index']for r in w['points']]!=recovery.MISSING:raise ValueError('Five completed physical positives required')
    origin=dict(p['origin_SHA256']);paths=[base/n for n in('protocol.json','release.json','execution.json','worker-result-LOCAL.json')]
    for row in w['points']:
        i=row['index'];metadata=base/'new-points-LOCAL'/f'point-{i:03d}.json';raw=ROOT/row['raw']['path']
        if not row['completed']or not row['passed']or row['protocol_SHA256']!=digest or row['original_protocol_SHA256']!=recovery.OLD_PROTOCOL_SHA or row['unit_SHA256']!=array_sha(oldp['points_unit_LOCAL'][i])or json.loads(metadata.read_text())!=row or sha(raw)!=row['raw']['sha256']:raise ValueError('New recovery field/metadata/protocol context')
        rows[i]=row;paths.extend((metadata,raw))
    if sorted(rows)!=list(range(113))or not all(r['completed']and r['passed']for r in rows.values()):raise ValueError('All113 completed positives exactly once')
    for path in paths:origin[str(path.relative_to(ROOT))]=sha(path)
    return oldp,rows,origin


def guard():
    if set(os.sched_getaffinity(0))!={17}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('CPU17/one-thread/noGPU')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>LIMITS['RSS_cap_bytes']:raise MemoryError('4GiB RSS cap')


def prepare():
    guard();_,_,origin=inputs()
    if (LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    p=dict(schema='surface-prefix-atlas-validation-only-v01',status='FROZEN_ROOT_RELEASE_REQUIRED',source_SHA256=sources(),origin_SHA256=origin,resources=LIMITS,original_points=108,recovery_points=5,total_points=113,new_PDE_calls=0,subprocess_calls=0,unchanged_atlas_checks=True,original_failures_preserved=True)
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')


def evaluate_rows(oldp,rows,digest):
    """Only committed archives, then identical frozen finite validation."""
    raw=[]
    if sorted(rows)!=list(range(113)):raise ValueError('All113 positive archives required')
    for i in range(113):
        row=rows[i];path=ROOT/row['raw']['path']
        if not row['completed']or not row['passed']or sha(path)!=row['raw']['sha256']:raise ValueError('Passed/hash-bound raw only')
        with np.load(path,allow_pickle=False)as z:
            if not np.array_equal(z['unit'],oldp['points_unit_LOCAL'][i]):raise ValueError('Original exact point unit bytes')
            raw.append((z['delta_m'].copy(),z['linear_response_m'].copy(),z['response_m'].copy()))
    return prefix.atlas_checks(json.loads(prefix.DESIGN.read_text()),np.array([r[0]for r in raw[:81]]).reshape(9,9,1,24),np.array([r[1]for r in raw[81:]]),np.array([r[2]for r in raw[81:]]),fingerprint=digest)


def run(digest):
    guard();p=json.loads((LOCAL/'protocol.json').read_text());oldp,rows,origin=inputs()
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['origin_SHA256']!=origin or p['resources']!=LIMITS:raise ValueError('Exact source/input/protocol/resources')
    release=json.loads((LOCAL/'release.json').read_text())
    if release.get('approved')is not True or release.get('protocol_SHA256')!=digest or release.get('source_SHA256')!=p['source_SHA256']:raise ValueError('Concrete root release')
    if (LOCAL/'result-LOCAL.json').exists()or (LOCAL/'prefix-atlas-LOCAL.npz').exists():raise FileExistsError('One validation only')
    start=time.monotonic();result=dict(completed=False,qualified=False,pid=os.getpid(),protocol_SHA256=digest,new_PDE_calls=0,subprocess_calls=0,all113_retained=True)
    def deadline(*_):raise TimeoutError('300s validation deadline')
    previous=signal.signal(signal.SIGALRM,deadline);signal.alarm(300)
    try:
        check,atlas=evaluate_rows(oldp,rows,digest);guard();target=LOCAL/'prefix-atlas-LOCAL.npz';np.savez_compressed(target,delta=atlas.delta,axis=atlas.axis)
        result.update(completed=True,qualified=check['passed'],validation=check,atlas=record(target),atlas_fingerprint=atlas.fingerprint,original_failures_preserved=True)
    except BaseException as error:result['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous);result.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'result-LOCAL.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','guard']);parser.add_argument('--sha');a=parser.parse_args()
    if a.action=='prepare':print(json.dumps(prepare()))
    elif a.action=='run':r=run(a.sha);print(json.dumps(dict(completed=r['completed'],qualified=r['qualified'])))
    else:guard();print('guard passed')
