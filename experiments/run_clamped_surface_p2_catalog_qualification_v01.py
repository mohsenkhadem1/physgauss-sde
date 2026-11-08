"""Separate all-load P2 physics admission; central qualification is mandatory.

No inference. The frozen central metric and mechanics are reused unchanged.
Every one of five prior cases and ten catalog loads remains in the denominator.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments import run_clamped_surface_p2_qualification_v01 as central
from experiments import campaign_surface_p2_study_design_v01 as study

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'runs/operator-dev/clamped-surface-p2-catalog-qualification-v01'
PUBLIC=ROOT/'reports/clamped-surface-p2-catalog-qualification-v01'
CASES=copy.deepcopy(central.CASES);GATES=copy.deepcopy(central.GATES)
RESOURCES=dict(worker_CPUs=[4,5,6,7],controller_CPU=7,threads_per_worker=1,workers=4,GPU=False,worker_wall_cap_s=1200,global_wall_cap_s=1800,worker_RSS_cap_bytes=4*1024**3,aggregate_RSS_cap_bytes=16*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=2*1024**3)
CENTRAL_PROTOCOL_SHA256='12bb39a9f79716913f4d7884a48b5bb2f855a7d27eda86e89223ad07a97aa4f8'
CENTRAL_ALIASES={'normal_x180':'normal','paired_axial_x180':'paired_axial'}
sha=central.sha;write=central.write


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float64).tobytes()).hexdigest()


def central_gate():
    base=central.LOCAL;p=json.loads((base/'protocol.json').read_text());s=json.loads((base/'summary.json').read_text());e=json.loads((base/'execution.json').read_text())
    if sha(base/'protocol.json')!=CENTRAL_PROTOCOL_SHA256 or p['source_sha256']!=central.sources():raise ValueError('Frozen central source/protocol mismatch')
    if e['status']!='COMPLETED' or not e['completed'] or not s['all10_retained'] or not s['all_qualified'] or s['qualified_cases']!=10:raise ValueError('All ten central groups must qualify before catalog')
    try:os.kill(e['pid'],0)
    except ProcessLookupError:pass
    else:raise ValueError('Central owned PID still live')
    paths=[base/name for name in ('protocol.json','summary.json','execution.json','publication-inventory-completed.json')]
    for c in CASES:
        for old in CENTRAL_ALIASES.values():
            paths.append(base/f'case-{c["index"]:02d}-{old}-summary.json')
            for n in (1,2,4):paths.append(base/f'case-{c["index"]:02d}-{old}-n{n}-LOCAL.npz')
    return {str(p.relative_to(ROOT)):sha(p)for p in paths}


def sources():
    pending=['experiments/run_clamped_surface_p2_catalog_qualification_v01.py','tests/test_clamped_surface_p2_catalog_qualification_v01.py'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=ROOT/name
        if not path.is_file():raise ValueError('Missing source '+name)
        seen.add(name);tree=ast.parse(path.read_text());package=name.rsplit('/',1)[0].replace('/','.')
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            if isinstance(node,ast.ImportFrom):
                parent=package.split('.');prefix='.'.join(parent[:len(parent)-node.level+1])if node.level else '';base='.'.join(x for x in(prefix,node.module or '')if x);modules=[base]+[base+'.'+x.name for x in node.names]
            for mod in modules:
                rel=mod.replace('.','/')+'.py'
                if (ROOT/rel).is_file()and rel not in seen:pending.append(rel)
    return {p:sha(ROOT/p)for p in sorted(seen)}


def guard(cpu=None):
    allowed=set(RESOURCES['worker_CPUs'])if cpu is None else {cpu}
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))or not set(os.sched_getaffinity(0))<=allowed:raise ValueError('Frozen CPU/one-thread/noGPU guard')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES['worker_RSS_cap_bytes']:raise central.ResourceStop('RSS cap exceeded')
    import shutil
    if shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise central.ResourceStop('Disk reserve failed')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise central.ResourceStop('Output budget exceeded')


def force_pins():
    return {str(n):{name:array_sha(f)for name,f in study.load_catalog(n).items()}for n in (1,2,4)}


def verify_central_loads():
    for n in (1,2,4):
        catalog=study.load_catalog(n);old=central.linear.qualification_loads(n)
        for name,alias in CENTRAL_ALIASES.items():
            if not np.array_equal(catalog[name],old[alias]):raise ValueError('Central load byte mismatch')
    return True


def prepare():
    guard();pins=central_gate();verify_central_loads()
    if (LOCAL/'protocol.json').exists():raise ValueError('No refreeze/retry')
    names=list(study.load_catalog(1))
    p=dict(schema='clamped-surface-p2-catalog-qualification-v01',cases=CASES,loads=names,levels=[1,2,4],gates=GATES,resources=RESOURCES,source_sha256=sources(),central_inputs_sha256=pins,force_bytes_sha256=force_pins(),central_aliases=CENTRAL_ALIASES,expected_groups=50,reused_central_groups=10,new_groups=40,new_paired_mesh_solves=120,new_equilibria=360,study_design=study.design(),metric='Exact frozen central metric functions/source hashes; common finest positiveKeast24 coordinate RMS',all50_retained=True,no_inference=True,no_retuning=True)
    write(LOCAL/'protocol.json',p);return dict(protocol=str(LOCAL/'protocol.json'),sha256=sha(LOCAL/'protocol.json'))


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['source_sha256']!=sources()or p['central_inputs_sha256']!=central_gate()or p['force_bytes_sha256']!=force_pins():raise ValueError('Frozen protocol/source/input/force mismatch')
    verify_central_loads();return p


def case_path(index,name):return LOCAL/f'case-{index:02d}-{name}'


def solve_group(case,name):
    start=time.monotonic();fields={};record=dict(case_index=case['index'],load=name,levels={},completed=True,reused_central=False);E=case['E_Pa'];K=case['K_N_m'];base=case_path(case['index'],name);base.mkdir(parents=True,exist_ok=False)
    for n in (1,2,4):
        guard();fwd=central.linear.SparseStaticForward(n);force=study.load_catalog(n)[name];row={};fields[n]={}
        try:
            u=fwd.staticforward(np.array([[E,K]]),force)[0,0];fields[n]['linear']=u
            row['linear']=dict(deformation=central.deformation(fwd,u),equilibrium=central.reaction_diagnostics(fwd,u,force,E,K))
            for order,key in ((24,'nonlinear'),(80,'NH80_reference')):
                model=central.linear.P2NonlinearReference.from_forward(fwd,E,K,quadrature_order=order);un,diag=model.solve(force);state=model.evaluate(un,force,tangent=True)
                try:stability=central.stability_diagnostic(state['tangent'][model.free][:,model.free],n)
                except Exception as exc:stability=dict(completed=False,passed=False,error=type(exc).__name__+': '+str(exc))
                row[key]=dict(deformation=central.deformation(fwd,un),equilibrium=central.reaction_diagnostics(fwd,un,force,E,K,state['residual']),Newton=diag,stability=stability)
                fields[n][key]=un
            un=fields[n]['nonlinear'];un80=fields[n]['NH80_reference']
            row['NH24_vs_NH80_RMS_m']=float(central.common_rms([(n,un),(n,un80)])[1,0]);row['NH24_vs_NH80_passed']=row['NH24_vs_NH80_RMS_m']<=GATES['maximum_NH24_vs_NH80_RMS_m']
            row['linear_vs_nonlinear_RMS_m']=float(central.common_rms([(n,u),(n,un)])[1,0]);row['linear_vs_nonlinear_passed']=row['linear_vs_nonlinear_RMS_m']<=GATES['maximum_linear_vs_nonlinear_RMS_m']
            np.savez_compressed(base/f'n{n}-LOCAL.npz',linear=u,nonlinear=un,nonlinear80=un80)
        except Exception as exc:row['completed']=False;row['error']=type(exc).__name__+': '+str(exc);record['completed']=False
        record['levels'][str(n)]=row;guard()
    record['spatial']={}
    for law in ('linear','nonlinear'):
        if all(law in fields[n]for n in (1,2,4)):
            d=central.common_rms([(n,fields[n][law])for n in (1,2,4)]);record['spatial'][law]=central.spatial_gate(float(d[1,0]),float(d[2,1]))
        else:record['spatial'][law]=dict(spatial_passed=False,medium_adequate=False,error='Incomplete retained fields')
    record['passed']=bool(record['completed'] and all(all(row[law]['deformation']['physical_passed']and row[law]['equilibrium']['passed']for law in ('linear','nonlinear','NH80_reference'))and row['nonlinear']['stability']['passed']and row['NH80_reference']['stability']['passed']and row['linear_vs_nonlinear_passed']and row['NH24_vs_NH80_passed']for row in record['levels'].values())and all(d['spatial_passed']and d['medium_adequate']for d in record['spatial'].values()))
    record.update(worker_wall_s=time.monotonic()-start,worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(base/'summary.json',record);return record


def worker(digest,index,name,cpu):
    os.sched_setaffinity(0,{cpu});guard(cpu);p=validate(digest)
    if name not in p['loads']or name in CENTRAL_ALIASES or index not in range(5):raise ValueError('Only prospective new groups in worker')
    def timeout(*_):raise central.ResourceStop('Worker frozen wall cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(RESOURCES['worker_wall_cap_s'])
    return solve_group(CASES[index],name)


def run(digest):
    guard();p=validate(digest)
    if (LOCAL/'execution.json').exists():raise ValueError('No retry')
    start=time.monotonic();execution=dict(status='RUNNING',controller_pid=os.getpid(),protocol_sha256=digest,workers=[]);write(LOCAL/'execution.json',execution)
    # These checks compile forces/integration only and do not solve new cases.
    quadrature={str(n):central.quadrature_checks(central.linear.SparseStaticForward(n))for n in (1,2,4)}
    if not all(v['passed']for v in quadrature.values()):raise ValueError('Pinned production quadrature failed')
    results=[]
    for case in CASES:
        for name,alias in CENTRAL_ALIASES.items():
            old=json.loads((central.LOCAL/f'case-{case["index"]:02d}-{alias}-summary.json').read_text());old=copy.deepcopy(old);old.update(load=name,reused_central=True,reused_central_load=alias);results.append(old)
    pending=[(c['index'],name)for c in CASES for name in p['loads']if name not in CENTRAL_ALIASES];active={};logs=[]
    try:
        while pending or active:
            guard()
            if time.monotonic()-start>RESOURCES['global_wall_cap_s']:raise central.ResourceStop('Global wall cap')
            used={r['cpu']for r in active.values()}
            for cpu in RESOURCES['worker_CPUs']:
                if cpu in used or not pending:continue
                index,name=pending.pop(0);log=(LOCAL/f'worker-case{index}-{name}.log').open('wb');logs.append(log)
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--protocol-sha256',digest,'--case-index',str(index),'--load-name',name,'--cpu',str(cpu)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row=dict(pid=process.pid,cpu=cpu,case_index=index,load=name,reaped=False);active[process]=row;execution['workers'].append(row)
            for process,row in list(active.items()):
                code=process.poll()
                if code is None:continue
                row.update(reaped=True,exit_code=code);del active[process];path=case_path(row['case_index'],row['load'])/'summary.json'
                if code==0 and path.is_file():results.append(json.loads(path.read_text()))
                else:results.append(dict(case_index=row['case_index'],load=row['load'],completed=False,passed=False,technical_failure=True,worker_exit_code=code))
            # Linux current RSS; children have no descendants by design.
            rss=0
            for row in active.values():
                try:
                    status=Path(f'/proc/{row["pid"]}/status').read_text();line=next(l for l in status.splitlines()if l.startswith('VmRSS:'));value=int(line.split()[1])*1024;rss+=value
                    if value>RESOURCES['worker_RSS_cap_bytes']:raise central.ResourceStop('Worker RSS cap')
                except(FileNotFoundError,StopIteration):pass
            if rss>RESOURCES['aggregate_RSS_cap_bytes']:raise central.ResourceStop('Aggregate RSS cap')
            execution.update(completed_groups=len(results),pending_groups=len(pending),live_groups=len(active));write(LOCAL/'execution.json',execution);time.sleep(.5)
    finally:
        for process,row in active.items():
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for process,row in active.items():
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            row.update(reaped=True,exit_code=process.returncode)
        for log in logs:log.close()
    results.sort(key=lambda r:(r['case_index'],p['loads'].index(r['load'])))
    expected={(c['index'],name)for c in CASES for name in p['loads']}
    if len(results)!=50 or {(r['case_index'],r['load'])for r in results}!=expected:raise ValueError('All50 denominator/mapping failed')
    summary=dict(completed=True,groups=50,prior_cases=5,loads=10,all50_retained=True,reused_central_groups=10,new_groups=40,qualified_groups=sum(r['passed']for r in results),all_qualified=all(r['passed']for r in results),quadrature=quadrature,per_case=results,protocol_sha256=digest,wall_s=time.monotonic()-start,no_inference_run=True,raw_arrays_LOCAL=True)
    write(LOCAL/'summary.json',summary);write(PUBLIC/'summary.json',summary);execution.update(status='COMPLETED',completed=True,all_owned_workers_reaped=True,qualified=summary['all_qualified']);write(LOCAL/'execution.json',execution);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker']);parser.add_argument('--protocol-sha256');parser.add_argument('--case-index',type=int);parser.add_argument('--load-name');parser.add_argument('--cpu',type=int);a=parser.parse_args()
    try:result=prepare()if a.mode=='prepare'else worker(a.protocol_sha256,a.case_index,a.load_name,a.cpu)if a.mode=='worker'else run(a.protocol_sha256)
    except BaseException as exc:
        if a.mode=='run':write(LOCAL/'execution-failed.json',dict(completed=False,status='FAILED',controller_pid=os.getpid(),error=type(exc).__name__+': '+str(exc)))
        raise
    print(json.dumps({k:v for k,v in result.items()if k not in ('per_case','quadrature','levels','spatial')}))
