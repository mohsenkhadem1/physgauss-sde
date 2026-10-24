"""Prospective P2 resolution/load repair, preserving all prior failures.

Five prior cases × two loads × levels2/4/8. Cached unchanged normal n2/n4
fields are reused only after exact geometry/operator/load and archival parity.
No inference or catalog admission follows automatically from this run.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments import run_clamped_surface_p2_qualification_v01 as central
from experiments import campaign_surface_p2_static_v02 as linear

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'runs/operator-dev/clamped-surface-p2-qualification-v02'
PUBLIC=ROOT/'reports/clamped-surface-p2-qualification-v02'
CASES=copy.deepcopy(central.CASES);GATES=copy.deepcopy(central.GATES)
RESOURCES=dict(worker_CPUs=[4,5,6,7],controller_CPU=7,threads_per_worker=1,workers=4,GPU=False,worker_wall_cap_s=1800,global_wall_cap_s=3600,worker_RSS_cap_bytes=8*1024**3,aggregate_RSS_cap_bytes=32*1024**3,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=2*1024**3)
CENTRAL_PROTOCOL_SHA256='12bb39a9f79716913f4d7884a48b5bb2f855a7d27eda86e89223ad07a97aa4f8'
LOADS=['normal','paired_axial']
sha=central.sha;write=central.write


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float64).tobytes()).hexdigest()


def central_gate():
    base=central.LOCAL;p=json.loads((base/'protocol.json').read_text());s=json.loads((base/'summary.json').read_text());e=json.loads((base/'execution.json').read_text())
    if sha(base/'protocol.json')!=CENTRAL_PROTOCOL_SHA256 or p['source_sha256']!=central.sources():raise ValueError('Frozen central source/protocol mismatch')
    if e['status']!='COMPLETED' or not e['completed'] or not s['completed'] or not s['all10_retained'] or e['protocol_sha256']!=CENTRAL_PROTOCOL_SHA256 or s['protocol_sha256']!=CENTRAL_PROTOCOL_SHA256:raise ValueError('All ten predecessor groups must be retained')
    try:os.kill(e['pid'],0)
    except ProcessLookupError:pass
    else:raise ValueError('Central owned PID still live')
    inventory=base/'publication-inventory-completed.json'
    if sha(inventory)!='9f641ce5a9851cbc322e31a9ee107eb5e2ec22528dc5b4cf09bbc4b2d3f8d446':raise ValueError('Published predecessor inventory mismatch')
    for item in json.loads(inventory.read_text())['files']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Predecessor public artifact changed')
    ownership=json.loads((central.PUBLIC/'terminal-ownership.json').read_text())
    if sha(base/'execution.json')!=ownership['execution_sha256']or sha(base/'summary.json')!=ownership['summary_sha256']:raise ValueError('Predecessor terminal binding failed')
    paths=[base/name for name in ('protocol.json','summary.json','execution.json','publication-inventory-completed.json')]
    for c in CASES:
        for old in ['normal']:
            paths.append(base/f'case-{c["index"]:02d}-{old}-summary.json')
            for n in (2,4):
                archive=base/f'case-{c["index"]:02d}-{old}-n{n}-LOCAL.npz'
                if sha(archive)!=ownership['raw_fields_sha256'][archive.name]:raise ValueError('Predecessor cached field hash failed')
                paths.append(archive)
    return {str(p.relative_to(ROOT)):sha(p)for p in paths}


def sources():
    pending=['experiments/run_clamped_surface_p2_qualification_v02.py','tests/test_clamped_surface_p2_qualification_v02.py','tests/test_surface_p2_static_v02.py'];seen=set()
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
    available=int(next(l for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')).split()[1])*1024
    if available<RESOURCES['RAM_reserve_bytes']:raise central.ResourceStop('Physical RAM reserve failed')
    import shutil
    if shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise central.ResourceStop('Disk reserve failed')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise central.ResourceStop('Output budget exceeded')


def force_pins():
    return {str(n):{name:array_sha(f)for name,f in linear.qualification_loads(n).items()}for n in (2,4,8)}


def verify_normal_reuse():
    proof={}
    for name in ('evaluate','solve','elastic'):
        new=getattr(linear.P2NonlinearReference,name).__code__;old=getattr(central.linear.P2NonlinearReference,name).__code__
        if new.co_code!=old.co_code or new.co_consts!=old.co_consts:raise ValueError('Constitutive/solver kernel mismatch '+name)
    for n in (2,4):
        new=linear.SparseStaticForward(n);old=central.linear.SparseStaticForward(n)
        arrays={key:(getattr(new,key),getattr(old,key))for key in ('X','elements','fixed','free')}
        for key,(x,y)in arrays.items():
            if x.dtype!=y.dtype or x.shape!=y.shape or x.tobytes()!=y.tobytes():raise ValueError('Reuse geometry mismatch '+key)
        for key in ('Kunit','W','W_scalar'):
            for attr in ('data','indices','indptr'):
                x=getattr(getattr(new,key),attr);y=getattr(getattr(old,key),attr)
                if x.tobytes()!=y.tobytes():raise ValueError('Reuse operator mismatch '+key+attr)
        f=linear.qualification_loads(n)['normal'];original=central.linear.qualification_loads(n)['normal']
        if not np.array_equal(f,original):raise ValueError('Normal load byte mismatch')
        proof[str(n)]=dict(normal_load_sha256=array_sha(f),geometry_and_operators_byte_identical=True)
    return proof


def resource_projection():
    old=json.loads((central.LOCAL/'summary.json').read_text());benchmark=json.loads((LOCAL/'synthetic-cost.json').read_text())
    counts=[row[law]['Newton']['iterations']for case in old['per_case']for row in case['levels'].values()for law in ('nonlinear','NH80_reference')]
    maximum=max(counts);factor=benchmark['synthetic_LU_s'];tangents=sum(row['tangent_s']for row in benchmark['rows'])
    per_group=2*maximum*factor+(maximum+2)*tangents+120
    return dict(observed_predecessor_Newton_iteration_range=[min(counts),maximum],synthetic_n8_default_LU_s=factor,estimated_group_s=per_group,three_wave_estimate_s=3*per_group,assumption='Observed iteration maximum extrapolated;120s extra/coarse/eigensolve allowance. Hard caps, not runtime guarantee.',benchmark_sha256=sha(LOCAL/'synthetic-cost.json'),MMD_negative_benchmark_sha256=sha(LOCAL/'synthetic-MMD-cost.json'),default_LU_unchanged=True)


def prepare():
    guard();pins=central_gate();verify_normal_reuse()
    if (LOCAL/'protocol.json').exists():raise ValueError('No refreeze/retry')
    names=LOADS
    p=dict(schema='clamped-surface-p2-qualification-v02',resource_projection=resource_projection(),cases=CASES,loads=names,levels=[2,4,8],gates=GATES,resources=RESOURCES,source_sha256=sources(),central_inputs_sha256=pins,force_bytes_sha256=force_pins(),normal_reuse_proof=verify_normal_reuse(),expected_groups=10,reused_normal_mesh_cases=10,new_paired_mesh_solves=20,new_equilibria=60,target_mesh_n=4,physical_design=linear.design(),metric='Exact frozen central metric functions/source hashes; common finest positiveKeast24 coordinate RMS',all10_retained=True,design_tuned_on_v01=True,no_inference=True,no_retuning=True)
    write(LOCAL/'protocol.json',p);return dict(protocol=str(LOCAL/'protocol.json'),sha256=sha(LOCAL/'protocol.json'))


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['source_sha256']!=sources()or p['central_inputs_sha256']!=central_gate()or p['force_bytes_sha256']!=force_pins():raise ValueError('Frozen protocol/source/input/force mismatch')
    verify_normal_reuse();return p


def common_rms(fields):
    """Each tuple (meshlevel,nodalfield); evaluate every pair on same finest rule."""
    sums=np.zeros((len(fields),len(fields)));volume=0.
    for points,w in linear.bulk_query_blocks(8):
        values=[linear.field_queries(n,u,points)for n,u in fields];volume+=w.sum()
        for i in range(len(fields)):
            for j in range(i):sums[i,j]+=np.sum(w[:,None]*(values[i]-values[j])**2)
    return np.sqrt(sums/(3*volume))


def deformation(forward,u):
    # Rigid fit uses exact second moments of P1 fields via positive Keast24.
    points=[];weights=[];values=[]
    for q,w in linear.bulk_query_blocks(8):points.append(q);weights.append(w);values.append(linear.field_queries(forward.n,u,q))
    result=central.weighted_rigid_metrics(np.concatenate(points),np.concatenate(values),np.concatenate(weights))
    samples=linear.strain_metrics(forward,u)
    result.update({key:float(np.asarray(samples[key]).reshape(-1)[0])for key in ('minimum_J','maximum_absolute_symmetric_strain_eigenvalue')})
    result.update(checked_nodes_and24and80=True,whole_element_J_certified=False)
    result['physical_passed']=bool(result['minimum_J']>=GATES['minimum_J'] and result['maximum_absolute_symmetric_strain_eigenvalue']<=GATES['maximum_absolute_symmetric_strain_eigenvalue'] and result['nonrigid_RMS_m']>=GATES['minimum_nonrigid_RMS_m'] and result['nonrigid_fraction']>=GATES['minimum_nonrigid_fraction'])
    return result


def quadrature_checks(forward):
    rows={};matrices={}
    for order in (4,8,12,16,20):
        matrices[order],rows[str(order)]=linear.consistent_support(forward.X,forward.faces,order=order)
    norm=lambda A:float(__import__('scipy').sparse.linalg.norm(A))
    e812=norm(matrices[8]-matrices[12])/norm(matrices[12]);e1216=norm(matrices[12]-matrices[16])/norm(matrices[16]);e1620=norm(matrices[16]-matrices[20])/norm(matrices[20])
    loads=linear.qualification_loads(forward.n);load_rows={}
    for name,f in loads.items():
        expected=np.array([0,0,-.1])if name=='normal'else np.array([.36,0,0]);force_error=float(np.linalg.norm(f.sum(0)-expected));moment=float(np.linalg.norm(np.cross(forward.X-[.18,.04,.04],f).sum(0)))
        load_rows[name]=dict(net_force_error_N=force_error,reference_moment_error_Nm=moment,passed=bool(force_error<=linear.shared.GATES['load_net_force_error_N'] and moment<=linear.shared.GATES['load_moment_error_Nm']))
    passed=rows['16']['normalization_relative_error']<=linear.shared.GATES['support_normalization_relative_error'] and e1620<=GATES['maximum_surface_operator_relative_error'] and all(x['passed']for x in load_rows.values())
    return dict(passed=bool(passed),production_order=16,independent_order=20,orders=rows,relative_Frobenius_8_12=e812,relative_Frobenius_12_16=e1216,old12_vs16_control_passed=bool(e1216<=GATES['maximum_surface_operator_relative_error']),relative_Frobenius_16_20=e1620,loads=load_rows,preproduction_quadrature_refinement_only=True)


def case_path(index,name):return LOCAL/f'case-{index:02d}-{name}'


def solve_group(case,name):
    start=time.monotonic();fields={};record=dict(case_index=case['index'],load=name,levels={},completed=True,reused_central=False);E=case['E_Pa'];K=case['K_N_m'];base=case_path(case['index'],name);base.mkdir(parents=True,exist_ok=False)
    for n in (2,4,8):
        guard();fwd=linear.SparseStaticForward(n);force=linear.qualification_loads(n)[name];row={};fields[n]={}
        try:
            cached=name=='normal'and n in (2,4)
            if cached:
                archive=central.LOCAL/f'case-{case["index"]:02d}-normal-n{n}-LOCAL.npz'
                with np.load(archive,allow_pickle=False)as raw:
                    u=raw['linear'].copy();fields[n]['nonlinear']=raw['nonlinear'].copy();fields[n]['NH80_reference']=raw['nonlinear80'].copy()
                oldrow=json.loads((central.LOCAL/f'case-{case["index"]:02d}-normal-summary.json').read_text())['levels'][str(n)]
                row['normal_reuse']=dict(input_archive_sha256=sha(archive),new_force_solve=False)
            else:u=fwd.staticforward(np.array([[E,K]]),force)[0,0]
            fields[n]['linear']=u
            row['linear']=dict(deformation=deformation(fwd,u),equilibrium=central.reaction_diagnostics(fwd,u,force,E,K))
            for order,key in ((24,'nonlinear'),(80,'NH80_reference')):
                if cached:
                    un=fields[n][key];row[key]=copy.deepcopy(oldrow[key]);row[key]['deformation']=deformation(fwd,un)
                    continue
                model=linear.P2NonlinearReference.from_forward(fwd,E,K,quadrature_order=order);un,diag=model.solve(force);state=model.evaluate(un,force,tangent=True)
                try:stability=central.stability_diagnostic(state['tangent'][model.free][:,model.free],n)
                except Exception as exc:stability=dict(completed=False,passed=False,error=type(exc).__name__+': '+str(exc))
                row[key]=dict(deformation=deformation(fwd,un),equilibrium=central.reaction_diagnostics(fwd,un,force,E,K,state['residual']),Newton=diag,stability=stability)
                fields[n][key]=un
            un=fields[n]['nonlinear'];un80=fields[n]['NH80_reference']
            row['NH24_vs_NH80_RMS_m']=float(common_rms([(n,un),(n,un80)])[1,0]);row['NH24_vs_NH80_passed']=row['NH24_vs_NH80_RMS_m']<=GATES['maximum_NH24_vs_NH80_RMS_m']
            row['linear_vs_nonlinear_RMS_m']=float(common_rms([(n,u),(n,un)])[1,0]);row['linear_vs_nonlinear_passed']=row['linear_vs_nonlinear_RMS_m']<=GATES['maximum_linear_vs_nonlinear_RMS_m']
            np.savez_compressed(base/f'n{n}-LOCAL.npz',linear=u,nonlinear=un,nonlinear80=un80)
        except Exception as exc:row['completed']=False;row['error']=type(exc).__name__+': '+str(exc);record['completed']=False
        record['levels'][str(n)]=row;guard()
    record['spatial']={}
    for law in ('linear','nonlinear'):
        if all(law in fields[n]for n in (2,4,8)):
            d=common_rms([(n,fields[n][law])for n in (2,4,8)]);record['spatial'][law]=central.spatial_gate(float(d[1,0]),float(d[2,1]))
        else:record['spatial'][law]=dict(spatial_passed=False,medium_adequate=False,error='Incomplete retained fields')
    record['passed']=bool(record['completed'] and all(all(row[law]['deformation']['physical_passed']and row[law]['equilibrium']['passed']for law in ('linear','nonlinear','NH80_reference'))and row['nonlinear']['stability']['passed']and row['NH80_reference']['stability']['passed']and row['linear_vs_nonlinear_passed']and row['NH24_vs_NH80_passed']for row in record['levels'].values())and all(d['spatial_passed']and d['medium_adequate']for d in record['spatial'].values()))
    record.update(worker_wall_s=time.monotonic()-start,worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(base/'summary.json',record);return record


def worker(digest,index,name,cpu):
    os.sched_setaffinity(0,{cpu});guard(cpu);p=validate(digest)
    if name not in p['loads']or index not in range(5):raise ValueError('Only prospective groups in worker')
    def timeout(*_):raise central.ResourceStop('Worker frozen wall cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(RESOURCES['worker_wall_cap_s'])
    return solve_group(CASES[index],name)


def retained_groups(results,loads=LOADS):
    expected={(c['index'],name)for c in CASES for name in loads}
    if len(results)!=10 or {(r['case_index'],r['load'])for r in results}!=expected:raise ValueError('All10 denominator/mapping failed')
    return dict(all10_retained=True,qualified_groups=sum(bool(r['passed'])for r in results),all_qualified=all(bool(r['passed'])for r in results))


def check_worker_deadlines(active,now):
    for row in active.values():
        if now-row['started_monotonic']>RESOURCES['worker_wall_cap_s']:
            raise central.ResourceStop('Independent parent worker wall cap')


def cleanup_workers(active,execution):
    for process,row in active.items():
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    for process,row in active.items():
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
        row.update(reaped=True,exit_code=process.returncode)
    active.clear()
    execution.update(cleanup_completed=True,live_groups=0,all_owned_workers_reaped=all(row['reaped']for row in execution['workers']))
    write(LOCAL/'execution.json',execution)


def run(digest):
    os.sched_setaffinity(0,{RESOURCES['controller_CPU']});guard();p=validate(digest)
    if (LOCAL/'execution.json').exists():raise ValueError('No retry')
    start=time.monotonic();execution=dict(status='RUNNING',controller_pid=os.getpid(),protocol_sha256=digest,workers=[]);write(LOCAL/'execution.json',execution)
    # These checks compile forces/integration only and do not solve new cases.
    quadrature={str(n):quadrature_checks(linear.SparseStaticForward(n))for n in (2,4,8)}
    if not all(v['passed']for v in quadrature.values()):raise ValueError('Pinned production quadrature failed')
    results=[]
    pending=[(c['index'],name)for c in CASES for name in p['loads']];active={};logs=[]
    try:
        while pending or active:
            check_worker_deadlines(active,time.monotonic());guard()
            if time.monotonic()-start>RESOURCES['global_wall_cap_s']:raise central.ResourceStop('Global wall cap')
            used={r['cpu']for r in active.values()}
            for cpu in RESOURCES['worker_CPUs']:
                if cpu in used or not pending:continue
                index,name=pending.pop(0);log=(LOCAL/f'worker-case{index}-{name}.log').open('wb');logs.append(log)
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--protocol-sha256',digest,'--case-index',str(index),'--load-name',name,'--cpu',str(cpu)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row=dict(pid=process.pid,cpu=cpu,case_index=index,load=name,reaped=False,started_monotonic=time.monotonic());active[process]=row;execution['workers'].append(row)
            for process,row in list(active.items()):
                code=process.poll()
                if code is None:continue
                row.update(reaped=True,exit_code=code);del active[process];path=case_path(row['case_index'],row['load'])/'summary.json'
                if code==0 and path.is_file():results.append(json.loads(path.read_text()))
                else:results.append(dict(case_index=row['case_index'],load=row['load'],completed=False,passed=False,technical_failure=True,worker_exit_code=code))
            # Linux current RSS; children have no descendants by design.
            status=Path('/proc/self/status').read_text();rss=int(next(l for l in status.splitlines()if l.startswith('VmRSS:')).split()[1])*1024
            for row in active.values():
                try:
                    status=Path(f'/proc/{row["pid"]}/status').read_text();line=next(l for l in status.splitlines()if l.startswith('VmRSS:'));value=int(line.split()[1])*1024;rss+=value
                    if value>RESOURCES['worker_RSS_cap_bytes']:raise central.ResourceStop('Worker RSS cap')
                except(FileNotFoundError,StopIteration):pass
            if rss>RESOURCES['aggregate_RSS_cap_bytes']:raise central.ResourceStop('Aggregate RSS cap')
            execution.update(completed_groups=len(results),pending_groups=len(pending),live_groups=len(active),current_aggregate_RSS_bytes=rss,maximum_observed_aggregate_RSS_bytes=max(rss,execution.get('maximum_observed_aggregate_RSS_bytes',0)));write(LOCAL/'execution.json',execution);time.sleep(.5)
    finally:
        if sys.exc_info()[0]is not None:execution.update(status='FAILED',completed=False)
        cleanup_workers(active,execution)
        for log in logs:log.close()
    results.sort(key=lambda r:(r['case_index'],p['loads'].index(r['load'])))
    retained_groups(results,p['loads'])
    summary=dict(completed=True,groups=10,prior_cases=5,loads=2,all10_retained=True,reused_normal_mesh_cases=10,new_paired_mesh_cases=20,new_equilibria=60,qualified_groups=sum(r['passed']for r in results),all_qualified=all(r['passed']for r in results),quadrature=quadrature,per_case=results,protocol_sha256=digest,wall_s=time.monotonic()-start,no_inference_run=True,raw_arrays_LOCAL=True)
    write(LOCAL/'summary.json',summary);write(PUBLIC/'summary.json',summary);execution.update(status='COMPLETED',completed=True,all_owned_workers_reaped=True,qualified=summary['all_qualified']);write(LOCAL/'execution.json',execution);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker']);parser.add_argument('--protocol-sha256');parser.add_argument('--case-index',type=int);parser.add_argument('--load-name');parser.add_argument('--cpu',type=int);a=parser.parse_args()
    try:result=prepare()if a.mode=='prepare'else worker(a.protocol_sha256,a.case_index,a.load_name,a.cpu)if a.mode=='worker'else run(a.protocol_sha256)
    except BaseException as exc:
        if a.mode=='run':write(LOCAL/'execution-failed.json',dict(completed=False,status='FAILED',controller_pid=os.getpid(),error=type(exc).__name__+': '+str(exc)))
        raise
    print(json.dumps({k:v for k,v in result.items()if k not in ('per_case','quadrature','levels','spatial')}))
