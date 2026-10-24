"""Separate all-load P2 physics admission; central qualification is mandatory.

No inference. The frozen central metric and mechanics are reused unchanged.
Every one of five prior cases and ten catalog loads remains in the denominator.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments import run_clamped_surface_p2_qualification_v03 as central
from experiments import campaign_surface_p2_force_v03 as force_program
from experiments import run_clamped_surface_p2_qualification_v01 as reference_metrics
from experiments.campaign_exact_rest_pcg_v01 import ExactRestPCG
from experiments.campaign_surface_preconditioned_p2_v01 import PreconditionedP2Reference
from experiments.campaign_generalized_stability_v01 import generalized_stability, DESIGN as GENERALIZED_STABILITY_DESIGN

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'runs/operator-dev/clamped-surface-p2-catalog-qualification-v02'
PUBLIC=ROOT/'reports/clamped-surface-p2-catalog-qualification-v02'
CASES=copy.deepcopy(central.CASES);GATES=copy.deepcopy(central.GATES)
ResourceStop=reference_metrics.ResourceStop
RESOURCES=dict(worker_CPUs=list(range(4,12)),controller_CPU=11,threads_per_worker=1,workers=8,GPU=False,worker_wall_cap_s=1800,global_wall_cap_s=7200,worker_RSS_cap_bytes=8*1024**3,aggregate_RSS_cap_bytes=64*1024**3,RAM_reserve_bytes=30*1024**3,RAM_preflight_bytes=94*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=2*1024**3)
CENTRAL_PROTOCOL_SHA256='4f93e3bbc2cab4c6640fe9a024ad36091de1e590940b8538a89419b82649e3c5'
SOLVER_DESIGN=dict(Newton='Unchanged P2 compressible Neo-Hookean residual, Armijo and final tolerance',direction='Shared exact-rest LU preconditioned CG; unmodified current-LU fallback with explicit residual/descent checks',linear='Same shared rest LU with explicit residual',stability=GENERALIZED_STABILITY_DESIGN,prospective_change_from_central='Generalized smallest-algebraic inertia diagnostic replaces ordinary smallest eigsh for NEW catalog states only; central reused ordinary diagnostics untouched',no_matrix_modification=True)
CENTRAL_ALIASES={'normal_x180':'normal','paired_axial_x180':'paired_axial'}
sha=central.sha;write=central.write


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float64).tobytes()).hexdigest()


def verify_terminal_bindings(base, public):
    inventory=base/'publication-inventory-completed.json'
    inv=json.loads(inventory.read_text())
    for item in inv['files']:
        if sha(ROOT/item['path'])!=item['sha256']:
            raise ValueError('Central published artifact changed')
    ownership_path=public/'terminal-ownership.json'
    ownership=json.loads(ownership_path.read_text())
    if sha(base/'execution.json')!=ownership['execution_sha256'] or sha(base/'summary.json')!=ownership['summary_sha256']:
        raise ValueError('Central terminal ownership binding failed')
    for case in CASES:
        for name in CENTRAL_ALIASES.values():
            for n in (2,4,8):
                archive=base/f'case-{case["index"]:02d}-{name}'/f'n{n}-LOCAL.npz'
                if sha(archive)!=ownership['raw_fields_sha256'][str(archive.relative_to(base))]:
                    raise ValueError('Central raw archive expected hash mismatch')
    return {str(inventory.relative_to(ROOT)):sha(inventory),str(ownership_path.relative_to(ROOT)):sha(ownership_path)}


def central_gate():
    base=central.LOCAL;p=json.loads((base/'protocol.json').read_text());s=json.loads((base/'summary.json').read_text());e=json.loads((base/'execution.json').read_text())
    if sha(base/'protocol.json')!=CENTRAL_PROTOCOL_SHA256 or p['source_sha256']!=central.sources():raise ValueError('Frozen central source/protocol mismatch')
    if e['status']!='COMPLETED'or not e['completed']or not s['all10_retained']or not s['all_qualified']or s['qualified_groups']!=10:raise ValueError('All ten central groups must qualify before catalog')
    try:os.kill(e['controller_pid'],0)
    except ProcessLookupError:pass
    else:raise ValueError('Central controller still live')
    expected={(c['index'],name)for c in CASES for name in ('normal','paired_axial')}
    if len(e['workers'])!=10 or {(r['case_index'],r['load'])for r in e['workers']}!=expected:raise ValueError('Central owned-worker mapping failed')
    for r in e['workers']:
        if not r['reaped']or r['exit_code']!=0:raise ValueError('Central worker not reaped/exit0')
        for check in(os.kill,os.killpg):
            try:check(r['pid'],0)
            except ProcessLookupError:pass
            else:raise ValueError('Central owned worker PID/group remains')
    binding_pins=verify_terminal_bindings(base,central.PUBLIC)
    paths=[base/name for name in ('protocol.json','summary.json','execution.json','publication-inventory-completed.json')]
    for c in CASES:
        for old in CENTRAL_ALIASES.values():
            group=base/f'case-{c["index"]:02d}-{old}';paths.append(group/'summary.json')
            for n in (2,4,8):paths.append(group/f'n{n}-LOCAL.npz')
    return {**binding_pins,**{str(p.relative_to(ROOT)):sha(p)for p in paths}}


def sources():
    pending=['experiments/run_clamped_surface_p2_catalog_qualification_v02.py','tests/test_clamped_surface_p2_catalog_qualification_v02.py','tests/test_generalized_stability_v01.py','tests/test_exact_rest_pcg_v01.py','tests/test_surface_preconditioned_p2_v01.py'];seen=set()
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
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES['worker_RSS_cap_bytes']:raise ResourceStop('RSS cap exceeded')
    if available_RAM()<RESOURCES['RAM_reserve_bytes']:raise ResourceStop('Physical RAM reserve failed')
    import shutil
    if shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise ResourceStop('Disk reserve failed')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise ResourceStop('Output budget exceeded')


def available_RAM():
    return int(next(l for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')).split()[1])*1024


def catalog_deformation(d,prefix):
    d=copy.deepcopy(d)
    d['validity_passed']=bool(d['minimum_J']>=GATES['minimum_J']and d['maximum_absolute_symmetric_strain_eigenvalue']<=GATES['maximum_absolute_symmetric_strain_eigenvalue'])
    d['excitation_passed']=bool(d['nonrigid_RMS_m']>=GATES['minimum_nonrigid_RMS_m']and d['nonrigid_fraction']>=GATES['minimum_nonrigid_fraction'])
    d['excitation_required_for_catalog_admission']=bool(prefix)
    d['catalog_physical_passed']=bool(d['validity_passed']and(not prefix or d['excitation_passed']))
    return d


def group_passed(record):
    return bool(record['completed']and all(all(row[law]['deformation']['catalog_physical_passed']and row[law]['equilibrium']['passed']for law in ('linear','nonlinear','NH80_reference'))and row['nonlinear']['stability']['passed']and row['NH80_reference']['stability']['passed']and row['linear_vs_nonlinear_passed']and row['NH24_vs_NH80_passed']for row in record['levels'].values())and all(d['spatial_passed']and d['medium_adequate']for d in record['spatial'].values()))


def catalog_force_checks(n):
    X=central.linear.geometry(n)[0];result={}
    for name,force in force_program.load_catalog(n).items():
        center=int(name.rsplit('_x',1)[1])/1000;normal=name.startswith('normal_')
        expected=np.array([0,0,-force_program.NORMAL_FORCE_N])if normal else np.array([.36,0,0]);moment=np.array([0,force_program.NORMAL_FORCE_N*(center-.18),0])if normal else np.zeros(3)
        df=float(np.linalg.norm(force.sum(0)-expected));dm=float(np.linalg.norm(np.cross(X-[.18,.04,.04],force).sum(0)-moment))
        result[name]=dict(net_force_error_N=df,known_resultant_moment_error_Nm=dm,passed=bool(df<=central.linear.shared.GATES['load_net_force_error_N']and dm<=central.linear.shared.GATES['load_moment_error_Nm']))
    return result


def validation_inputs():
    paths=['runs/operator-dev/pcg-n4-parity-result.json',
           'runs/operator-dev/pcg-n8-parity-result.json',
           'runs/operator-dev/generalized-stability-v01/result.json',
           'runs/operator-dev/generalized-stability-n8-v01/result.json',
           'runs/operator-dev/clamped-surface-p2-catalog-qualification-v02/controls-pcg-generalized.json']
    for name in paths:
        value=json.loads((ROOT/name).read_text())
        if not value.get('all_passed',value.get('passed',False)):
            raise ValueError('Numerical controls or parity not passed')
    return {name:sha(ROOT/name)for name in paths}


def resource_projection():
    old=json.loads((central.LOCAL/'summary.json').read_text());maximum=max(row['worker_wall_s']for row in old['per_case'])
    parity=json.loads((ROOT/'runs/operator-dev/pcg-n8-parity-result.json').read_text());empirical_base=max(maximum,parity['wall_s'])
    estimate=2*5*(empirical_base+20)
    if estimate>RESOURCES['global_wall_cap_s']:raise ValueError('Observed central cost exceeds declared catalog budget; review before freeze')
    return dict(maximum_observed_central_group_s=maximum,observed_n8_four_cold_equilibria_and_rest_factor_s=parity['wall_s'],empirical_base_s=empirical_base,waves=5,safety_factor=2,empirical_planning_estimate_wall_s=estimate,extra_validation_s_per_group=20,no_runtime_promise=True,proven_worst_case_bound=False)


def force_pins():
    return {str(n):{name:array_sha(f)for name,f in force_program.load_catalog(n).items()}for n in (2,4,8)}


def verify_central_loads():
    for n in (2,4,8):
        catalog=force_program.load_catalog(n);old=central.forces.qualification_loads(n)
        for name,alias in CENTRAL_ALIASES.items():
            if not np.array_equal(catalog[name],old[alias]):raise ValueError('Central load byte mismatch')
    return True


def prepare():
    guard();pins=central_gate();verify_central_loads()
    if available_RAM()<RESOURCES['RAM_preflight_bytes']:raise ValueError('94GiB RAM preflight required')
    if (LOCAL/'protocol.json').exists():raise ValueError('No refreeze/retry')
    names=list(force_program.load_catalog(1))
    p=dict(schema='clamped-surface-p2-catalog-qualification-v02',resource_projection=resource_projection(),solver_design=SOLVER_DESIGN,prefix_only_excitation_required='normal_x180',new_prospective_validity_vs_excitation_rule=True,cases=CASES,loads=names,levels=[2,4,8],gates=GATES,resources=RESOURCES,source_sha256=sources(),validation_inputs_sha256=validation_inputs(),central_inputs_sha256=pins,force_bytes_sha256=force_pins(),central_aliases=CENTRAL_ALIASES,expected_groups=50,reused_central_groups=10,new_groups=40,new_paired_mesh_solves=120,new_equilibria=360,force_program_design=force_program.design(),metric='Exact frozen central metric functions/source hashes; common finest positiveKeast24 coordinate RMS',all50_retained=True,no_inference=True,no_retuning=True)
    write(LOCAL/'protocol.json',p);return dict(protocol=str(LOCAL/'protocol.json'),sha256=sha(LOCAL/'protocol.json'))


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['source_sha256']!=sources()or p['central_inputs_sha256']!=central_gate()or p['force_bytes_sha256']!=force_pins()or p['validation_inputs_sha256']!=validation_inputs():raise ValueError('Frozen protocol/source/input/force mismatch')
    verify_central_loads();return p


def case_path(index,name):return LOCAL/f'case-{index:02d}-{name}'


def solve_group(case,name):
    start=time.monotonic();fields={};record=dict(case_index=case['index'],load=name,levels={},completed=True,reused_central=False);E=case['E_Pa'];K=case['K_N_m'];base=case_path(case['index'],name);base.mkdir(parents=True,exist_ok=False)
    for n in (2,4,8):
        guard();fwd=central.linear.SparseStaticForward(n);force=force_program.load_catalog(n)[name];row={};fields[n]={}
        try:
            rest=ExactRestPCG(E*fwd.Kunit_free+K*fwd.W_free)
            u=np.zeros_like(fwd.X);u.ravel()[fwd.free]=rest.rest_solve(force.ravel()[fwd.free]);fields[n]['linear']=u
            row['linear']=dict(deformation=catalog_deformation(central.deformation(fwd,u),name=='normal_x180'),equilibrium=reference_metrics.reaction_diagnostics(fwd,u,force,E,K))
            for order,key in ((24,'nonlinear'),(80,'NH80_reference')):
                model=PreconditionedP2Reference.from_forward(fwd,E,K,quadrature_order=order,direction_solver=rest);un,diag=model.solve(force);state=model.evaluate(un,force,tangent=True)
                try:stability=generalized_stability(state['tangent'][model.free][:,model.free],rest.rest,rest.rest_LU.solve)
                except Exception as exc:stability=dict(completed=False,passed=False,error=type(exc).__name__+': '+str(exc))
                row[key]=dict(deformation=catalog_deformation(central.deformation(fwd,un),name=='normal_x180'),equilibrium=reference_metrics.reaction_diagnostics(fwd,un,force,E,K,state['residual']),Newton=diag,stability=stability)
                fields[n][key]=un
            un=fields[n]['nonlinear'];un80=fields[n]['NH80_reference']
            row['NH24_vs_NH80_RMS_m']=float(central.common_rms([(n,un),(n,un80)])[1,0]);row['NH24_vs_NH80_passed']=row['NH24_vs_NH80_RMS_m']<=GATES['maximum_NH24_vs_NH80_RMS_m']
            row['linear_vs_nonlinear_RMS_m']=float(central.common_rms([(n,u),(n,un)])[1,0]);row['linear_vs_nonlinear_passed']=row['linear_vs_nonlinear_RMS_m']<=GATES['maximum_linear_vs_nonlinear_RMS_m']
            np.savez_compressed(base/f'n{n}-LOCAL.npz',linear=u,nonlinear=un,nonlinear80=un80)
        except Exception as exc:row['completed']=False;row['error']=type(exc).__name__+': '+str(exc);record['completed']=False
        record['levels'][str(n)]=row;guard()
    record['spatial']={}
    for law in ('linear','nonlinear'):
        if all(law in fields[n]for n in (2,4,8)):
            d=central.common_rms([(n,fields[n][law])for n in (2,4,8)]);record['spatial'][law]=reference_metrics.spatial_gate(float(d[1,0]),float(d[2,1]))
        else:record['spatial'][law]=dict(spatial_passed=False,medium_adequate=False,error='Incomplete retained fields')
    record['passed']=group_passed(record)
    record.update(worker_wall_s=time.monotonic()-start,worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(base/'summary.json',record);return record


def worker(digest,index,name,cpu):
    os.sched_setaffinity(0,{cpu});guard(cpu);p=validate(digest)
    if name not in p['loads']or name in CENTRAL_ALIASES or index not in range(5):raise ValueError('Only prospective new groups in worker')
    def timeout(*_):raise ResourceStop('Worker frozen wall cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(RESOURCES['worker_wall_cap_s'])
    return solve_group(CASES[index],name)


def check_worker_deadlines(active,now):
    for row in active.values():
        if now-row['started_monotonic']>RESOURCES['worker_wall_cap_s']:
            raise ResourceStop('Independent parent worker wall cap')


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
    if available_RAM()<RESOURCES['RAM_preflight_bytes']:raise ValueError('94GiB RAM preflight required')
    if (LOCAL/'execution.json').exists():raise ValueError('No retry')
    start=time.monotonic();execution=dict(status='RUNNING',controller_pid=os.getpid(),protocol_sha256=digest,workers=[]);write(LOCAL/'execution.json',execution)
    # These checks compile forces/integration only and do not solve new cases.
    quadrature={str(n):central.quadrature_checks(central.linear.SparseStaticForward(n))for n in (2,4,8)}
    catalog_forces={str(n):catalog_force_checks(n)for n in (2,4,8)}
    if not all(v['passed']for v in quadrature.values())or not all(v['passed']for block in catalog_forces.values()for v in block.values()):raise ValueError('Pinned production quadrature/catalog force failed')
    results=[]
    for case in CASES:
        for name,alias in CENTRAL_ALIASES.items():
            old=json.loads((central.LOCAL/f'case-{case["index"]:02d}-{alias}'/'summary.json').read_text());old=copy.deepcopy(old);old.update(load=name,reused_central=True,reused_central_load=alias)
            for row in old['levels'].values():
                for law in('linear','nonlinear','NH80_reference'):row[law]['deformation']=catalog_deformation(row[law]['deformation'],name=='normal_x180')
            old['passed']=group_passed(old);results.append(old)

    pending=[(c['index'],name)for c in CASES for name in p['loads']if name not in CENTRAL_ALIASES];active={};logs=[]
    try:
        while pending or active:
            check_worker_deadlines(active,time.monotonic());guard()
            if time.monotonic()-start>RESOURCES['global_wall_cap_s']:raise ResourceStop('Global wall cap')
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
                    if value>RESOURCES['worker_RSS_cap_bytes']:raise ResourceStop('Worker RSS cap')
                except(FileNotFoundError,StopIteration):pass
            if rss>RESOURCES['aggregate_RSS_cap_bytes']:raise ResourceStop('Aggregate RSS cap')
            execution.update(completed_groups=len(results),pending_groups=len(pending),live_groups=len(active),current_aggregate_RSS_bytes=rss,maximum_observed_aggregate_RSS_bytes=max(rss,execution.get('maximum_observed_aggregate_RSS_bytes',0)));write(LOCAL/'execution.json',execution);time.sleep(.5)
    finally:
        if sys.exc_info()[0]is not None:execution.update(status='FAILED',completed=False)
        cleanup_workers(active,execution)
        for log in logs:log.close()
    results.sort(key=lambda r:(r['case_index'],p['loads'].index(r['load'])))
    expected={(c['index'],name)for c in CASES for name in p['loads']}
    if len(results)!=50 or {(r['case_index'],r['load'])for r in results}!=expected:raise ValueError('All50 denominator/mapping failed')
    summary=dict(completed=True,groups=50,prior_cases=5,loads=10,all50_retained=True,reused_central_groups=10,new_groups=40,qualified_groups=sum(r['passed']for r in results),all_qualified=all(r['passed']for r in results),quadrature=quadrature,catalog_force_checks=catalog_forces,per_case=results,protocol_sha256=digest,wall_s=time.monotonic()-start,no_inference_run=True,raw_arrays_LOCAL=True)
    write(LOCAL/'summary.json',summary);write(PUBLIC/'summary.json',summary);execution.update(status='COMPLETED',completed=True,all_owned_workers_reaped=True,qualified=summary['all_qualified']);write(LOCAL/'execution.json',execution)
    admission=dict(qualified=summary['all_qualified'],target_mesh_n=4,all50_cases_retained=True,all50_load_cases_qualified=summary['all_qualified'],force_SHA256=force_pins()['4'],source_SHA256=p['source_sha256'],protocol_SHA256=digest)
    write(PUBLIC/'admission.json',admission);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker']);parser.add_argument('--protocol-sha256');parser.add_argument('--case-index',type=int);parser.add_argument('--load-name');parser.add_argument('--cpu',type=int);a=parser.parse_args()
    try:result=prepare()if a.mode=='prepare'else worker(a.protocol_sha256,a.case_index,a.load_name,a.cpu)if a.mode=='worker'else run(a.protocol_sha256)
    except BaseException as exc:
        if a.mode=='run':write(LOCAL/'execution-failed.json',dict(completed=False,status='FAILED',controller_pid=os.getpid(),error=type(exc).__name__+': '+str(exc)))
        raise
    print(json.dumps({k:v for k,v in result.items()if k not in ('per_case','quadrature','levels','spatial')}))
