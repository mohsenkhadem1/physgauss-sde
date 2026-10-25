"""Prospective n4 log-NH80 prefix cost/initialization control; no inference.

No physics runs on import or during prepare. Every new equilibrium is solved
independently from zero or its exact matched linear field. NH80 means 80 bulk
quadrature points, not eighty load increments. All raw fields remain LOCAL.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,os,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'runs/operator-dev/surface-nh80-cost-v01'
SOURCE='experiments/run_surface_nh80_cost_v01.py';TEST='tests/test_surface_nh80_cost_v01.py'
PARAMETERS=((2000.,5.),(4000.,float(np.sqrt(150.))),(8000.,30.))
ADMISSION=ROOT/'reports/clamped-surface-p2-catalog-qualification-v02/admission.json'
LIMITS=dict(worker_CPU=17,controller_CPU=16,threads=1,worker_RSS_cap_bytes=4*1024**3,wall_cap_s=300,cleanup_grace_s=3,disk_reserve_bytes=30*1024**3,RAM_reserve_bytes=30*1024**3,maximum_writes_bytes=1024**3,GPU=False)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def record(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha(path))

def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()

def write(path,a):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(a,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def sources():
    seen=set();pending=[SOURCE,TEST]
    while pending:
        f=pending.pop()
        if f in seen:continue
        seen.add(f)
        for node in ast.walk(ast.parse((ROOT/f).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.level==0:modules=[node.module or '']+[(node.module+'.'+x.name) for x in node.names if node.module]
            for name in modules:
                rel=name.replace('.','/')+'.py'
                if (ROOT/rel).is_file() and rel not in seen:pending.append(rel)
    return {f:sha(ROOT/f) for f in sorted(seen)}


def validate_parameters(values):
    a=np.asarray(values,dtype=float)
    if a.shape!=(3,2) or not np.array_equal(a,np.asarray(PARAMETERS)):raise ValueError('Exactly the three prospective public parameter controls')
    return a


def field_parity(cold,warm):
    a,b=np.asarray(cold,float),np.asarray(warm,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Matching finite nodal fields required')
    error=float(np.linalg.norm(a-b,axis=1).max());return dict(maximum_allnode_Euclidean_difference_m=error,tolerance_m=2e-12,passed=bool(error<=2e-12))


def admission_gate():
    a=json.loads(ADMISSION.read_text())
    if not all(a.get(k) is True for k in ['qualified','all50_cases_retained','all50_load_cases_qualified']) or a.get('target_mesh_n')!=4:raise ValueError('Qualified n4 fixed-catalog prerequisite required')
    for name,digest in a['source_SHA256'].items():
        f=Path(name) if Path(name).is_absolute() else ROOT/name
        if sha(f)!=digest:raise ValueError('Changed qualified source '+name)
    return a


def guard(role):
    if role not in ('worker','controller'):raise ValueError('Explicit process role')
    expected={LIMITS[role+'_CPU']}
    if set(os.sched_getaffinity(0))!=expected or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('Exact frozen CPU/core/one-thread/no-GPU scope')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>LIMITS['worker_RSS_cap_bytes']:raise MemoryError('4GiB own RSS limit')
    available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    if available<LIMITS['RAM_reserve_bytes'] or shutil.disk_usage(ROOT).free<LIMITS['disk_reserve_bytes']:raise MemoryError('Physical RAM/filesystem30GiB reserve')
    if LOCAL.exists() and sum(p.stat().st_size for p in LOCAL.rglob('*') if p.is_file())>LIMITS['maximum_writes_bytes']:raise OSError('Own1GiB write cap')


def prepare():
    guard('controller');a=admission_gate()
    from experiments.campaign_surface_p2_force_v03 import load_catalog
    force=load_catalog(4)['normal_x180']
    p=dict(schema='surface-n4-NH80-cost-control-v01',status='FROZEN_BEFORE_PHYSICS_ROOT_GO_REQUIRED',source_sha256=sources(),prerequisite=record(ADMISSION),parameters_E_Pa_K_N_m=[list(x) for x in PARAMETERS],n=4,label='normal_x180',quadrature_points=80,force_sha256=array_sha(force),resources=LIMITS,branches=['cold_zero','exact_linear_warm'],new_equilibria=6,linear_factors=3,geometry_and_support_shared=True,solver='Unchanged PreconditionedP2Reference; exact-rest PCG/current-LU fallback with original residual/descent/Armijo. No shifts, material or tolerance changes.',gates=dict(original_free_residual_l2_N=1e-10,minimum_sampled_J=.8,maximum_symmetric_strain=.05,minimum_nonrigid_RMS_m=100e-6,minimum_nonrigid_fraction=.3,field_parity_m=2e-12,physical_reactions='Original frozen reaction_diagnostics',stability='Original generalized_stability with full current tangent, numerical inertia evidence not rigorous certificate'),Newton_cap=40,maximum_backtracks=40,atlas_cost_projection=dict(prefix_only=True,nested_node_counts=[3,5,9],largest_nodes=81,new_physics_not_authorized_by_this_projection=True,offgrid_checks='Prospectively separate direct NH80 fields required; no old16 data fit'),scope='Six same-discrete-model equilibria/cost controls only. No atlas fit, posterior, observation/noise read, coverage calibration or architecture.',runtime_versions=dict(numpy=np.__version__,python=sys.version))
    f=LOCAL/'protocol.json'
    if f.exists():raise FileExistsError('No refreeze')
    write(f,p);return record(f)


def validate(digest):
    f=LOCAL/'protocol.json';p=json.loads(f.read_text())
    if sha(f)!=digest or p['source_sha256']!=sources() or p['resources']!=LIMITS or p['prerequisite']!=record(ADMISSION):raise ValueError('Exact source/protocol/resources/admission pins')
    validate_parameters(p['parameters_E_Pa_K_N_m']);a=admission_gate()
    from experiments.campaign_surface_p2_force_v03 import load_catalog
    if p['label']!='normal_x180' or p['n']!=4 or p['quadrature_points']!=80 or p['branches']!=['cold_zero','exact_linear_warm'] or p['new_equilibria']!=6 or array_sha(load_catalog(4)['normal_x180'])!=p['force_sha256'] or a['force_SHA256']['normal_x180']!=p['force_sha256']:raise ValueError('Exact prefix/force/80-point/six-equilibrium scope')
    return p


def worker(digest):
    guard('worker');p=validate(digest);directory=LOCAL/'raw-LOCAL'
    if directory.exists():raise FileExistsError('No retry/overwrite')
    directory.mkdir();start=time.monotonic();a=dict(schema='surface-n4-NH80-cost-result-v01',completed=False,passed=False,protocol_sha256=digest,worker_pid=os.getpid(),cases=[],raw_fields_LOCAL=True,no_cohort_data_read=True,no_inference_run=True)
    def save():a['wall_s']=time.monotonic()-start;a['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;write(LOCAL/'result.json',a)
    def stopped(*_):raise TimeoutError('Frozen300s worker cap')
    signal.signal(signal.SIGALRM,stopped);signal.signal(signal.SIGTERM,stopped);signal.alarm(300)
    try:
        from experiments.campaign_surface_p2_static_v02 import SparseStaticForward
        from experiments.campaign_surface_p2_force_v03 import load_catalog
        from experiments.campaign_surface_preconditioned_p2_v01 import PreconditionedP2Reference
        from experiments.campaign_exact_rest_pcg_v01 import ExactRestPCG
        from experiments.campaign_generalized_stability_v01 import generalized_stability
        from experiments.run_clamped_surface_p2_qualification_v03 import deformation
        from experiments.run_clamped_surface_p2_qualification_v01 import reaction_diagnostics
        begin=time.monotonic();fwd=SparseStaticForward(4);force=load_catalog(4)['normal_x180'];a['common_geometry_force_construction_s']=time.monotonic()-begin;save()
        for index,(E,K) in enumerate(PARAMETERS):
            guard('worker');begin=time.monotonic();rest=E*fwd.Kunit_free+K*fwd.W_free;directions=ExactRestPCG(rest);uL=np.zeros_like(fwd.X);uL.ravel()[fwd.free]=directions.rest_solve(force.ravel()[fwd.free]);linear_r=(E*fwd.Kunit+K*fwd.W)@uL.ravel()-force.ravel();linear_R=float(np.linalg.norm(linear_r[fwd.free]))
            if linear_R>1e-10 or np.any(uL[fwd.fixed]!=0):raise RuntimeError('Exact matched linear baseline original residual/clamp failed')
            row=dict(index=index,E_Pa=E,K_N_m=K,rest_factorization_s=directions.rest_factorization_s,linear_factor_and_solve_s=time.monotonic()-begin,linear_original_free_R_l2_N=linear_R,branches={});a['cases'].append(row);fields={};save()
            for name,initial in [('cold_zero',None),('exact_linear_warm',uL)]:
                begin=time.monotonic();model=PreconditionedP2Reference.from_forward(fwd,E,K,quadrature_order=80,direction_solver=directions);u,diag=model.solve(force,initial=initial,maximum_iterations=40,force_tolerance_N=1e-10);solve_s=time.monotonic()-begin;state=model.evaluate(u,force,tangent=True);R=float(np.linalg.norm(state['residual'].ravel()[fwd.free]));d=deformation(fwd,u);eq=reaction_diagnostics(fwd,u,force,E,K,state['residual']);stable=generalized_stability(state['tangent'][fwd.free][:,fwd.free],rest,directions.rest_LU.solve)
                passed=R<=1e-10 and d['physical_passed'] and eq['passed'] and stable['passed'] and np.all(u[fwd.fixed]==0);target=directory/f'case{index}-{name}.npz';np.savez_compressed(target,displacement_m=u,linear_displacement_m=uL,force_N=force,rest_m=fwd.X,elements=fwd.elements,exact_constraint_nodes=fwd.fixed)
                row['branches'][name]=dict(completed=True,passed=bool(passed),construction_and_equilibrium_s=solve_s,total_with_physical_checks_s=time.monotonic()-begin,original_free_R_l2_N=R,Newton=diag,deformation=d,equilibrium=eq,stability=stable,capture=record(target),current_fallback_factorizations_cumulative=directions.current_fallback_factorizations);fields[name]=u;guard('worker');save()
                if not passed:raise RuntimeError('Original physical/residual/stability gate failed; captured negative retained')
            row['field_parity']=field_parity(fields['cold_zero'],fields['exact_linear_warm']);save()
            if not row['field_parity']['passed']:raise RuntimeError('Cold/warm same-field parity gate failed')
        a.update(completed=True,passed=True);save()
    except BaseException as error:a['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());save();raise
    finally:signal.alarm(0);save()
    return a


def cleanup(child):
    if child.poll() is None:
        if os.getpgid(child.pid)!=child.pid:raise RuntimeError('Owned isolated process group only')
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
    return child.wait()


def run(digest):
    guard('controller');p=validate(digest)
    if (LOCAL/'execution.json').exists() or (LOCAL/'raw-LOCAL').exists():raise FileExistsError('No retry')
    start=time.monotonic();child=None;state=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_sha256=digest)
    def stopped(*_):raise InterruptedError('Owned controller stopped')
    old={sig:signal.signal(sig,stopped) for sig in [signal.SIGTERM,signal.SIGINT]}
    try:
        with (LOCAL/'worker.log').open('xb') as log:
            child=subprocess.Popen(['taskset','-c','17',sys.executable,'-u','-m','experiments.run_surface_nh80_cost_v01','worker','--sha',digest],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['worker_pid']=child.pid;write(LOCAL/'execution.json',state)
            while child.poll() is None:
                guard('controller')
                try:rss=int(next(x for x in Path('/proc',str(child.pid),'status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024
                except FileNotFoundError:rss=0
                if rss>LIMITS['worker_RSS_cap_bytes'] or time.monotonic()-start>300:raise TimeoutError('Independent parent wall/RSS cap')
                time.sleep(.1)
            state['worker_exit_code']=child.wait()
    except BaseException as e:state['failure']=repr(e)
    finally:
        if child is not None:state['cleanup_exit_code']=cleanup(child)
        for sig,handler in old.items():signal.signal(sig,handler)
        state.update(terminal=True,wall_s=time.monotonic()-start)
        f=LOCAL/'result.json'
        if f.exists():
            a=json.loads(f.read_text());state.update(completed=bool(a['completed'] and a['passed'] and state.get('cleanup_exit_code')==0),result=record(f));cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(sha(f)+'.json');cache.parent.mkdir(parents=True,exist_ok=True)
            if cache.exists() and cache.read_bytes()!=f.read_bytes():raise ValueError('Immutable metadata cache mismatch')
            if not cache.exists():cache.write_bytes(f.read_bytes())
            state['external_cache']=record(cache)
        write(LOCAL/'execution.json',state)
    return state

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','run']);p.add_argument('--sha');args=p.parse_args()
    if args.action=='prepare':print(json.dumps(prepare()))
    elif args.action=='worker':r=worker(args.sha);print(json.dumps({k:r[k] for k in ['completed','passed','wall_s']}))
    else:r=run(args.sha);print(json.dumps(r));raise SystemExit(0 if r['completed'] else 1)
