"""NEW Gaussian-load NH80 development cost pilot; old narrow-load negative retained.

Preparing pins no fields. Root release is required for six new equilibria.
The full 660-field catalog is a plan, never an automatic queue expansion.
"""
import argparse,ast,json,os,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import norm as sparse_norm
from experiments import campaign_active_fixture_gaussian_load_v02 as bank
from experiments import run_active_fixture_mechanics_screen_v01 as prior
from experiments import run_active_fixture_nh80_cost_v01 as predecessor
from experiments.campaign_exact_rest_pcg_v01 import ExactRestPCG
from experiments.campaign_surface_preconditioned_p2_v01 import PreconditionedP2Reference
from experiments.campaign_generalized_stability_v01 import generalized_stability
from experiments import run_clamped_surface_p2_qualification_v03 as metrics
from experiments.run_clamped_surface_p2_qualification_v01 import reaction_diagnostics,spatial_gate

ROOT=prior.ROOT;LOCAL=ROOT/'runs/operator-dev/active-fixture-gaussian-nh80-cost-v02'
SOURCE='experiments/run_active_fixture_gaussian_nh80_cost_v02.py';TEST='tests/test_active_fixture_gaussian_nh80_cost_v02.py'
sha,record,write=prior.sha,prior.record,prior.write
PARAMETERS=((2000.,5.),(2000.,30.),(8000.,5.),(8000.,30.),(4000.,float(np.sqrt(150.))))
LEVELS=(2,4,8);PILOT_LAYOUT=bank.LAYOUTS[0]['id'];PILOT_PARAMETERS=(2000.,5.)
PILOT_LOADS=('prefix-center-down','probe-x270-y060-plusY')
LIMITS=dict(worker_CPU=17,controller_CPU=18,threads=1,GPU=False,wall_cap_s=600,worker_RSS_cap_bytes=4*1024**3,controller_RSS_cap_bytes=4*1024**3,aggregate_RSS_cap_bytes=8*1024**3,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=512*1024**2)


def catalog():
    return [dict(id=f'{layout["id"]}-parameter{j}',layout=layout['id'],parameter_index=j,E_Pa=E,K_N_m=K) for layout in bank.LAYOUTS for j,(E,K) in enumerate(PARAMETERS)]


def plan():
    return dict(schema='active-fixture-Gaussian-NH80-cost-v02',resources=LIMITS,bank_design=bank.design(),full_future_catalog=catalog(),future_loads=[x['id'] for x in bank.load_specs()],future_levels=list(LEVELS),future_state_loads=220,future_NH80_fields=660,future_execution_authorized=False,pilot=dict(layout=PILOT_LAYOUT,E_Pa=PILOT_PARAMETERS[0],K_N_m=PILOT_PARAMETERS[1],loads=list(PILOT_LOADS),levels=list(LEVELS),new_equilibria=6,selection='Same fixed layout/softcorner/loadlabels as old negative. NEW full-topGaussian std(.035,.025), candidate/held.06N,prefix.04N. No response-based case filtering.'),gates=dict(metrics.GATES),weak_prefix_excitation_exemption='Only minimum100um nonrigid RMS; nonrigidfraction>=.3 and residual/J/strain/reactions/clamps/stability unchanged.',warm_start='Exact same-load/same-layout/E/K linear equilibrium independently at each mesh; no old nonlinear fields.',Newton_iterations=40,force_tolerance_l2_N=1e-10,quadrature_points=80,quadrature_scope='Only known traction law changed; support/material/geometry/thresholds frozen. Unchanged positive80 log-NH reference. NH24vs80 production admission remains untested; this pilot does not admit a24point surrogate.',J_scope='Sampled vertices/edges/24+80 guard; no whole-element determinant certificate.',spatial_scope='Original finest positiveKeast24 coordinate RMS, p>=1 conditional fine indicator and2*delta medium indicator<=20um. Additional actual public point max20um. Conditional indicators are not rigorous continuum error bounds.',cost_projection=dict(estimated_nodes=[425,2673,18785],estimated_free_DOF=[1125,7533,54621],base_tets=[192,1536,12288],n8_local_H_values_bytes=12288*30*30*8,n8_H_rows_cols_bytes=2*12288*30*30*8,factor_fill_unknown=True,authoritative_guard='4GiB actual worker RSS and600s hard cap; no automatic budget escalation',full660_cost_must_use_measured_timings=True),scope='DEVELOPMENT cost/reference pilot only; no posterior/observations/noise/test draw, policy or bank admission. All negatives retained, no force search.')


def sources():
    todo=[SOURCE,TEST,'tests/test_active_fixture_gaussian_load_v02.py'];seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.level==0:modules=[node.module or '']+[(node.module+'.'+x.name) for x in node.names if node.module]
            for module in modules:
                f=module.replace('.','/')+'.py'
                if (ROOT/f).is_file() and f not in seen:todo.append(f)
    return {f:sha(ROOT/f) for f in sorted(seen)}


def negative_origin():
    base=predecessor.LOCAL;digest=sha(base/'protocol.json');predecessor.validate(digest)
    e=json.loads((base/'execution.json').read_text());r=json.loads((base/'result-LOCAL.json').read_text())
    if sha(base/'result-LOCAL.json')!='3edfb9380c58f02bd8c280be850429e0d944d10e634326eed3eeb9bf4a6fe296' or not e.get('terminal') or not e.get('completed') or e.get('worker_exit_code')!=0 or e.get('cleanup_exit_code')!=0 or not r.get('completed') or not r.get('all6_retained') or r.get('passed') is not False or r.get('bank_admitted') is not False:raise ValueError('Preserve exact COMPLETE six-solve narrow-load negative')
    if e['result']!=record(base/'result-LOCAL.json') or e['release']!=record(base/'release.json') or r['release']!=e['release'] or e['external_cache']!=record(ROOT/e['external_cache']['path']) or (ROOT/e['external_cache']['path']).read_bytes()!=(base/'result-LOCAL.json').read_bytes():raise ValueError('Negative exact result/release/cache')
    if len(r['rows'])!=6 or {(x['n'],x['label']) for x in r['rows']}!={(n,label) for n in LEVELS for label in PILOT_LOADS}:raise ValueError('All negative pilot members retained')
    for row in r['rows']:
        if row['capture']!=record(ROOT/row['capture']['path']):raise ValueError('Old negative capture changed')
    prior.shared.old.absent(e['controller_pid']);prior.shared.old.absent(e['worker_pid'])
    review=json.loads((base/'independent-review.json').read_text())
    if review.get('audit_passed') is not True or review.get('scientific_admission') is not False or review['source_result']!=e['result']:raise ValueError('Independent negative audit binding')
    return dict(protocol=record(base/'protocol.json'),release=record(base/'release.json'),execution=record(base/'execution.json'),result=e['result'],independent_review=record(base/'independent-review.json'),captures=[x['capture'] for x in r['rows']],new_force_law=True,old_negative_preserved=True)


def guard(role):
    if set(os.sched_getaffinity(0))!={LIMITS[role+'_CPU']} or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact CPU17worker/18controller/one-thread/noGPU')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>LIMITS[role+'_RSS_cap_bytes']:raise MemoryError('Own4GiB RSS')
    available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    if available<LIMITS['RAM_reserve_bytes'] or shutil.disk_usage(ROOT).free<LIMITS['disk_reserve_bytes']+LIMITS['maximum_writes_bytes']:raise MemoryError('30GiB RAM/disk plus projected write reserve')
    if LOCAL.exists() and sum(p.stat().st_size for p in LOCAL.rglob('*') if p.is_file())>LIMITS['maximum_writes_bytes']:raise OSError('Own512MiB output cap')


def prepare():
    guard('controller')
    if (LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    origin=negative_origin()
    p=dict(plan=plan(),source_SHA256=sources(),origin=origin,status='FROZEN_ROOT_RELEASE_REQUIRED')
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text());r=json.loads((LOCAL/'release.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['plan']!=plan() or p['source_SHA256']!=sources() or p['origin']!=negative_origin():raise ValueError('Frozen design/source/origin')
    if r.get('approved') is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']:raise ValueError('Root release before physics')
    return p


def admissibility(deformation,R,clamp,equilibrium,stability,load):
    weak=load=='prefix-center-down';g=metrics.GATES
    material=bool(np.isfinite([deformation['minimum_J'],deformation['maximum_absolute_symmetric_strain_eigenvalue'],R]).all() and deformation['minimum_J']>=g['minimum_J'] and deformation['maximum_absolute_symmetric_strain_eigenvalue']<=g['maximum_absolute_symmetric_strain_eigenvalue'])
    fraction=bool(deformation['nonrigid_fraction']>=g['minimum_nonrigid_fraction'])
    excitation=bool(deformation['nonrigid_RMS_m']>=g['minimum_nonrigid_RMS_m'] and fraction)
    physical=bool(material and R<=1e-10 and clamp and equilibrium['passed'] and stability['passed'])
    return dict(physical_passed=physical,excitation_goal_met=excitation,weak_prefix_control=weak,load_eligible=bool(physical and fraction and (weak or excitation)),bank_admitted=False)


def surface_checks(f,loads):
    W20,_=bank.consistent_fixture(f.X,f.faces,PILOT_LAYOUT,order=20)
    error=float(sparse_norm(f.W_scalar-W20)/sparse_norm(W20));ref,info=bank.load_catalog(f.n,order=20);production_info=bank.load_catalog(f.n)[1]
    load_error=max(float(np.max(abs(loads[k]-ref[k]))) for k in loads)
    # PSD follows from the explicit positive quadrature Gram, not a large dense eigensolve.
    if f.support_quadrature['normalization_error']>1e-10 or f.support_quadrature['first_moment_error_m']>1e-10 or error>1e-10 or load_error>1e-12 or any(x['net_force_error_N']>1e-12 or x['moment_error_Nm']>1e-13 for x in list(info.values())+list(production_info.values())) or any(x['normalization_error']>1e-10 or x['first_moment_error_m']>1e-10 for x in list(info.values())+list(production_info.values())):raise ValueError('Same physical normalized PSDsupport/knownload quadrature guards')
    return dict(support=f.support_quadrature,positive_quadrature_Gram_PSD=True,relative_support16_vs20=error,maximum_load16_vs20_N=load_error,refined_known_load_checks=info)


def spatial_results(fields):
    out={}
    for label in PILOT_LOADS:
        values=[(n,fields[n,label]) for n in LEVELS];d=metrics.common_rms(values);g=spatial_gate(float(d[1,0]),float(d[2,1]));q=[bank.core.field_queries(n,u,bank.measurement_points()) for n,u in values]
        delta=float(np.max(abs(q[2]-q[1])));coarse=float(np.max(abs(q[1]-q[0])))
        g.update(public_point_n2_n4_max_coordinate_m=coarse,public_point_n4_n8_max_coordinate_m=delta,public_point20um_passed=delta<=20e-6,spatial_development_passed=bool(g['spatial_passed'] and g['medium_adequate'] and delta<=20e-6),bank_admitted=False)
        out[label]=g
    return out


def worker(digest):
    guard('worker');p=validate(digest)
    if (LOCAL/'result-LOCAL.json').exists():raise FileExistsError('No retry')
    begin=time.monotonic();out=dict(completed=False,passed=False,protocol_SHA256=digest,release=record(LOCAL/'release.json'),worker_pid=os.getpid(),rows=[],meshes={},no_noise_or_truth_draws=True,bank_admitted=False)
    def save():out.update(wall_s=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'result-LOCAL.json',out)
    def timeout(*_):raise TimeoutError('Frozen600s cap')
    def interrupted(*_):raise InterruptedError('Explicit worker interruption')
    previous={s:signal.signal(s,h) for s,h in ((signal.SIGALRM,timeout),(signal.SIGTERM,interrupted))};signal.alarm(LIMITS['wall_cap_s']);fields={}
    try:
        E,K=PILOT_PARAMETERS
        for n in LEVELS:
            guard('worker');start=time.monotonic();f=bank.make_forward(n,PILOT_LAYOUT);loads,info=bank.load_catalog(n);checks=surface_checks(f,loads);rest=E*f.Kunit_free+K*f.W_free;directions=ExactRestPCG(rest);model=PreconditionedP2Reference.from_forward(f,E,K,quadrature_order=80,direction_solver=directions)
            out['meshes'][str(n)]=dict(initialization_s=time.monotonic()-start,rest_factorization_s=directions.rest_factorization_s,surface=checks,nodes=len(f.X),free_DOF=len(f.free));save()
            for label in PILOT_LOADS:
                out['pending_case']=dict(n=n,label=label,layout=PILOT_LAYOUT,E_Pa=E,K_N_m=K);save()
                guard('worker');start=time.monotonic();force=loads[label];uL=np.zeros_like(f.X);uL.ravel()[f.free]=directions.rest_solve(force.ravel()[f.free]);linear_R=float(np.linalg.norm(((E*f.Kunit+K*f.W)@uL.ravel()-force.ravel())[f.free]))
                if linear_R>1e-10 or np.any(uL[f.fixed]!=0):raise ValueError('Exact linear warm-start R/clamp failed')
                linear_s=time.monotonic()-start;start=time.monotonic();u,diag=model.solve(force,initial=uL,maximum_iterations=40,force_tolerance_N=1e-10);solve_s=time.monotonic()-start;start=time.monotonic();state=model.evaluate(u,force,tangent=True);R=float(np.linalg.norm(state['residual'].ravel()[f.free]));d=metrics.deformation(f,u);eq=reaction_diagnostics(f,u,force,E,K,state['residual']);stable=generalized_stability(state['tangent'][f.free][:,f.free],rest,directions.rest_LU.solve);status=admissibility(d,R,np.all(u[f.fixed]==0),eq,stable,label)
                target=LOCAL/'raw-LOCAL'/f'n{n}-{label}.npz';target.parent.mkdir(exist_ok=True);np.savez_compressed(target,displacement_m=u,linear_displacement_m=uL,rest_m=f.X,force_N=force,elements=f.elements,exact_constraint_nodes=f.fixed,query_points_m=bank.measurement_points(),query_displacement_m=bank.core.field_queries(n,u,bank.measurement_points()))
                row=dict(n=n,label=label,layout=PILOT_LAYOUT,E_Pa=E,K_N_m=K,completed=True,**status,linear_original_R_l2_N=linear_R,original_free_R_l2_N=R,deformation=d,equilibrium=eq,stability=stable,Newton=diag,capture=record(target),known_load=info[label],actual_deformed_wrench=prior.force_wrench(f.X,force,u),linear_warm_s=linear_s,NH80_solve_s=solve_s,independent_checks_and_capture_s=time.monotonic()-start,current_LU_fallbacks_cumulative=directions.current_fallback_factorizations)
                out['rows'].append(row);fields[n,label]=u;save();guard('worker')
        out.update(completed=True,all6_retained=len(out['rows'])==6,all_physical_passed=all(x['physical_passed'] for x in out['rows']),spatial=spatial_results(fields));out['passed']=bool(out['all6_retained'] and out['all_physical_passed']);out['scope']=p['plan']['scope'];save()
    except BaseException as error:out['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),accepted_Newton_history=getattr(error,'history',None));save();raise
    finally:
        signal.alarm(0)
        for sig,handler in previous.items():signal.signal(sig,handler)
        save()
    return out


def run(digest):
    guard('controller');validate(digest)
    if (LOCAL/'execution.json').exists():raise FileExistsError('No retry')
    begin=time.monotonic();child=None;e=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'))
    def stopped(*_):raise InterruptedError('Explicit controller interruption')
    previous={sig:signal.signal(sig,stopped) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        with (LOCAL/'worker.log').open('xb') as log:
            child=subprocess.Popen(['taskset','-c','17',sys.executable,'-u','-m','experiments.run_active_fixture_gaussian_nh80_cost_v02','worker','--sha',digest],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);e['worker_pid']=child.pid;write(LOCAL/'execution.json',e)
            while child.poll() is None:
                guard('controller')
                if prior.shared.owned_RSS(child)>LIMITS['worker_RSS_cap_bytes'] or time.monotonic()-begin>LIMITS['wall_cap_s']:raise TimeoutError('Owned600s/4GiB cap')
                time.sleep(.1)
            e['worker_exit_code']=child.wait()
    except BaseException as error:e['failure']=repr(error)
    finally:
        if child is not None:e['cleanup_exit_code']=prior.shared.old.cost.cleanup(child)
        for sig,handler in previous.items():signal.signal(sig,handler)
        path=LOCAL/'result-LOCAL.json'
        if path.exists():
            result=json.loads(path.read_text());e.update(completed=bool(result['completed'] and not e.get('failure') and e.get('worker_exit_code')==0 and e.get('cleanup_exit_code')==0),physical_passed=bool(result.get('all_physical_passed',False)),result=record(path));cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(sha(path)+'.json')
            if cache.exists() and cache.read_bytes()!=path.read_bytes():raise ValueError('Immutable cache mismatch')
            if not cache.exists():cache.parent.mkdir(parents=True,exist_ok=True);cache.write_bytes(path.read_bytes())
            e['external_cache']=record(cache)
        e.update(terminal=True,wall_s=time.monotonic()-begin);write(LOCAL/'execution.json',e)
    return e

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','run']);parser.add_argument('--sha');a=parser.parse_args()
    if a.action=='prepare':print(json.dumps(prepare()))
    elif a.action=='worker':print(json.dumps(dict(completed=worker(a.sha)['completed'])))
    else:r=run(a.sha);print(json.dumps(r));raise SystemExit(0 if r['completed'] else 1)
