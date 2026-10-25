"""SOURCE-only six NEW balanced-held NH80 refinement controls.

Requires closed independent balanced36 numerical AND adequacy PASS before
preparation/release. No import-time data/model calls; no full catalog queue.
"""
import argparse,ast,json,os,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import norm as sparse_norm
from experiments import campaign_fixture_balanced_held_v01 as bank
from experiments import run_fixture_balanced_decision_screen_v01 as screen
from experiments import run_active_fixture_gaussian_refinement_v03 as original
from experiments import run_active_fixture_mechanics_screen_v01 as prior
from experiments import run_active_fixture_gaussian_nh80_cost_v02 as predecessor
from experiments import campaign_active_fixture_resolution_v03 as resolution
from experiments.campaign_exact_rest_pcg_v01 import ExactRestPCG
from experiments.campaign_surface_preconditioned_p2_v01 import PreconditionedP2Reference
from experiments.campaign_generalized_stability_v01 import generalized_stability
from experiments import run_clamped_surface_p2_qualification_v03 as metrics
from experiments.run_clamped_surface_p2_qualification_v01 import reaction_diagnostics,spatial_gate

ROOT=prior.ROOT;LOCAL=ROOT/'runs/operator-dev/fixture-balanced-held-nh-refinement-v01'
SOURCE='experiments/run_fixture_balanced_held_nh_refinement_v01.py';TEST='tests/test_fixture_balanced_held_nh_refinement_v01.py'
sha,record,write=prior.sha,prior.record,prior.write
PARAMETERS=((2000.,5.),(2000.,30.),(8000.,5.),(8000.,30.),(4000.,float(np.sqrt(150.))))
LEVELS=(3,6,12);PILOT_LAYOUT=bank.LAYOUTS[0]['id'];PILOT_PARAMETERS=(2000.,5.)
PILOT_LOADS=('held-oblique-positiveY-down','held-oblique-negativeY-down')
LIMITS=dict(worker_CPU=17,controller_CPU=18,threads=1,GPU=False,wall_cap_s=1800,worker_RSS_cap_bytes=24*1024**3,controller_RSS_cap_bytes=4*1024**3,aggregate_RSS_cap_bytes=28*1024**3,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=512*1024**2)


def resource_projection():
    return dict(observed_original_result_SHA256='dfddef7f1e86b259b7c15ed535b09e549b7ae0dd0e3ce621b0e3684bc3329894',observed_original_wall_s=1120.49,observed_original_peak_RSS_bytes=13802950656,observed_original_n12_initialization_s=478.432871,observed_original_n12_rest_LU_s=469.115644,observed_original_n12_two_solve_s=401.213539,observed_original_n12_two_check_s=151.534227,estimated_same_geometry_cost_s=1120.49,worker_margin=LIMITS['worker_RSS_cap_bytes']/13802950656,uncertainty='Different held forces can alter nonlinear iterations and stability-check cost. Same hard1800s/24GiBworker+4GiBcontroller/28GiBaggregate authoritative. No retries, force tuning or automatic expansion.')


def plan():
    return dict(schema='fixture-balanced-held-nh-refinement-v01',resources=LIMITS,bank_design=bank.design(),pilot=dict(layout=PILOT_LAYOUT,E_Pa=PILOT_PARAMETERS[0],K_N_m=PILOT_PARAMETERS[1],loads=list(PILOT_LOADS),levels=list(LEVELS),new_equilibria=6,selection='Prospective same softcorner/layout0, two NEW balanced held forces; all six fields genuinely new. Existing prefix/candidate NH controls and all old negatives immutable.'),gates=dict(metrics.GATES),weak_prefix_excitation_exemption=False,warm_start='Exact same-load/same-layout/E/K linear equilibrium at each mesh; no saved nonlinear warm start.',Newton_iterations=40,force_tolerance_l2_N=1e-10,quadrature_points=80,quadrature_scope='Unchanged log-NH80 bulk, normalized Gaussian traction/PSD fixture16vs20 guards. PositiveKeast24 is common physical volume metric, not the NH energy rule. NH24vs80 production admission remains untested.',J_scope='Sampled vertices/edges/24+80 guard; no whole-element determinant certificate.',spatial_scope='Same commonfinest12 positiveKeast24 coordinate RMS; p>=1 conditional fine and2delta medium indicators<=20um; actual query16 max-coordinate n6to12<=20um separately. Conditional indicators are not rigorous continuum error bounds.',qualification='Complete physical checks AND all6 original excitation/load eligibility AND both original spatial gates. Pilot-qualified is not bank admission.',cost_projection=resource_projection(),scope='DEVELOPMENT six-field reference pilot only; no posterior/noise/truth draws, policy claims or full660 execution. All negatives retained, no gate/force tuning.')


def sources():
    todo=[SOURCE,TEST,'experiments/audit_fixture_balanced_held_nh_refinement_v01.py','tests/test_audit_fixture_balanced_held_nh_refinement_v01.py','tests/test_active_fixture_gaussian_load_v02.py','tests/test_active_fixture_resolution_v03.py'];seen=set()
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


def balanced_origin():
    """Metadata-only prerequisite. Refuse before any current outcome read."""
    base=screen.LOCAL;e=json.loads((base/'execution.json').read_text())
    if not(e.get('terminal')is True and e.get('completed')is True and 'failure'not in e):raise ValueError('Balanced36 must be clean terminal before independent adequacy/source metadata')
    digest=sha(base/'protocol.json');p=screen.validate(digest)
    review=json.loads((base/'independent-terminal-review.json').read_text());closure=json.loads((base/'root-closure.json').read_text());summary=review['summary']
    if not(summary.get('integrity_passed')is True and summary.get('execution_clean')is True and summary.get('completed_case_records')==36 and summary.get('qualified_category_references')==144 and summary.get('completed_category_references')==144 and summary.get('completed_decision_banks')==180 and summary.get('independently_recomputed_rankings')==36 and summary.get('completed_utility_jobs')==1728 and summary['adequacy'].get('qualified_cases')==36 and summary['adequacy'].get('adequacy_qualified')is True and e.get('adequacy')==summary['adequacy']):raise ValueError('Independent all36 numerical AND adequacy PASS prerequisite; no cherry picking')
    audit_sources={k:sha(ROOT/k) for k in('experiments/audit_fixture_balanced_decision_screen_v01.py','tests/test_audit_fixture_balanced_decision_screen_v01.py')}
    if review.get('source_SHA256')!=audit_sources or review.get('protocol')!=record(base/'protocol.json')or review.get('release')!=record(base/'release.json')or review.get('execution')!=record(base/'execution.json')or review.get('root_closure')!=record(base/'root-closure.json'):raise ValueError('Exact independent review/source/protocol/release/rootclosure bytes')
    expected=[e['controller_pid'],e['builder']['pid'],*[v['pid']for v in e['workers']]]
    if not(closure.get('root_exit_code')==0 and closure.get('root_reaped')is True and closure.get('all_owned_PID_and_groups_absent')is True and closure.get('protocol_SHA256')==digest and closure.get('source_SHA256')==p['source_SHA256']and closure.get('execution_SHA256')==sha(base/'execution.json')and len(closure.get('owned_pids',[]))==len(expected)and set(closure['owned_pids'])==set(expected)):raise ValueError('Root exact owned closure before any pilot field')
    for v in[e['builder'],*e['workers']]:
        if not(v.get('reaped')is True and v.get('exit_code')==0 and v.get('cleanup_exit_code')==0):raise ValueError('All builder/workers reaped0/cleanup0')
    for pid in expected:screen.absent(pid)
    screen.verify_records(review['scientific_scalar_bindings'])
    return dict(protocol=record(base/'protocol.json'),release=record(base/'release.json'),execution=record(base/'execution.json'),independent_review=record(base/'independent-terminal-review.json'),root_closure=record(base/'root-closure.json'),independent_audit_source_SHA256=audit_sources,all36_numerical_and_adequacy_passed=True,old_negative_preserved=True)


def guard(role):
    if set(os.sched_getaffinity(0))!={LIMITS[role+'_CPU']} or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact CPU17worker/18controller/one-thread/noGPU')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>LIMITS[role+'_RSS_cap_bytes']:raise MemoryError('Own24GiBworker/4GiBcontroller RSS')
    available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    if available<LIMITS['RAM_reserve_bytes'] or shutil.disk_usage(ROOT).free<LIMITS['disk_reserve_bytes']+LIMITS['maximum_writes_bytes']:raise MemoryError('30GiB RAM/disk plus projected write reserve')
    if LOCAL.exists() and sum(p.stat().st_size for p in LOCAL.rglob('*') if p.is_file())>LIMITS['maximum_writes_bytes']:raise OSError('Own512MiB output cap')


def prepare():
    guard('controller')
    if (LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    origin=balanced_origin()
    p=dict(plan=plan(),source_SHA256=sources(),origin=origin,status='FROZEN_ROOT_RELEASE_REQUIRED')
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text());r=json.loads((LOCAL/'release.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['plan']!=plan() or p['source_SHA256']!=sources() or p['origin']!=balanced_origin():raise ValueError('Frozen design/source/origin')
    if r.get('approved') is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('origin')!=p['origin']:raise ValueError('Root release before physics')
    return p


def admissibility(deformation,R,clamp,equilibrium,stability,load):
    if load not in PILOT_LOADS:raise ValueError('Exactly two declared held loads; no weak-prefix exemption')
    return original.admissibility(deformation,R,clamp,equilibrium,stability,load)


def classification(rows,spatial):
    members={(n,label)for n in LEVELS for label in PILOT_LOADS}
    complete=len(rows)==6 and{(r['n'],r['label'])for r in rows}==members and all(r.get('completed')is True for r in rows)
    physical=complete and all(r['physical_passed']for r in rows)
    eligible=complete and all(r['load_eligible']and r['excitation_goal_met']and not r['weak_prefix_control']for r in rows)
    space=set(spatial)==set(PILOT_LOADS)and all(r['spatial_development_passed']for r in spatial.values())
    return dict(all6_retained=bool(complete),all_physical_passed=bool(physical),all_load_eligible=bool(eligible),both_spatial_passed=bool(space),passed=bool(physical),pilot_qualified=bool(physical and eligible and space),bank_admitted=False)


def surface_checks(f,loads):
    W20,_=bank.bank.consistent_fixture(f.X,f.faces,PILOT_LAYOUT,order=20)
    error=float(sparse_norm(f.W_scalar-W20)/sparse_norm(W20));ref,info=bank.load_catalog(f.n,order=20);production_info=bank.load_catalog(f.n)[1]
    load_error=max(float(np.max(abs(loads[k]-ref[k]))) for k in loads)
    # PSD follows from the explicit positive quadrature Gram, not a large dense eigensolve.
    if f.support_quadrature['normalization_error']>1e-10 or f.support_quadrature['first_moment_error_m']>1e-10 or error>1e-10 or load_error>1e-12 or any(x['net_force_error_N']>1e-12 or x['moment_error_Nm']>1e-13 for x in list(info.values())+list(production_info.values())) or any(x['normalization_error']>1e-10 or x['first_moment_error_m']>1e-10 for x in list(info.values())+list(production_info.values())):raise ValueError('Same physical normalized PSDsupport/knownload quadrature guards')
    return dict(support=f.support_quadrature,positive_quadrature_Gram_PSD=True,relative_support16_vs20=error,maximum_load16_vs20_N=load_error,refined_known_load_checks=info)


def spatial_results(fields):
    out={}
    for label in PILOT_LOADS:
        values=[(n,fields[n,label]) for n in LEVELS];d=resolution.common_rms(values,finest=12);g=spatial_gate(float(d[1,0]),float(d[2,1]));q=[resolution.field_queries(n,u,bank.measurement_points()) for n,u in values]
        delta=float(np.max(abs(q[2]-q[1])));coarse=float(np.max(abs(q[1]-q[0])))
        g.update(public_point_n3_n6_max_coordinate_m=coarse,public_point_n6_n12_max_coordinate_m=delta,public_point20um_passed=delta<=20e-6,spatial_development_passed=bool(g['spatial_passed'] and g['medium_adequate'] and delta<=20e-6),bank_admitted=False)
        out[label]=g
    return out


def worker(digest):
    guard('worker');p=validate(digest)
    if (LOCAL/'result-LOCAL.json').exists():raise FileExistsError('No retry')
    begin=time.monotonic();out=dict(completed=False,passed=False,protocol_SHA256=digest,release=record(LOCAL/'release.json'),worker_pid=os.getpid(),rows=[],meshes={},no_noise_or_truth_draws=True,bank_admitted=False)
    def save():out.update(wall_s=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'result-LOCAL.json',out)
    def timeout(*_):raise TimeoutError('Frozen1800s cap')
    def interrupted(*_):raise InterruptedError('Explicit worker interruption')
    previous={s:signal.signal(s,h) for s,h in ((signal.SIGALRM,timeout),(signal.SIGTERM,interrupted))};signal.alarm(LIMITS['wall_cap_s']);fields={}
    try:
        E,K=PILOT_PARAMETERS
        for n in LEVELS:
            guard('worker');start=time.monotonic();f=resolution.make_forward(n,PILOT_LAYOUT);loads,info=bank.load_catalog(n);checks=surface_checks(f,loads);rest=E*f.Kunit_free+K*f.W_free;directions=ExactRestPCG(rest);model=PreconditionedP2Reference.from_forward(f,E,K,quadrature_order=80,direction_solver=directions)
            out['meshes'][str(n)]=dict(initialization_s=time.monotonic()-start,rest_factorization_s=directions.rest_factorization_s,surface=checks,nodes=len(f.X),free_DOF=len(f.free));save()
            for label in PILOT_LOADS:
                out['pending_case']=dict(n=n,label=label,layout=PILOT_LAYOUT,E_Pa=E,K_N_m=K);save()
                guard('worker');start=time.monotonic();force=loads[label];uL=np.zeros_like(f.X);uL.ravel()[f.free]=directions.rest_solve(force.ravel()[f.free]);linear_R=float(np.linalg.norm(((E*f.Kunit+K*f.W)@uL.ravel()-force.ravel())[f.free]))
                if linear_R>1e-10 or np.any(uL[f.fixed]!=0):raise ValueError('Exact linear warm-start R/clamp failed')
                linear_s=time.monotonic()-start;start=time.monotonic();u,diag=model.solve(force,initial=uL,maximum_iterations=40,force_tolerance_N=1e-10);solve_s=time.monotonic()-start;start=time.monotonic();state=model.evaluate(u,force,tangent=True);R=float(np.linalg.norm(state['residual'].ravel()[f.free]));d=resolution.deformation(f,u,finest=12);eq=reaction_diagnostics(f,u,force,E,K,state['residual']);stable=generalized_stability(state['tangent'][f.free][:,f.free],rest,directions.rest_LU.solve);status=admissibility(d,R,np.all(u[f.fixed]==0),eq,stable,label)
                target=LOCAL/'raw-LOCAL'/f'n{n}-{label}.npz';target.parent.mkdir(exist_ok=True);np.savez_compressed(target,displacement_m=u,linear_displacement_m=uL,rest_m=f.X,force_N=force,elements=f.elements,exact_constraint_nodes=f.fixed,query_points_m=bank.measurement_points(),query_displacement_m=resolution.field_queries(n,u,bank.measurement_points()))
                row=dict(n=n,label=label,layout=PILOT_LAYOUT,E_Pa=E,K_N_m=K,completed=True,**status,linear_original_R_l2_N=linear_R,original_free_R_l2_N=R,deformation=d,equilibrium=eq,stability=stable,Newton=diag,capture=record(target),known_load=info[label],actual_deformed_wrench=prior.force_wrench(f.X,force,u),linear_warm_s=linear_s,NH80_solve_s=solve_s,independent_checks_and_capture_s=time.monotonic()-start,current_LU_fallbacks_cumulative=directions.current_fallback_factorizations)
                out['rows'].append(row);fields[n,label]=u;save();guard('worker')
        out.update(completed=True,spatial=spatial_results(fields));out.update(classification(out['rows'],out['spatial']));out['scope']=p['plan']['scope'];save()
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
            child=subprocess.Popen(['taskset','-c','17',sys.executable,'-u','-m','experiments.run_fixture_balanced_held_nh_refinement_v01','worker','--sha',digest],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);e['worker_pid']=child.pid;write(LOCAL/'execution.json',e)
            while child.poll() is None:
                guard('controller')
                rss=prior.shared.owned_RSS(child);own=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;e['peak_aggregate_RSS_bytes']=max(e.get('peak_aggregate_RSS_bytes',0),rss+own)
                if rss>LIMITS['worker_RSS_cap_bytes']or rss+own>LIMITS['aggregate_RSS_cap_bytes']:raise MemoryError('Owned24GiBworker/4GiBcontroller/28GiBaggregate cap')
                if time.monotonic()-begin>LIMITS['wall_cap_s']:raise TimeoutError('Owned1800s cap')
                time.sleep(.1)
            e['worker_exit_code']=child.wait()
    except BaseException as error:e['failure']=repr(error)
    finally:
        if child is not None:e['cleanup_exit_code']=prior.shared.old.cost.cleanup(child)
        for sig,handler in previous.items():signal.signal(sig,handler)
        path=LOCAL/'result-LOCAL.json'
        if path.exists():
            result=json.loads(path.read_text());e.update(completed=bool(result['completed'] and not e.get('failure') and e.get('worker_exit_code')==0 and e.get('cleanup_exit_code')==0),physical_passed=bool(result.get('all_physical_passed',False)),pilot_qualified=bool(result.get('pilot_qualified',False)and not e.get('failure')and e.get('worker_exit_code')==0 and e.get('cleanup_exit_code')==0),result=record(path));cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(sha(path)+'.json')
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
