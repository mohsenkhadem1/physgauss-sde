"""Prospective fixed72-row LINEAR development screen; no NH/policy admission.

The first declared center case at n3/n6 is a timing checkpoint IN the same
retained queue. No reruns, category filters, observation/noise/test draws.
"""
import argparse,ast,gc,json,os,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import splu,norm as sparse_norm
from experiments import campaign_fixture_gaussian_linear_screen_v01 as mechanics
from experiments import run_active_fixture_gaussian_nh80_cost_v02 as origin
from experiments.run_clamped_surface_p2_qualification_v01 import reaction_diagnostics

ROOT=origin.ROOT;LOCAL=ROOT/'runs/operator-dev/fixture-Gaussian-linear-screen-v01'
SOURCE='experiments/run_fixture_gaussian_linear_screen_v01.py';TEST='tests/test_run_fixture_gaussian_linear_screen_v01.py'
sha,record,write=origin.sha,origin.record,origin.write
RESOURCES=dict(worker_CPU=14,controller_CPU=15,threads=1,GPU=False,wall_cap_s=900,worker_RSS_cap_bytes=8*1024**3,controller_RSS_cap_bytes=4*1024**3,aggregate_RSS_cap_bytes=12*1024**3,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=256*1024**2)


def sources():
    todo=[SOURCE,TEST,'tests/test_fixture_gaussian_linear_screen_v01.py'];seen=set()
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


def origin_gate():
    # CLOSED v02 only. Never open any running v03 execution/result/field.
    base=origin.LOCAL;digest=sha(base/'protocol.json');origin.validate(digest)
    e=json.loads((base/'execution.json').read_text())
    if not e.get('terminal') or not e.get('completed') or e.get('worker_exit_code')!=0 or e.get('cleanup_exit_code')!=0:raise ValueError('Closed Gaussian-v02 origin required')
    if e['result']!=record(base/'result-LOCAL.json') or e['result']['sha256']!='0ca4a79bc870a13bbe709bdac26a455f8732c2dc6b575cfd348f8ce46b8400fe':raise ValueError('Frozen physical-pass/point-negative Gaussian-v02')
    review=json.loads((base/'independent-review.json').read_text())
    if review.get('audit_passed') is not True or review.get('scientific_admission') is not False or review['source_result']!=e['result']:raise ValueError('Independent original negative retained')
    origin.prior.shared.old.absent(e['controller_pid']);origin.prior.shared.old.absent(e['worker_pid'])
    return dict(protocol=record(base/'protocol.json'),release=record(base/'release.json'),execution=record(base/'execution.json'),result=e['result'],review=record(base/'independent-review.json'),live_v03_used=False,scope='Origin is retained negative, NOT an NH/mesh admission prerequisite.')


def plan():
    return dict(schema='fixture-Gaussian-linear-development-screen-v01',resources=RESOURCES,design=mechanics.bank.design(),worklist=mechanics.worklist(),states=72,loads_per_state=11,load_responses=792,first_paired_timing_checkpoint_indices=[0,1],full_queue_unconditional=True,modal='Exact Schur condensation of foundation-free DOFs, retainALLboundary modes AND exact1/E particular; explicit resolution.point_query_matrix. No thresholds/truncation.',modal_parity_tolerance_m=2e-12,free_residual_l2_N=1e-10,surface_support_relative_Frobenius=1e-10,surface_load_absolute_N=1e-12,linear_vs_NH='No equivalence/admission claimed; sampledJ/strain/NR are descriptive linear-field geometry only.',resource_projection=dict(n6_full_free_DOF=23829,n6_foundation_boundary_DOF=1833,n6_interior_DOF=21996,one_dense_elimination_workspace_bytes=21996*1833*8,two_mesh_models_per_layout_only=True,eight_models_not_retained=True,actual_wall_RSS_guards_authoritative=True,first_pair_cost_is_not_case_selection=True),scope='Fixed DEVELOPMENT mechanics/identifiability screen only; no NH/continuum/calibratedinverse/policy/adaptivewin claim. All4layouts/72states/792responses retained or explicitlyqueued/incomplete; no draws/filter/retry.')


def guard(role):
    if set(os.sched_getaffinity(0))!={RESOURCES[role+'_CPU']} or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('ExactCPU14worker/15controller/one-thread/noGPU')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES[role+'_RSS_cap_bytes']:raise MemoryError('Own8GiBworker/4GiBcontroller RSS')
    available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    if available<RESOURCES['RAM_reserve_bytes'] or shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']+RESOURCES['maximum_writes_bytes']:raise MemoryError('30GiB RAM/disk plus projectedwrite reserve')
    if LOCAL.exists() and sum(p.stat().st_size for p in LOCAL.rglob('*') if p.is_file())>RESOURCES['maximum_writes_bytes']:raise OSError('Own256MiB output cap')


def prepare():
    guard('controller')
    if (LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    p=dict(plan=plan(),source_SHA256=sources(),origin=origin_gate(),status='FROZEN_ROOT_RELEASE_REQUIRED');write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text());r=json.loads((LOCAL/'release.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['plan']!=plan() or p['source_SHA256']!=sources() or p['origin']!=origin_gate():raise ValueError('Frozen exact72source/design/origin')
    if r.get('approved') is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']:raise ValueError('Explicitrootrelease before development solves')
    return p


def surface_checks(f,loads,info):
    W20,_=mechanics.bank.consistent_fixture(f.X,f.faces,f.support_quadrature['layout'],order=20)
    relative=float(sparse_norm(f.W_scalar-W20)/sparse_norm(W20));ref,more=mechanics.geometry.load_catalog(f.n,order=20)
    error=max(float(np.max(abs(loads[name]-ref[name]))) for name in loads)
    if relative>1e-10 or sparse_norm(f.W_scalar-f.W_scalar.T)>1e-12*max(sparse_norm(f.W_scalar),1e-300) or f.support_quadrature['normalization_error']>1e-10 or f.support_quadrature['first_moment_error_m']>1e-10 or error>1e-12 or any(d['normalization_error']>1e-10 or d['first_moment_error_m']>1e-10 or d['net_force_error_N']>1e-12 or d['moment_error_Nm']>1e-13 for d in list(info.values())+list(more.values())):raise ValueError('Unchanged normalized PSDsupport/knownload16vs20 guards')
    return dict(support=f.support_quadrature,support16_vs20_relative_Frobenius=relative,support_PSD_by_positive_Gram_quadrature=True,no_dense_W_eigenanalysis=True,maximum_load16_vs20_N=error,known_load_checks=info)


def save_raw(rows):
    if not rows:return None
    tmp=LOCAL/'responses-prefix-LOCAL.tmp.npz';target=LOCAL/'responses-LOCAL.npz'
    np.savez_compressed(tmp,responses=np.array([r['response'] for r in rows]),jacobians=np.array([r['jacobian'] for r in rows]),field_queries=np.array([r['field_queries'] for r in rows]),n=np.array([r['n'] for r in rows]),layout=np.array([r['layout'] for r in rows]),E=np.array([r['E_Pa'] for r in rows]),K=np.array([r['K_N_m'] for r in rows]),measurement_labels=np.array([s['id'] for s in mechanics.bank.measurement_specs()]),load_labels=np.array([s['id'] for s in mechanics.bank.load_specs()]),query_points=mechanics.bank.measurement_points());tmp.replace(target);return record(target)


def worker(digest):
    guard('worker');p=validate(digest)
    if (LOCAL/'result-LOCAL.json').exists():raise FileExistsError('No retry')
    start=time.monotonic();queue=[dict(row,status='queued') for row in p['plan']['worklist']];out=dict(completed=False,passed=False,protocol_SHA256=digest,release=record(LOCAL/'release.json'),worker_pid=os.getpid(),planned_states=72,expected_load_responses=792,queue=queue,rows=[],models=[],no_noise_truth_or_test_draws=True,NH_solves=0,bank_admitted=False)
    raw=[];cache={};current_layout=None
    def save():out.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,completed_states=len(out['rows']),completed_load_responses=11*len(out['rows']));write(LOCAL/'result-LOCAL.json',out)
    def deadline(*_):raise TimeoutError('Frozen900s developmentcap')
    def interrupted(*_):raise InterruptedError('Explicit worker interruption')
    previous={sig:signal.signal(sig,fn) for sig,fn in ((signal.SIGALRM,deadline),(signal.SIGTERM,interrupted))};signal.alarm(900)
    try:
        labels=[s['id'] for s in mechanics.bank.measurement_specs()];load_ids=[s['id'] for s in mechanics.bank.load_specs()];query=mechanics.bank.measurement_points()
        for row in queue:
            guard('worker');row['status']='started';save();n=row['n'];layout=row['layout'];E=row['E_Pa'];K=row['K_N_m']
            if current_layout!=layout:cache.clear();gc.collect();current_layout=layout
            if n not in cache:
                begin=time.monotonic();f=mechanics.geometry.make_forward(n,layout);loads,info=mechanics.geometry.load_catalog(n);surface=surface_checks(f,loads,info);construction=time.monotonic()-begin;begin=time.monotonic();model=mechanics.make_measurement_modal(f,loads);modal_s=time.monotonic()-begin;cache[n]=(f,loads,model)
                out['models'].append(dict(n=n,layout=layout,geometry_surface_s=construction,modal_construction_s=modal_s,all_modes=model.modal.diagnostics,surface=surface));save();guard('worker')
            f,loads,model=cache[n];begin=time.monotonic();force=np.array([loads[k] for k in load_ids]);rhs=force.reshape(11,-1).T;A=E*f.Kunit_free+K*f.W_free;lu=splu(A);u=np.zeros_like(rhs);u[f.free]=lu.solve(rhs[f.free]);residual=(E*f.Kunit+K*f.W)@u-rhs;R=float(np.linalg.norm(residual[f.free],axis=0).max());fields=u.T.reshape(11,len(f.X),3)
            if not np.isfinite(fields).all() or R>1e-10 or np.any(fields[:,f.fixed]!=0):raise ValueError('OriginallinearR/exactclamp failed')
            selected={k:fields[j:j+1] for j,k in enumerate(load_ids)};value=mechanics.measurement_response(n,selected)[0,:,0];unit=np.log([E/2000,K/5])/np.log([4,6]);parity=float(np.max(abs(model.select(labels)(unit[None])[0,:,0]-value)))
            if parity>2e-12:raise ValueError('Allmode/direct parity2pm failed')
            derivatives=[]
            for D in (E*f.Kunit,K*f.W):
                v=np.zeros_like(u);v[f.free]=lu.solve(-(D@u)[f.free]);vf=v.T.reshape(11,len(f.X),3);derivatives.append(mechanics.measurement_response(n,{k:vf[j:j+1] for j,k in enumerate(load_ids)})[0,:,0])
            jac=np.stack(derivatives,axis=1)*np.log([4,6])[None];geom=mechanics.bank.core.strain_metrics(f,fields);nr=mechanics.bank.nonrigid_rms(f,fields);equilibrium=[reaction_diagnostics(f,fields[j],force[j],E,K) for j in range(11)]
            if not all(d['passed'] for d in equilibrium):raise ValueError('Originallinear reaction/resultant/restmoment guard')
            force_moment=[dict(resultant_norm_N=float(np.linalg.norm(force[j].sum(0))),actual_deformed_moment_norm_Nm=float(np.linalg.norm(np.cross(f.X+fields[j]-mechanics.bank.ORIGIN_M,force[j]).sum(0)))) for j in range(11)]
            record_row=dict(**{k:row[k] for k in ('index','n','layout','E_Pa','K_N_m')},free_residual_l2_N=R,exact_clamp_zeros=True,modal_parity_m=parity,nonrigid_Euclidean_RMS_m=nr.tolist(),nonrigid_coordinate_RMS_m=(nr/np.sqrt(3)).tolist(),NR_units='Euclidean=sqrt(integral||nonrigid||²/V); coordinate=Euclidean/sqrt3',minimum_sampled_linear_J=geom['minimum_J'].tolist(),maximum_linear_symmetric_strain=geom['maximum_absolute_symmetric_strain_eigenvalue'].tolist(),equilibrium=equilibrium,known_deformed_wrench_norms=force_moment,linear_solve_and_diagnostic_s=time.monotonic()-begin)
            out['rows'].append(record_row);raw.append(dict(row,response=value,jacobian=jac,field_queries=mechanics.geometry.field_queries(n,fields,query)));row['status']='completed';out['raw']=save_raw(raw)
            if len(out['rows'])==2:out['first_fixed_paired_case_cost_checkpoint']=dict(indices=[0,1],wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,remaining_states=70,estimated_full_wall_linear_multiple_s=(time.monotonic()-start)*36,projection_only=True,queue_unchanged=True)
            save();guard('worker')
        out['layout_separation']=mechanics.separation(raw);pairs=[]
        for a,b in zip(raw[::2],raw[1::2]):
            if (a['layout'],a['E_Pa'],a['K_N_m'])!=(b['layout'],b['E_Pa'],b['K_N_m']) or (a['n'],b['n'])!=(3,6):raise ValueError('Exactpairedmembership')
            pairs.append(dict(layout=a['layout'],E_Pa=a['E_Pa'],K_N_m=a['K_N_m'],maximum_common_point_coordinate_difference_m=float(np.max(abs(a['field_queries']-b['field_queries']))),maximum_measurement_difference_m=float(np.max(abs(a['response']-b['response'])))))
        out.update(completed=True,passed=True,all72_retained=True,all792_load_responses_retained=True,paired_mesh_differences=pairs,scope=p['plan']['scope']);save()
    except BaseException as error:out['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());save();raise
    finally:
        signal.alarm(0)
        for sig,handler in previous.items():signal.signal(sig,handler)
        save()
    return out


def run(digest):
    guard('controller');validate(digest)
    if (LOCAL/'execution.json').exists():raise FileExistsError('No retry')
    begin=time.monotonic();child=None;e=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'))
    def stop(*_):raise InterruptedError('Explicitcontrollerinterruption')
    previous={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        with (LOCAL/'worker.log').open('xb') as log:
            child=subprocess.Popen(['taskset','-c','14',sys.executable,'-u','-m','experiments.run_fixture_gaussian_linear_screen_v01','worker','--sha',digest],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);e['worker_pid']=child.pid;write(LOCAL/'execution.json',e)
            while child.poll() is None:
                guard('controller')
                if origin.prior.shared.owned_RSS(child)>RESOURCES['worker_RSS_cap_bytes'] or time.monotonic()-begin>900:raise TimeoutError('Owned900s/8GiBworker cap')
                time.sleep(.1)
            e['worker_exit_code']=child.wait()
    except BaseException as error:e['failure']=repr(error)
    finally:
        if child is not None:e['cleanup_exit_code']=origin.prior.shared.old.cost.cleanup(child)
        for sig,handler in previous.items():signal.signal(sig,handler)
        path=LOCAL/'result-LOCAL.json'
        if path.exists():
            r=json.loads(path.read_text());e.update(completed=bool(r.get('completed') and r.get('passed') and not e.get('failure') and e.get('worker_exit_code')==0 and e.get('cleanup_exit_code')==0),result=record(path));cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(sha(path)+'.json')
            if cache.exists() and cache.read_bytes()!=path.read_bytes():raise ValueError('Immutablecachemismatch')
            if not cache.exists():cache.parent.mkdir(parents=True,exist_ok=True);cache.write_bytes(path.read_bytes())
            e['external_cache']=record(cache)
        e.update(terminal=True,wall_s=time.monotonic()-begin);write(LOCAL/'execution.json',e)
    return e

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','run']);p.add_argument('--sha');a=p.parse_args()
    if a.action=='prepare':print(json.dumps(prepare()))
    elif a.action=='worker':print(json.dumps(dict(completed=worker(a.sha)['completed'])))
    else:r=run(a.sha);print(json.dumps(r));raise SystemExit(0 if r['completed'] else 1)
