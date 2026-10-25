"""Bounded deterministic LINEAR development screen, no calibration or draws.

New geometry/operator bank only. All4 layouts ×3E×3K ×11 public loads on n1/n2
are retained. No nonlinear solve, noise, posterior or test object is generated.
Root release after fixed source/protocol is mandatory before the first solve.
"""
import argparse,ast,json,os,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import splu
from experiments import campaign_active_fixture_bank_v01 as bank
from experiments import run_surface_nonlinear_posterior_cost_v03 as shared
ROOT=shared.ROOT;LOCAL=ROOT/'runs/operator-dev/active-fixture-mechanics-screen-v01'
SOURCE='experiments/run_active_fixture_mechanics_screen_v01.py';TEST='tests/test_active_fixture_mechanics_screen_v01.py'
E_VALUES=(2000.,4000.,8000.);K_VALUES=(5.,float(np.sqrt(150.)),30.)
RESOURCES=dict(worker_CPU=17,controller_CPU=18,worker_RSS_cap_bytes=4*1024**3,controller_RSS_cap_bytes=4*1024**3,threads=1,GPU=False,wall_cap_s=120,aggregate_RSS_cap_bytes=8*1024**3,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=128*1024**2)
sha,record,write=shared.sha,shared.record,shared.write


def sources():
    todo=[SOURCE,TEST,'tests/test_active_fixture_bank_v01.py'];seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:modules=[node.module or '']+[(node.module+'.'+v.name)for v in node.names if node.module]
            for module in modules:
                path=module.replace('.','/')+'.py'
                if(ROOT/path).is_file()and path not in seen:todo.append(path)
    return {f:sha(ROOT/f)for f in sorted(seen)}


def guard(role):
    if set(os.sched_getaffinity(0))!={RESOURCES[role+'_CPU']}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('ExactCPU17worker/18controller/one-thread/noGPU')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES[role+'_RSS_cap_bytes']:raise MemoryError('Own4GiB RSS')
    if int(next(line for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')).split()[1])*1024<RESOURCES['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise MemoryError('30GiB hostRAM/disk reserve')
    if sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise OSError('Own128MiB output cap')


def prepare():
    guard('controller')
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    p=dict(schema='active-fixture-linear-mechanics-screen-v01',status='FROZEN_ROOT_RELEASE_REQUIRED',source_SHA256=sources(),resources=RESOURCES,design=bank.design(),mesh_n=[1,2],E_Pa=list(E_VALUES),K_N_m=list(K_VALUES),states=72,load_responses=792,random_draws=0,nonlinear_solves=0,modal_parity_tolerance_m=2e-12,physical_residual_N=1e-10,scope='Deterministic development screen only; no NH/continuum/inference/adaptive-policy/calibration admission. Retain all under/overexcitation and mesh discrepancies; no force search.')
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=RESOURCES or p['design']!=bank.design()or p['mesh_n']!=[1,2]or p['E_Pa']!=list(E_VALUES)or p['K_N_m']!=list(K_VALUES):raise ValueError('Exact source/design/resources/protocol')
    r=json.loads((LOCAL/'release.json').read_text())
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']:raise ValueError('Explicit root release before any development solve')
    return p


def force_wrench(X,force,u):
    return dict(resultant_N=force.sum(0).tolist(),moment_at_deformed_nodes_Nm=np.cross(X+u-bank.ORIGIN_M,force).sum(0).tolist())


def worker(digest):
    guard('worker');p=validate(digest)
    if(LOCAL/'result-LOCAL.json').exists():raise FileExistsError('No actual retry')
    start=time.monotonic();state=dict(completed=False,passed=False,pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),meshes={},rows=[],new_RNG_calls=0,nonlinear_solves=0)
    def save():state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'result-LOCAL.json',state)
    def stop(*_):raise TimeoutError('Frozen120s development cap')
    def interrupted(*_):raise InterruptedError('Explicit owned worker interruption')
    previous={s:signal.signal(s,h)for s,h in((signal.SIGALRM,stop),(signal.SIGTERM,interrupted))};signal.alarm(120)
    raw=[]
    try:
        labels=[s['id']for s in bank.measurement_specs()];load_ids=[s['id']for s in bank.load_specs()];query=bank.measurement_points()
        for n in p['mesh_n']:
            begin=time.monotonic();loads,linfo=bank.load_catalog(n);refloads,_=bank.load_catalog(n,order=20)
            if any(d['normalization_error']>1e-10 or d['net_force_error_N']>1e-12 or d['moment_error_Nm']>1e-13 for d in linfo.values())or any(np.max(abs(loads[k]-refloads[k]))>1e-12 for k in loads):raise ValueError('Unchanged known-force normalization/wrench/quadrature guards')
            state['meshes'][str(n)]=dict(loads=linfo,layouts={},initialization_s=time.monotonic()-begin);save()
            for spec in bank.LAYOUTS:
                begin=time.monotonic();f=bank.make_forward(n,spec['id']);W20,_=bank.consistent_fixture(f.X,f.faces,spec['id'],order=20);norm=lambda A:float(np.sqrt(np.sum(A.data**2)));qerror=norm(f.W_scalar-W20)/norm(W20);eig=np.linalg.eigvalsh(f.W_scalar.toarray());scale=max(float(abs(eig).max()),1e-30)
                if f.support_quadrature['normalization_error']>1e-10 or qerror>1e-10 or eig[0]<-64*np.finfo(float).eps*scale:raise ValueError('Consistent fixture normalization/PSD/quadrature fail')
                model=bank.make_measurement_modal(f,loads);loadarray=np.array([loads[k]for k in load_ids]);rhs=loadarray.reshape(11,-1).T;fields_by_state=[]
                layout_meta=dict(support=f.support_quadrature,support16_vs20_relative_Frobenius=qerror,minimum_W_eigenvalue=float(eig[0]),PSD_roundoff_tolerance=64*np.finfo(float).eps*scale,all_modes=model.modal.diagnostics,construction_s=time.monotonic()-begin)
                state['meshes'][str(n)]['layouts'][spec['id']]=layout_meta
                for E in E_VALUES:
                    for K in K_VALUES:
                        begin=time.monotonic();A=E*f.Kunit_free+K*f.W_free;factor=splu(A);u=np.zeros_like(rhs);u[f.free]=factor.solve(rhs[f.free]);residual=(E*f.Kunit+K*f.W)@u-rhs;error=float(np.max(abs(residual[f.free])))
                        fields=u.T.reshape(11,len(f.X),3)
                        if not np.isfinite(fields).all()or error>1e-10 or np.any(fields[:,f.fixed]!=0):raise ValueError('Original free residual/clamp check failed')
                        values=bank.measurement_response(n,{k:fields[j:j+1]for j,k in enumerate(load_ids)})[0,:,0];unit=np.log(np.array([E/2000.,K/5.]))/np.log([4.,6.]);modal=model.select(labels)(unit[None])[0,:,0];parity=float(np.max(abs(values-modal)))
                        if parity>2e-12:raise ValueError('Exact all-mode callback/direct field parity')
                        du=[]
                        for derivative in(E*(f.Kunit@u),K*(f.W@u)):
                            v=np.zeros_like(u);v[f.free]=factor.solve(-derivative[f.free]);du.append(bank.measurement_response(n,{k:v.T.reshape(11,len(f.X),3)[j:j+1]for j,k in enumerate(load_ids)})[0,:,0])
                        jac=np.stack(du,axis=1)*np.log([4.,6.])[None,:]
                        # Explicit development surrogate only: actual proper-prior
                        # mixture acquisition is a separately controlled stage.
                        Jp=jac[:4];Jc=jac[4:12];Jt=jac[12:];C=np.linalg.inv(12*np.eye(2)+Jp.T@Jp/bank.SIGMA_M**2)
                        reductions=np.sum((Jt@C@Jc.T)**2,axis=0)/(bank.SIGMA_M**2+np.sum((Jc@C)*Jc,axis=1))/48
                        order=np.argsort(-reductions,kind='stable');winner=int(order[0]);margin=float((reductions[order[0]]-reductions[order[1]])/max(reductions[order[0]],1e-300))
                        fieldquery=bank.core.field_queries(n,fields,query);nr=bank.nonrigid_rms(f,fields);deformation=bank.core.strain_metrics(f,fields)
                        supportforce=-K*(f.W@u).T.reshape(fields.shape);reactions=residual.T.reshape(fields.shape);clamp=np.zeros_like(reactions);clamp[:,f.fixed]=reactions[:,f.fixed];forcebal=np.max(abs((loadarray+supportforce+clamp).sum(1)));mom=np.max(abs(np.cross(f.X[None]+fields-bank.ORIGIN_M,loadarray+supportforce+clamp).sum(1)))
                        row=dict(n=n,layout=spec['id'],E_Pa=E,K_N_m=K,development_local_risk_winner=bank.load_specs()[1+winner]['id'],development_local_risk_relative_margin=margin,local_risk_surrogate_only=True,free_residual_N=error,exact_clamp_zeros=True,modal_parity_m=parity,nonrigid_RMS_m=nr.tolist(),minimum_sampled_J=deformation['minimum_J'].tolist(),maximum_symmetric_strain=deformation['maximum_absolute_symmetric_strain_eigenvalue'].tolist(),linear_reaction_force_balance_N=float(forcebal),moment_balance_at_deformed_nodes_Nm=float(mom),known_deformed_wrenches=[force_wrench(f.X,loadarray[j],fields[j])for j in range(11)],seconds=time.monotonic()-begin)
                        state['rows'].append(row);raw.append(dict(n=n,layout=spec['id'],E=E,K=K,response=values,jacobian=jac,local_risk_reductions=reductions,field_queries=fieldquery));save();guard('worker')
        np.savez_compressed(LOCAL/'responses-LOCAL.npz',responses=np.array([r['response']for r in raw]),jacobians=np.array([r['jacobian']for r in raw]),local_risk_reductions=np.array([r['local_risk_reductions']for r in raw]),field_queries=np.array([r['field_queries']for r in raw]),n=np.array([r['n']for r in raw]),layout=np.array([r['layout']for r in raw]),E=np.array([r['E']for r in raw]),K=np.array([r['K']for r in raw]),measurement_labels=np.array(labels),load_labels=np.array(load_ids),query_points=query)
        mesh=[]
        for i in range(36):
            a,b=raw[i],raw[36+i];assert(a['layout'],a['E'],a['K'])==(b['layout'],b['E'],b['K']);mesh.append(dict(layout=a['layout'],E_Pa=a['E'],K_N_m=a['K'],maximum_common_point_difference_m=float(np.max(abs(a['field_queries']-b['field_queries']))),maximum_measurement_difference_m=float(np.max(abs(a['response']-b['response'])))))
        separation=[]
        for n in(1,2):
            for E in E_VALUES:
                for K in K_VALUES:
                    selected=[r for r in raw if(r['n'],r['E'],r['K'])==(n,E,K)]
                    for i,a in enumerate(selected):
                        for b in selected[i+1:]:separation.append(dict(n=n,E_Pa=E,K_N_m=K,layouts=[a['layout'],b['layout']],prefix_L2_over_sigma=float(np.linalg.norm(a['response'][:4]-b['response'][:4])/bank.SIGMA_M),candidate_RMS_difference_m=float(np.sqrt(np.mean((a['response'][4:12]-b['response'][4:12])**2))),held_target_RMS_difference_m=float(np.sqrt(np.mean((a['response'][12:]-b['response'][12:])**2)))))
        state.update(completed=True,passed=True,layout_separation=separation,raw=record(LOCAL/'responses-LOCAL.npz'),coarse_fine=mesh,all72_retained=len(state['rows'])==72,all792_load_responses_retained=True,scope=p['scope']);save()
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());save();raise
    finally:
        signal.alarm(0)
        for s,h in previous.items():signal.signal(s,h)
        save()
    return state


def completed_screen_gate(digest):
    p=validate(digest);e=json.loads((LOCAL/'execution.json').read_text());r=json.loads((LOCAL/'result-LOCAL.json').read_text())
    if not e.get('terminal')or not all(shared.phase_completion(e,r).values())or e.get('protocol_SHA256')!=digest or r.get('protocol_SHA256')!=digest or e['result']!=record(LOCAL/'result-LOCAL.json')or not r.get('all72_retained')or not r.get('all792_load_responses_retained')or record(ROOT/r['raw']['path'])!=r['raw']:raise ValueError('Complete closed72-state development screen and exact raw hashes required')
    if e.get('release')!=record(LOCAL/'release.json')or r.get('release')!=e['release']or e.get('external_cache')!=record(ROOT/e['external_cache']['path'])or (ROOT/e['external_cache']['path']).read_bytes()!=(LOCAL/'result-LOCAL.json').read_bytes():raise ValueError('Exact actual release/result/cache bytes')
    expected={(n,s['id'],E,K)for n in(1,2)for s in bank.LAYOUTS for E in E_VALUES for K in K_VALUES}
    if len(r['rows'])!=72 or {(v['n'],v['layout'],v['E_Pa'],v['K_N_m'])for v in r['rows']}!=expected or any(v['free_residual_N']>1e-10 or v['modal_parity_m']>2e-12 or v['exact_clamp_zeros']is not True for v in r['rows']):raise ValueError('Every declared development state and original numerical checks')
    shared.old.absent(e['controller_pid']);shared.old.absent(e['worker_pid'])
    return dict(protocol=record(LOCAL/'protocol.json'),release=record(LOCAL/'release.json'),execution=record(LOCAL/'execution.json'),result=record(LOCAL/'result-LOCAL.json'),raw=r['raw'],source_SHA256=p['source_SHA256'],wall_s=e['wall_s'],scope=p['scope'],development_only=True,all_numeric_checks_passed=True)


def run(digest):
    guard('controller');validate(digest)
    if(LOCAL/'execution.json').exists():raise FileExistsError('No actual retry')
    start=time.monotonic();child=None;e=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'))
    def stopped(*_):raise InterruptedError('Explicit owned controller interruption')
    previous={sig:signal.signal(sig,stopped)for sig in(signal.SIGTERM,signal.SIGINT)}
    try:
        with(LOCAL/'worker.log').open('xb')as log:
            child=subprocess.Popen(['taskset','-c','17',sys.executable,'-u','-m','experiments.run_active_fixture_mechanics_screen_v01','worker','--sha',digest],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);e['worker_pid']=child.pid;write(LOCAL/'execution.json',e)
            while child.poll()is None:
                guard('controller')
                if shared.owned_RSS(child)>RESOURCES['worker_RSS_cap_bytes']or time.monotonic()-start>120:raise TimeoutError('Owned120s/4GiB budget')
                time.sleep(.1)
            e['worker_exit_code']=child.wait()
    except BaseException as error:e['failure']=repr(error)
    finally:
        if child is not None:e['cleanup_exit_code']=shared.old.cost.cleanup(child)
        for sig,handler in previous.items():signal.signal(sig,handler)
        path=LOCAL/'result-LOCAL.json'
        if path.exists():
            result=json.loads(path.read_text());e.update(**shared.phase_completion(e,result),result=record(path))
            cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(sha(path)+'.json');cache.parent.mkdir(parents=True,exist_ok=True)
            if cache.exists()and cache.read_bytes()!=path.read_bytes():raise ValueError('Immutable metadata cache mismatch')
            if not cache.exists():cache.write_bytes(path.read_bytes())
            e['external_cache']=record(cache)
        e.update(terminal=True,wall_s=time.monotonic()-start);write(LOCAL/'execution.json',e)
    return e


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','run']);parser.add_argument('--sha');a=parser.parse_args()
    if a.action=='prepare':print(json.dumps(prepare()))
    elif a.action=='worker':print(json.dumps(dict(completed=worker(a.sha)['completed'])))
    else:r=run(a.sha);print(json.dumps(r));raise SystemExit(0 if r['completed']else 1)
