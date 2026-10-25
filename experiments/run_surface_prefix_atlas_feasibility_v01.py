"""Prospective113-point prefix-only NH80 correction-atlas validation.

No data/PDE calls on import/prepare. All points/new outcomes are retained;
finite validation does not establish a global atlas or posterior guarantee.
Root owns actual release. Every field, parameter, noise and query stays LOCAL.
"""
import argparse,ast,json,os,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_surface_nh80_cost_v01 as cost
from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas,unit_axis
ROOT=cost.ROOT;LOCAL=ROOT/'runs/operator-dev/surface-prefix-atlas-feasibility-v01'
SOURCE='experiments/run_surface_prefix_atlas_feasibility_v01.py';TEST='tests/test_surface_prefix_atlas_feasibility_v01.py'
DESIGN=ROOT/'runs/operator-dev/clamped-surface-bridge-review-v01/nonlinear-atlas-validation-design-v04-LOCAL.json'
DESIGN_SHA='f7770c5b7029d70ad97893c00acd22bb2175ce6da488f7dcf79ebc97de0aba2f'
RESOURCES=dict(worker_CPUs=list(range(8)),controller_CPU=16,threads=1,GPU=False,worker_RSS_cap_bytes=4*1024**3,aggregate_RSS_cap_bytes=32*1024**3,worker_wall_cap_s=900,global_wall_cap_s=1200,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=2*1024**3)
sha=cost.sha;write=cost.write;record=cost.record;array_sha=cost.array_sha


def sources():
    pending=[SOURCE,TEST];seen=set()
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


def absent(pid):
    if type(pid)is not int or pid<=1:raise ValueError('Owned positive PID')
    for function in(os.kill,os.killpg):
        try:function(pid,0)
        except ProcessLookupError:pass
        else:raise ValueError('Prerequisite owned PID/group remains live')


def cost_gate():
    directory=cost.LOCAL;p=json.loads((directory/'protocol.json').read_text());r=json.loads((directory/'result.json').read_text());e=json.loads((directory/'execution.json').read_text())
    if p['source_sha256']!=cost.sources()or r['protocol_sha256']!=sha(directory/'protocol.json')or e['protocol_sha256']!=r['protocol_sha256']:raise ValueError('Cost source/protocol pins')
    if not all(r.get(k)is True for k in('completed','passed'))or not e.get('completed')or not e.get('terminal')or e.get('worker_exit_code')!=0 or e.get('cleanup_exit_code')!=0:raise ValueError('Successful closed cost prerequisite')
    if e.get('result')!=record(directory/'result.json')or e.get('worker_pid')!=r.get('worker_pid'):raise ValueError('Cost result/PID receipt binding')
    absent(e['controller_pid']);absent(e['worker_pid'])
    if len(r['cases'])!=3 or not all(c['field_parity']['passed']and all(b['passed']for b in c['branches'].values())for c in r['cases']):raise ValueError('All six physics cost branches required')
    return {str((directory/n).relative_to(ROOT)):sha(directory/n)for n in('protocol.json','result.json','execution.json')}


def point_design(design):
    axes=unit_axis(9)
    for n in(3,5,9):
        if not np.array_equal(unit_axis(n),np.asarray(design['anchor_axis_values'][str(n)])):raise ValueError('Frozen Chebyshev coordinate bytes')
    anchor=np.stack(np.meshgrid(axes,axes,indexing='ij'),-1).reshape(-1,2);query=np.asarray(design['offgrid_design']['queries'],float)
    if query.shape!=(32,2)or not np.isfinite(query).all()or np.any((query<0)|(query>1)):raise ValueError('All32 fixed strict-prior queries')
    points=np.vstack((anchor,query))
    if len(np.unique(points,axis=0))!=113:raise ValueError('Exactly113 distinct points, no exclusions')
    return points


def guard(slot=None):
    cpu=RESOURCES['controller_CPU']if slot is None else RESOURCES['worker_CPUs'][slot]
    if set(os.sched_getaffinity(0))!={cpu}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact CPU/one-thread/noGPU guard')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES['worker_RSS_cap_bytes']:raise MemoryError('Own4GiB RSS cap')
    available=int(next(v for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:')).split()[1])*1024
    if available<RESOURCES['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise MemoryError('30GiB RAM/disk reserves')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise OSError('Own2GiB writes cap')


def prepare():
    guard();cost.admission_gate();prerequisite=cost_gate()
    if sha(DESIGN)!=DESIGN_SHA:raise ValueError('Frozen validation design changed')
    design=json.loads(DESIGN.read_text());points=point_design(design)
    from experiments.campaign_surface_p2_force_v03 import load_catalog
    protocol=dict(schema='surface-prefix-atlas-feasibility-v01',status='FROZEN_ROOT_RELEASE_REQUIRED',source_SHA256=sources(),design=record(DESIGN),cost_SHA256=prerequisite,admission=record(cost.ADMISSION),resources=RESOURCES,points_unit_LOCAL=points.tolist(),points_SHA256=array_sha(points),new_PDE_points=113,mesh_n=4,label='normal_x180',force_SHA256=array_sha(load_catalog(4)['normal_x180']),NH_quadrature=80,initialization='Exact matched linear equilibrium, same rest-LU reused for PCG',all113_retained=True,implicit_retries=False,atlas_levels=[3,5,9],refinement_query_grid=33,old16_data_read=False,inference_calls=0,scope='Prefix-only feasibility; all10-load admission and independent posterior convergence remain separate',pass_gates=design['pass_gates'])
    if (LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    write(LOCAL/'protocol.json',protocol);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=RESOURCES or p['design']!=record(DESIGN)or sha(DESIGN)!=DESIGN_SHA or p['admission']!=record(cost.ADMISSION)or p['cost_SHA256']!=cost_gate():raise ValueError('Exact source/design/cost/admission protocol binding')
    cost.admission_gate();points=point_design(json.loads(DESIGN.read_text()))
    if not np.array_equal(points,np.asarray(p['points_unit_LOCAL']))or p['points_SHA256']!=array_sha(points):raise ValueError('Frozen113-point membership')
    from experiments.campaign_surface_p2_force_v03 import load_catalog
    if p['label']!='normal_x180'or p['mesh_n']!=4 or p['NH_quadrature']!=80 or p['force_SHA256']!=array_sha(load_catalog(4)['normal_x180']):raise ValueError('Exact prefix physics scope')
    return p


def solve_point(fwd,force,Q,unit):
    from experiments.campaign_exact_rest_pcg_v01 import ExactRestPCG
    from experiments.campaign_surface_preconditioned_p2_v01 import PreconditionedP2Reference
    from experiments.campaign_generalized_stability_v01 import generalized_stability
    from experiments.run_clamped_surface_p2_qualification_v03 import deformation
    from experiments.run_clamped_surface_p2_qualification_v01 import reaction_diagnostics
    E,K=np.exp(np.log([2000.,5.])+np.asarray(unit)*np.log([4.,6.]));rest=E*fwd.Kunit_free+K*fwd.W_free;directions=ExactRestPCG(rest)
    uL=np.zeros_like(fwd.X);uL.ravel()[fwd.free]=directions.rest_solve(force.ravel()[fwd.free]);linear_r=(E*fwd.Kunit+K*fwd.W)@uL.ravel()-force.ravel();linear_R=float(np.linalg.norm(linear_r[fwd.free]))
    if linear_R>1e-10 or np.any(uL[fwd.fixed]!=0):raise RuntimeError('Original exact linear residual/clamps failed')
    model=PreconditionedP2Reference.from_forward(fwd,E,K,quadrature_order=80,direction_solver=directions)
    u,diag=model.solve(force,initial=uL,maximum_iterations=40,force_tolerance_N=1e-10)
    state=model.evaluate(u,force,tangent=True);R=float(np.linalg.norm(state['residual'].ravel()[fwd.free]));d=deformation(fwd,u);eq=reaction_diagnostics(fwd,u,force,E,K,state['residual']);stable=generalized_stability(state['tangent'][fwd.free][:,fwd.free],rest,directions.rest_LU.solve)
    response=np.asarray(Q@u.ravel()[fwd.free]);linear=np.asarray(Q@uL.ravel()[fwd.free]);passed=R<=1e-10 and d['physical_passed']and eq['passed']and stable['passed']and np.all(u[fwd.fixed]==0)
    diagnostics=dict(passed=bool(passed),E_Pa=float(E),K_N_m=float(K),original_linear_free_R_l2_N=linear_R,original_nonlinear_free_R_l2_N=R,Newton=diag,deformation=d,equilibrium=eq,stability=stable,rest_factorization_s=directions.rest_factorization_s,current_fallback_factorizations=directions.current_fallback_factorizations)
    arrays=dict(displacement_m=u,linear_displacement_m=uL,response_m=response,linear_response_m=linear,delta_m=response-linear,unit=np.asarray(unit),physical_parameters=np.array([E,K]))
    return arrays,diagnostics


def worker(slot,digest):
    if type(slot)is not int or not 0<=slot<8:raise ValueError('Eight fixed worker slots')
    guard(slot);p=validate(digest);directory=LOCAL/f'worker-{slot}'
    if directory.exists():raise FileExistsError('No worker retry/overwrite')
    directory.mkdir();start=time.monotonic();state=dict(completed=False,slot=slot,pid=os.getpid(),protocol_SHA256=digest,points=[])
    def save():state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(directory/'summary.json',state)
    def stopped(*_):raise TimeoutError('Owned900s worker cap')
    signal.signal(signal.SIGALRM,stopped);signal.signal(signal.SIGTERM,stopped);signal.alarm(900)
    try:
        from experiments.campaign_surface_p2_static_v02 import SparseStaticForward,point_query_matrix
        from experiments.campaign_surface_p2_force_v03 import load_catalog
        from experiments.campaign_surface_p2_study_design_v01 import observation_points
        fwd=SparseStaticForward(4);force=load_catalog(4)['normal_x180'];Q=point_query_matrix(fwd,observation_points());save()
        for index in range(slot,113,8):
            guard(slot);begin=time.monotonic();row=dict(index=index,completed=False,passed=False,protocol_SHA256=digest,unit_SHA256=array_sha(p['points_unit_LOCAL'][index]))
            try:
                arrays,diag=solve_point(fwd,force,Q,p['points_unit_LOCAL'][index]);guard(slot);target=directory/f'point-{index:03d}-LOCAL.npz'
                if target.exists():raise FileExistsError('No point overwrite')
                np.savez_compressed(target,**arrays);row['raw']=record(target);row['diagnostics']=diag;guard(slot)
                row.update(completed=True,passed=diag['passed'],raw=record(target),diagnostics=diag)
            except Exception as error:
                row['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
                if isinstance(error,(MemoryError,TimeoutError)):raise
            finally:
                row['wall_s']=time.monotonic()-begin;write(directory/f'point-{index:03d}.json',row);state['points'].append(row);save()
        state['completed']=True;state['passed']=all(r['passed']for r in state['points']);save()
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));save();raise
    finally:signal.alarm(0);save()
    return state


def atlas_checks(design,delta,linear_queries,direct_queries,*,fingerprint):
    """Pure finite-validation calculation; no PDE or old data reads."""
    table=np.asarray(delta,float)
    if table.shape!=(9,9,1,24):raise ValueError('All81 prefix24 correction anchors')
    query=np.asarray(design['offgrid_design']['queries']);L=np.asarray(linear_queries,float);D=np.asarray(direct_queries,float)
    if L.shape!=(32,24)or D.shape!=L.shape or not np.isfinite(L).all()or not np.isfinite(D).all():raise ValueError('All32 independently solved query responses')
    final=NonlinearResidualAtlas(table,['normal_x180'],provenance_fingerprint=fingerprint);grid=np.stack(np.meshgrid(np.linspace(0,1,33),np.linspace(0,1,33),indexing='ij'),-1).reshape(-1,2)
    likelihood=design['observed_likelihood_control'];levels=[]
    for n in(3,5,9):
        atlas=final.coarsen(n);pred=L+atlas.predict(query,['normal_x180'])[:,0,:];error=pred-D;cases=[]
        for truth,seed in zip(likelihood['truth_indices_into_interior'],likelihood['noise_seeds']):
            noise=np.random.default_rng(seed).normal(0,20e-6,size=(3,24))
            for reset in range(3):
                y=D[truth]+noise[reset];residual=(y-D)/20e-6;approx=(y-pred)/20e-6
                directLL=-.5*np.sum(residual**2,axis=1)-24*np.log(20e-6*np.sqrt(2*np.pi));atlasLL=-.5*np.sum(approx**2,axis=1)-24*np.log(20e-6*np.sqrt(2*np.pi));diff=abs(atlasLL-directLL);near=directLL>=directLL.max()-25
                white=np.linalg.norm(error/20e-6,axis=1);bound=np.linalg.norm(residual,axis=1)*white+.5*white**2
                if np.any(diff>bound+1e-10):raise FloatingPointError('Gaussian discrepancy bound violated')
                cases.append(dict(truth_query_index=int(truth),reset=reset,maximum_allpoint_loglikelihood_error=float(diff.max()),maximum_near_tested_peak_loglikelihood_error=float(diff[near].max()),near_tested_peak_points=int(near.sum()),maximum_whitened_error_norm=float(white.max()),maximum_Gaussian_discrepancy_bound=float(bound.max()),pointwise_loglikelihood_error=diff.tolist(),pointwise_whitened_error_norm=white.tolist(),pointwise_Gaussian_discrepancy_bound=bound.tolist()))
        levels.append(dict(n=n,maximum_coordinate_error_m=float(abs(error).max()),RMS_coordinate_error_m=float(np.sqrt(np.mean(error**2))),pointwise_maximum_coordinate_error_m=abs(error).max(axis=1).tolist(),likelihood_cases=cases))
    grid_refinement=float(abs(final.coarsen(5).predict(grid,['normal_x180'])-final.predict(grid,['normal_x180'])).max())
    query_refinement=float(abs(final.coarsen(5).predict(query,['normal_x180'])-final.predict(query,['normal_x180'])).max())
    refinement=max(grid_refinement,query_refinement)
    finalrow=levels[-1];passed=finalrow['maximum_coordinate_error_m']<=1e-7 and refinement<=1e-7 and all(c['maximum_near_tested_peak_loglikelihood_error']<=.01 for c in finalrow['likelihood_cases'])
    return dict(passed=bool(passed),levels=levels,maximum_grid5_vs_grid9_coordinate_difference_m=grid_refinement,maximum_fixedquery5_vs_grid9_coordinate_difference_m=query_refinement,maximum_combined5_vs_grid9_coordinate_difference_m=refinement,all32_queries_retained=True,all12_likelihood_cases_retained=True,scope='Finite prefix validation only; no global or posterior/calibration guarantee'),final


def summarize(digest):
    p=validate(digest);points=[]
    for slot in range(8):
        a=json.loads((LOCAL/f'worker-{slot}/summary.json').read_text())
        if not a['completed']or a['slot']!=slot or a['protocol_SHA256']!=digest:raise ValueError('All eight terminal point workers')
        points+=a['points']
    points.sort(key=lambda r:r['index'])
    if [r['index']for r in points]!=list(range(113)):raise ValueError('All113 points once')
    result=dict(completed=True,qualified=False,protocol_SHA256=digest,all113_retained=True,physical_passed=sum(r['passed']for r in points),failures=[r['index']for r in points if not r['passed']],all_old_results_preserved=True,no_old16_data_read=True)
    if result['physical_passed']==113:
        raw=[]
        for row in points:
            if row['protocol_SHA256']!=digest or row['unit_SHA256']!=array_sha(p['points_unit_LOCAL'][row['index']])or row['raw']['sha256']!=sha(ROOT/row['raw']['path']):raise ValueError('Point raw/context binding')
            with np.load(ROOT/row['raw']['path'],allow_pickle=False)as z:
                if not np.array_equal(z['unit'],p['points_unit_LOCAL'][row['index']]):raise ValueError('Raw point coordinates')
                raw.append((z['delta_m'].copy(),z['linear_response_m'].copy(),z['response_m'].copy()))
        check,atlas=atlas_checks(json.loads(DESIGN.read_text()),np.array([r[0]for r in raw[:81]]).reshape(9,9,1,24),np.array([r[1]for r in raw[81:]]),np.array([r[2]for r in raw[81:]]),fingerprint=digest)
        archive=LOCAL/'prefix-atlas-LOCAL.npz'
        if archive.exists():raise FileExistsError('No atlas overwrite')
        np.savez_compressed(archive,delta=atlas.delta,axis=atlas.axis)
        result.update(qualified=check['passed'],validation=check,atlas=record(archive),atlas_fingerprint=atlas.fingerprint)
    write(LOCAL/'summary-LOCAL.json',result);return result


def worker_command(slot,digest,*,guard_only=False):
    if type(slot)is not int or not 0<=slot<8:raise ValueError('Fixed eight slots')
    action='worker-guard'if guard_only else'worker'
    return ['taskset','-c',str(slot),sys.executable,'-u','-m','experiments.run_surface_prefix_atlas_feasibility_v01',action,'--slot',str(slot),'--sha',digest]


def run(digest):
    guard();p=validate(digest);release=json.loads((LOCAL/'release.json').read_text())
    if release.get('approved')is not True or release.get('protocol_SHA256')!=digest or release.get('source_SHA256')!=p['source_SHA256']:raise ValueError('Concrete root-approved hash-bound release required')
    if (LOCAL/'execution.json').exists()or any(LOCAL.glob('worker-*')):raise FileExistsError('One actual execution, no retries')
    available=int(next(v for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:')).split()[1])*1024
    if available<RESOURCES['aggregate_RSS_cap_bytes']+RESOURCES['RAM_reserve_bytes']:raise MemoryError('32GiB own budget plus30GiB reserve preflight')
    start=time.monotonic();children=[];logs=[];state=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),workers=[])
    def save():state['wall_s']=time.monotonic()-start;write(LOCAL/'execution.json',state)
    def stopped(*_):raise InterruptedError('Owned controller stop')
    old={sig:signal.signal(sig,stopped)for sig in(signal.SIGTERM,signal.SIGINT)}
    try:
        for slot in range(8):
            log=(LOCAL/f'worker-{slot}.log').open('xb');logs.append(log)
            child=subprocess.Popen(worker_command(slot,digest),cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child);state['workers'].append(dict(slot=slot,pid=child.pid,reaped=False,started_s=time.monotonic()-start));save()
        while any(child.poll()is None for child in children):
            guard();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            for child,row in zip(children,state['workers']):
                if child.poll()is not None:
                    row.update(reaped=True,exit_code=child.wait())
                    if child.returncode!=0:raise RuntimeError('Owned worker abnormal exit; cleanup all groups')
                    continue
                try:own=int(next(v for v in Path('/proc',str(child.pid),'status').read_text().splitlines()if v.startswith('VmRSS:')).split()[1])*1024
                except FileNotFoundError:own=0
                rss+=own
                if own>RESOURCES['worker_RSS_cap_bytes']or time.monotonic()-start-row['started_s']>RESOURCES['worker_wall_cap_s']:raise TimeoutError('Independent owned worker RSS/wall cap')
            state['maximum_observed_aggregate_RSS_bytes']=max(state.get('maximum_observed_aggregate_RSS_bytes',0),rss)
            if rss>RESOURCES['aggregate_RSS_cap_bytes']or time.monotonic()-start>RESOURCES['global_wall_cap_s']:raise TimeoutError('Independent aggregate RSS/global wall cap')
            save();time.sleep(.2)
        for child,row in zip(children,state['workers']):row.update(reaped=True,exit_code=child.wait())
        if any(row['exit_code']!=0 for row in state['workers']):raise RuntimeError('Worker failed')
        for child in children:absent(child.pid)
        result=summarize(digest);state.update(completed=True,qualified=result['qualified'],summary=record(LOCAL/'summary-LOCAL.json'))
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        for child,row in zip(children,state['workers']):row.update(reaped=True,exit_code=cost.cleanup(child))
        for child in children:absent(child.pid)
        for log in logs:log.close()
        for sig,handler in old.items():signal.signal(sig,handler)
        state['terminal']=True;save()
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','run','worker-guard','controller-guard']);parser.add_argument('--sha');parser.add_argument('--slot',type=int);args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare()))
    elif args.action=='worker':a=worker(args.slot,args.sha);print(json.dumps(dict(completed=a['completed'],passed=a['passed'])))
    elif args.action=='run':a=run(args.sha);print(json.dumps(dict(completed=a['completed'],qualified=a.get('qualified',False))));raise SystemExit(0 if a['completed'] else 1)
    elif args.action=='worker-guard':guard(args.slot);print('guard passed')
    else:guard();print('guard passed')
