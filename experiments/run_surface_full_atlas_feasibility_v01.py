"""DRAFT full10-load atlas continuation; no actual launch/admission on import.

113 prefix states are reused by immutable context hashes; 1017 other force
states are new. One rest LU/model per material point, independently initialized
linear warm states per force. Original prefixes/failed runs remain immutable.
"""
import argparse,hashlib,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_surface_prefix_atlas_recovery_v01 as recovery
from experiments import run_surface_prefix_atlas_feasibility_v01 as prefix
from experiments import validate_surface_prefix_atlas_v01 as validation
ROOT=prefix.ROOT;LOCAL=ROOT/'runs/operator-dev/surface-full-atlas-feasibility-v01'
SOURCE='experiments/run_surface_full_atlas_feasibility_v01.py';TEST='tests/test_surface_full_atlas_feasibility_v01.py'
# This source-only draft cannot prepare until the independent joint design
# and its exact SHA are frozen and reviewed. No automatic gate/seed selection.
JOINT_DESIGN=ROOT/'runs/operator-dev/clamped-surface-bridge-review-v01/nonlinear-full10-atlas-joint-validation-design-v01-LOCAL.json'
JOINT_DESIGN_SHA='4085af817051570da18ea1632c3eab928abe6a70c05a0634319cde1ead30263a'
RESOURCES=dict(worker_CPUs=list(range(16)),controller_CPU=16,threads=1,GPU=False,worker_RSS_cap_bytes=4*1024**3,aggregate_RSS_cap_bytes=64*1024**3,worker_wall_cap_s=900,global_wall_cap_s=1200,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=2*1024**3)
PREFIX='normal_x180'
ACTIONS=('normal_x090','normal_x180','normal_x270','paired_axial_x090','paired_axial_x180','paired_axial_x270')
sha=prefix.sha;record=prefix.record;write=prefix.write;array_sha=prefix.array_sha


def sources():
    import ast
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


def prefix_gate():
    oldp,rows,origin=validation.inputs();base=validation.LOCAL;digest=sha(base/'protocol.json')
    p=json.loads((base/'protocol.json').read_text());a=json.loads((base/'result-LOCAL.json').read_text());release=json.loads((base/'release.json').read_text())
    if p['source_SHA256']!=validation.sources()or p['origin_SHA256']!=origin or not a.get('completed')or not a.get('qualified')or a['protocol_SHA256']!=digest or a['new_PDE_calls']!=0 or a['subprocess_calls']!=0 or not a['all113_retained']:raise ValueError('Qualified unchanged validation-only prefix admission')
    if release.get('approved')is not True or release.get('protocol_SHA256')!=digest or release.get('source_SHA256')!=p['source_SHA256']:raise ValueError('Prefix validation release context')
    prefix.absent(a['pid']);origin=dict(origin)
    for name in('protocol.json','release.json','result-LOCAL.json'):
        path=base/name;origin[str(path.relative_to(ROOT))]=sha(path)
    path=ROOT/a['atlas']['path']
    if sha(path)!=a['atlas']['sha256']:raise ValueError('Admitted prefix atlas bytes')
    origin[str(path.relative_to(ROOT))]=sha(path)
    return oldp,rows,origin


def guard(slot=None):
    import shutil
    cpu=16 if slot is None else slot
    if slot is not None and(type(slot)is not int or not 0<=slot<16):raise ValueError('Fixed16worker slots')
    if set(os.sched_getaffinity(0))!={cpu}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact CPU/one-thread/noGPU guard')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES['worker_RSS_cap_bytes']:raise MemoryError('Own4GiB RSS cap')
    available=int(next(v for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:')).split()[1])*1024
    if available<RESOURCES['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise MemoryError('30GiB RAM/disk reserves')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise OSError('Own2GiB writes')


def prepare():
    guard()
    if JOINT_DESIGN_SHA is None:raise ValueError('Source-only draft: independent full48-channel joint design must be frozen first')
    if sha(JOINT_DESIGN)!=JOINT_DESIGN_SHA:raise ValueError('Fixed new joint validation design')
    oldp,rows,origin=prefix_gate()
    from experiments.campaign_surface_p2_force_v03 import load_catalog
    loads=load_catalog(4);labels=tuple(loads)
    if len(labels)!=10 or PREFIX not in labels or not set(ACTIONS)<=set(labels):raise ValueError('Exactly10frozen force programs')
    p=dict(schema='surface-full-atlas-feasibility-v01',status='FROZEN_ROOT_RELEASE_REQUIRED',source_SHA256=sources(),resources=RESOURCES,prefix_origin_SHA256=origin,joint_design=record(JOINT_DESIGN),prefix_design=record(prefix.DESIGN),force_SHA256={k:array_sha(v)for k,v in loads.items()},labels=list(labels),points_unit_LOCAL=oldp['points_unit_LOCAL'],new_equilibria=1017,reused_prefix_states=113,total_states=1130,all_states_retained=True,implicit_retries=False,no_old16_data=True,material_gates=dict(maximum_strain=.05,minimum_J=.8,free_residual_N=1e-10,exact_clamps=True,original_reaction=True,generalized_stability=True),excitation='100um/.3 ONLY prefix; nonprefix nonrigid amplitudes recorded, not validity filters',factor_reuse='One exact rest-LU and NH80 model per E/K; nine independent force warmstarts/Newton solves',joint_likelihood_gate='Full independent48channel prefix+probe near jointdirecttestedmax−25 discrepancy<=.01; each six actions, four truths, three resets; allpoints reported',response_gates=dict(maximum_coordinate_error_m=1e-7,maximum_5_vs_9_difference_m=1e-7),scope='Finite-model/full10load atlas validation only, not global certification or posterior calibration')
    if (LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text());oldp,rows,origin=prefix_gate()
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=RESOURCES or p['prefix_origin_SHA256']!=origin or p['joint_design']!=record(JOINT_DESIGN)or sha(JOINT_DESIGN)!=JOINT_DESIGN_SHA or p['prefix_design']!=record(prefix.DESIGN):raise ValueError('Exact fullatlas source/origin/jointdesign/resources binding')
    if not np.array_equal(p['points_unit_LOCAL'],oldp['points_unit_LOCAL']):raise ValueError('Same113original points, no resampling')
    from experiments.campaign_surface_p2_force_v03 import load_catalog
    if p['force_SHA256']!={k:array_sha(v)for k,v in load_catalog(4).items()}:raise ValueError('Same10physical force programs')
    return p,rows


def solve_unit(fwd,forces,Q,unit):
    """Yield independent force equilibria sharing only the exact rest factor."""
    from experiments.campaign_exact_rest_pcg_v01 import ExactRestPCG
    from experiments.campaign_surface_preconditioned_p2_v01 import PreconditionedP2Reference
    from experiments.campaign_generalized_stability_v01 import generalized_stability
    from experiments.run_clamped_surface_p2_qualification_v03 import deformation
    from experiments.run_clamped_surface_p2_qualification_v01 import reaction_diagnostics
    E,K=np.exp(np.log([2000.,5.])+np.asarray(unit)*np.log([4.,6.]));rest=E*fwd.Kunit_free+K*fwd.W_free;directions=ExactRestPCG(rest)
    model=PreconditionedP2Reference.from_forward(fwd,E,K,quadrature_order=80,direction_solver=directions)
    for label,force in forces.items():
        if label==PREFIX:continue
        try:
            begin=time.monotonic();uL=np.zeros_like(fwd.X);uL.ravel()[fwd.free]=directions.rest_solve(force.ravel()[fwd.free]);lr=(E*fwd.Kunit+K*fwd.W)@uL.ravel()-force.ravel();linear_R=float(np.linalg.norm(lr[fwd.free]))
            if linear_R>1e-10 or np.any(uL[fwd.fixed]!=0):raise RuntimeError('Exact linear force/clamp gate')
            u,diag=model.solve(force,initial=uL,maximum_iterations=40,force_tolerance_N=1e-10);state=model.evaluate(u,force,tangent=True);R=float(np.linalg.norm(state['residual'].ravel()[fwd.free]));d=deformation(fwd,u);eq=reaction_diagnostics(fwd,u,force,E,K,state['residual']);stable=generalized_stability(state['tangent'][fwd.free][:,fwd.free],rest,directions.rest_LU.solve)
            material=d['minimum_J']>=.8 and d['maximum_absolute_symmetric_strain_eigenvalue']<=.05
            passed=material and R<=1e-10 and eq['passed']and stable['passed']and np.all(u[fwd.fixed]==0)
            nonlinear=np.asarray(Q@u.ravel()[fwd.free]);linear=np.asarray(Q@uL.ravel()[fwd.free])
            arrays=dict(displacement_m=u,linear_displacement_m=uL,response_m=nonlinear,linear_response_m=linear,delta_m=nonlinear-linear,unit=np.asarray(unit),physical_parameters=np.array([E,K]))
            diagnostics=dict(passed=bool(passed),material_valid=bool(material),excitation_descriptive_only=True,original_linear_free_R_l2_N=linear_R,original_nonlinear_free_R_l2_N=R,deformation=d,equilibrium=eq,stability=stable,Newton=diag,rest_factorization_s=directions.rest_factorization_s,current_fallback_factorizations_cumulative=directions.current_fallback_factorizations,wall_s=time.monotonic()-begin)
            yield label,arrays,diagnostics
        except Exception as error:
            if isinstance(error,(MemoryError,TimeoutError,InterruptedError)):raise
            yield label,None,dict(passed=False,completed=False,failure=dict(type=type(error).__name__,message=str(error)))



def joint_likelihood_checks(prefix_direct,prefix_approx,probe_direct,probe_approx,*,truth_indices,seeds,resets=3):
    """Full48channel Gaussian discrepancy; predictions never generate observations."""
    P=np.asarray(prefix_direct,float);A=np.asarray(prefix_approx,float)
    if P.shape!=(32,24)or A.shape!=P.shape or not np.isfinite(P).all()or not np.isfinite(A).all()or len(truth_indices)!=4 or len(seeds)!=4 or resets!=3:raise ValueError('Fixed32queries24channels/fourtruths/threeresets')
    rows=[]
    for truth,seed in zip(truth_indices,seeds):
        noise=np.random.default_rng(seed).normal(0,20e-6,size=(3,2,24))
        for action in ACTIONS:
            D=np.asarray(probe_direct[action],float);B=np.asarray(probe_approx[action],float)
            if D.shape!=P.shape or B.shape!=P.shape or not np.isfinite(D).all()or not np.isfinite(B).all():raise ValueError('All six direct/predicted probe response arrays')
            direct=np.column_stack((P,D));approx=np.column_stack((A,B))
            for reset in range(3):
                y=np.r_[P[truth]+noise[reset,0],D[truth]+noise[reset,1]];r=(y-direct)/20e-6;e=(approx-direct)/20e-6
                dll=-.5*np.sum(r*r,axis=1)-48*np.log(20e-6*np.sqrt(2*np.pi));alll=-.5*np.sum((r-e)**2,axis=1)-48*np.log(20e-6*np.sqrt(2*np.pi));difference=abs(alll-dll);near=dll>=dll.max()-25
                white=np.linalg.norm(e,axis=1);bound=np.linalg.norm(r,axis=1)*white+.5*white**2
                if np.any(difference>bound+1e-9):raise FloatingPointError('Joint Gaussian discrepancy bound violation')
                rows.append(dict(action=action,truth_query_index=int(truth),reset=reset,maximum_near_joint_tested_peak_loglikelihood_error=float(difference[near].max()),maximum_allpoint_loglikelihood_error=float(difference.max()),near_joint_tested_peak_points=int(near.sum()),pointwise_loglikelihood_error=difference.tolist(),pointwise_whitened_error_norm=white.tolist(),pointwise_Gaussian_discrepancy_bound=bound.tolist()))
    return dict(passed=bool(all(r['maximum_near_joint_tested_peak_loglikelihood_error']<=.01 for r in rows)),cases=rows,all72_joint_cases_retained=len(rows)==72,noise='Independent prefix/probe24vectors; paired across actions within truth/reset',scope='Finite32query joint-likelihood validation, not global guarantee')


def owned_RSS(child,*,read_status=None,grace_s=.1,pause_s=.005):
    """Bounded exit-transition grace; living unknown RSS still fails closed."""
    deadline=time.monotonic()+grace_s
    while True:
        try:return recovery.owned_RSS(child,read_status=read_status)
        except RuntimeError as error:
            if 'unresolved RSS'not in str(error)or time.monotonic()>=deadline:raise
            time.sleep(pause_s)


def worker(slot,digest):
    guard(slot);p,origins=validate(digest);directory=LOCAL/f'worker-{slot}'
    if directory.exists():raise FileExistsError('No actual worker retries')
    directory.mkdir();start=time.monotonic();state=dict(completed=False,slot=slot,pid=os.getpid(),protocol_SHA256=digest,rows=[])
    def save():state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(directory/'summary-LOCAL.json',state)
    def deadline(*_):raise TimeoutError('Actual900s worker deadline')
    def terminated(*_):raise InterruptedError('Explicit supervisor SIGTERM cleanup')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,terminated);signal.alarm(900)
    try:
        from experiments.campaign_surface_p2_static_v02 import SparseStaticForward,point_query_matrix
        from experiments.campaign_surface_p2_force_v03 import load_catalog
        from experiments.campaign_surface_p2_study_design_v01 import observation_points
        fwd=SparseStaticForward(4);forces=load_catalog(4);Q=point_query_matrix(fwd,observation_points());save()
        for index in range(slot,113,16):
            guard(slot);unit=p['points_unit_LOCAL'][index];rows=[]
            def commit(row):
                row.update(index=index,protocol_SHA256=digest,unit_SHA256=array_sha(unit));write(directory/f'point-{index:03d}-{row["label"]}.json',row);state['rows'].append(row);rows.append(row);save()
            origin=origins[index];commit(dict(label=PREFIX,completed=True,passed=True,reused=True,raw=origin['raw'],origin_record_SHA256=hashlib.sha256(json.dumps(origin,sort_keys=True).encode()).hexdigest(),origin_original_protocol_SHA256=origin['protocol_SHA256']))
            try:
                for label,arrays,diag in solve_unit(fwd,forces,Q,unit):
                    guard(slot);row=dict(label=label,completed=arrays is not None,passed=diag['passed'],reused=False,diagnostics=diag)
                    if arrays is not None:
                        target=directory/f'point-{index:03d}-{label}-LOCAL.npz'
                        if target.exists():raise FileExistsError('No raw overwrite')
                        np.savez_compressed(target,**arrays);row['raw']=record(target);guard(slot)
                    commit(row)
            except Exception as error:
                if isinstance(error,(MemoryError,TimeoutError,InterruptedError)):raise
                for label in p['labels']:
                    if label not in {r['label']for r in rows}:commit(dict(label=label,completed=False,passed=False,reused=False,failure=dict(type=type(error).__name__,message=str(error))))
            if {r['label']for r in rows}!=set(p['labels']):raise ValueError('Every point all10forces retained')
        state.update(completed=True,passed=all(r['passed']for r in state['rows']));save()
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());save();raise
    finally:signal.alarm(0);save()
    return state


def summarize(digest):
    p,origins=validate(digest);rows={};labels=p['labels']
    for slot in range(16):
        directory=LOCAL/f'worker-{slot}';state=json.loads((directory/'summary-LOCAL.json').read_text())
        if not state['completed']or state['slot']!=slot or state['protocol_SHA256']!=digest:raise ValueError('All16 terminal workers')
        for row in state['rows']:
            i=row['index'];label=row['label'];metadata=directory/f'point-{i:03d}-{label}.json'
            if i%16!=slot or (i,label)in rows or row['protocol_SHA256']!=digest or row['unit_SHA256']!=array_sha(p['points_unit_LOCAL'][i])or json.loads(metadata.read_text())!=row or (row['passed']and not row['completed']):raise ValueError('Unique point/force metadata/context')
            if row['reused']!=(label==PREFIX):raise ValueError('Prefix reuse only, every other force newly solved')
            if row['reused']and(row['raw']!=origins[i]['raw']or row['origin_record_SHA256']!=hashlib.sha256(json.dumps(origins[i],sort_keys=True).encode()).hexdigest()or row['origin_original_protocol_SHA256']!=origins[i]['protocol_SHA256']):raise ValueError('Exact prefix-only reuse')
            rows[i,label]=row
    if set(rows)!={(i,l)for i in range(113)for l in labels}:raise ValueError('All1130states once')
    result=dict(completed=True,qualified=False,protocol_SHA256=digest,all1130_retained=True,reused_prefix_states=113,new_state_count=1017,physical_passed=sum(r['passed']for r in rows.values()),failed_states=[dict(index=i,label=l)for (i,l),r in rows.items()if not r['passed']],old_failures_preserved=True)
    if result['physical_passed']==1130:
        data={}
        for label in labels:
            values=[]
            for i in range(113):
                row=rows[i,label];path=ROOT/row['raw']['path']
                if sha(path)!=row['raw']['sha256']:raise ValueError('Every old/new raw SHA')
                with np.load(path,allow_pickle=False)as z:
                    if not np.array_equal(z['unit'],p['points_unit_LOCAL'][i]):raise ValueError('Exact original113coordinates')
                    values.append((z['delta_m'].copy(),z['linear_response_m'].copy(),z['response_m'].copy()))
            data[label]=values
        checks={};atlases={};design=json.loads(prefix.DESIGN.read_text());query=np.asarray(design['offgrid_design']['queries'])
        for label in labels:
            values=data[label];checks[label],single=prefix.atlas_checks(design,np.array([r[0]for r in values[:81]]).reshape(9,9,1,24),np.array([r[1]for r in values[81:]]),np.array([r[2]for r in values[81:]]),fingerprint=digest+label)
            # Relabel the mathematically identical single-force correction.
            atlases[label]=prefix.NonlinearResidualAtlas(single.delta,[label],provenance_fingerprint=digest)
        joint_design=json.loads(JOINT_DESIGN.read_text());mapping=joint_design['joint_noise']['truth_seed_mapping'];joint=[]
        direct={label:np.array([r[2]for r in data[label][81:]])for label in labels}
        for n in(3,5,9):
            approximate={label:np.array([r[1]for r in data[label][81:]])+atlases[label].coarsen(n).predict(query,[label])[:,0,:]for label in labels}
            report=joint_likelihood_checks(direct[PREFIX],approximate[PREFIX],direct,approximate,truth_indices=[r['truth_query_index']for r in mapping],seeds=[r['seed']for r in mapping]);joint.append(dict(n=n,**report))
        target=LOCAL/'full-atlas-LOCAL.npz'
        if target.exists():raise FileExistsError('No fullatlas overwrite')
        delta=np.concatenate([atlases[label].delta for label in labels],axis=2);atlas=prefix.NonlinearResidualAtlas(delta,labels,provenance_fingerprint=digest)
        np.savez_compressed(target,delta=delta,axis=atlas.axis,labels=np.asarray(labels))
        result.update(qualified=bool(all(c['passed']for c in checks.values())and joint[-1]['passed']),perforce_validation=checks,joint_validation_levels=joint,atlas=record(target),atlas_fingerprint=atlas.fingerprint,labels=labels,channels=24,target_mesh_n=4,scope='Finite full10force response/24+48LL validation only; independent posterior convergence/calibration remain necessary')
    write(LOCAL/'summary-LOCAL.json',result);return result


def worker_command(slot,digest,*,guard_only=False):
    if type(slot)is not int or not 0<=slot<16:raise ValueError('Fixed16 slots')
    action='worker-guard'if guard_only else'worker'
    return ['taskset','-c',str(slot),sys.executable,'-u','-m','experiments.run_surface_full_atlas_feasibility_v01',action,'--slot',str(slot),'--sha',digest]


def run(digest):
    guard();p,_=validate(digest);release=json.loads((LOCAL/'release.json').read_text())
    if release.get('approved')is not True or release.get('protocol_SHA256')!=digest or release.get('source_SHA256')!=p['source_SHA256']:raise ValueError('Concrete root-approved hash-bound release required')
    if (LOCAL/'execution.json').exists()or any(LOCAL.glob('worker-*')):raise FileExistsError('One actual execution, no retries')
    available=int(next(v for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:')).split()[1])*1024
    if available<RESOURCES['aggregate_RSS_cap_bytes']+RESOURCES['RAM_reserve_bytes']:raise MemoryError('64GiB own budget plus30GiB reserve preflight')
    start=time.monotonic();children=[];logs=[];state=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),workers=[])
    def save():state['wall_s']=time.monotonic()-start;write(LOCAL/'execution.json',state)
    def stopped(*_):raise InterruptedError('Owned controller stop')
    old={sig:signal.signal(sig,stopped)for sig in(signal.SIGTERM,signal.SIGINT)}
    try:
        for slot in range(16):
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
                own=owned_RSS(child)
                rss+=own
                if own>RESOURCES['worker_RSS_cap_bytes']or time.monotonic()-start-row['started_s']>RESOURCES['worker_wall_cap_s']:raise TimeoutError('Independent owned worker RSS/wall cap')
            state['maximum_observed_aggregate_RSS_bytes']=max(state.get('maximum_observed_aggregate_RSS_bytes',0),rss)
            if rss>RESOURCES['aggregate_RSS_cap_bytes']or time.monotonic()-start>RESOURCES['global_wall_cap_s']:raise TimeoutError('Independent aggregate RSS/global wall cap')
            save();time.sleep(.2)
        for child,row in zip(children,state['workers']):row.update(reaped=True,exit_code=child.wait())
        if any(row['exit_code']!=0 for row in state['workers']):raise RuntimeError('Worker failed')
        for child in children:prefix.absent(child.pid)
        result=summarize(digest);state.update(completed=True,qualified=result['qualified'],summary=record(LOCAL/'summary-LOCAL.json'))
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        for child,row in zip(children,state['workers']):row.update(reaped=True,exit_code=prefix.cost.cleanup(child))
        for child in children:prefix.absent(child.pid)
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
