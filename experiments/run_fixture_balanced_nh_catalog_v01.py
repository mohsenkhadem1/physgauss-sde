"""Prospective bounded eight-interior-state NH80 mechanics catalog.

All264 fields are NEW. No RNG/inference/noise/GPU or mechanics executes on
import. One fixed group is a cost checkpoint inside the immutable worklist;
the remaining seven require a separate explicit root continuation release.
"""
import argparse,ast,copy,gc,json,os,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import norm as sparse_norm
from experiments import run_fixture_balanced_held_nh_refinement_v01 as controls
from experiments import campaign_fixture_balanced_held_v01 as bank
from experiments import campaign_active_fixture_resolution_v03 as geometry

ROOT=controls.ROOT;LOCAL=ROOT/'runs/operator-dev/fixture-balanced-nh-catalog-v01'
SOURCE='experiments/run_fixture_balanced_nh_catalog_v01.py';TEST='tests/test_fixture_balanced_nh_catalog_v01.py'
AUDIT_SOURCE='experiments/audit_fixture_balanced_nh_catalog_v01.py';AUDIT_TEST='tests/test_audit_fixture_balanced_nh_catalog_v01.py'
sha,record,write=controls.sha,controls.record,controls.write
E_VALUES=(2500.,6500.);K_VALUE=float(np.sqrt(150.));LEVELS=(3,6,12)
LOADS=tuple(v['id']for v in bank.load_specs())
LIMITS=dict(worker_CPUs=[14,15,16,17],controller_CPU=18,threads=1,GPU=False,group_wall_s=5400,global_wall_s=21600,worker_RSS_bytes=18*1024**3,controller_RSS_bytes=4*1024**3,aggregate_RSS_bytes=76*1024**3,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,projected_writes_bytes=4*1024**3,maximum_writes_bytes=4*1024**3)
OLD_RESULT_SHA='dfddef7f1e86b259b7c15ed535b09e549b7ae0dd0e3ce621b0e3684bc3329894'
HELD_RESULT_SHA='e7e8842bab85f7d4a01fd0864dd50114b239f811bcaabba880e2f9cfacdfe7b6'


def groups():
    return [dict(index=2*i+j,id=f'{layout["id"]}__E{int(E)}__Ksqrt150',layout_index=i,layout=layout['id'],E_Pa=E,K_N_m=K_VALUE)for i,layout in enumerate(bank.LAYOUTS)for j,E in enumerate(E_VALUES)]

def worklist():
    return [dict(id=f'g{g["index"]:02d}__n{n}__{label}',group_index=g['index'],layout=g['layout'],E_Pa=g['E_Pa'],K_N_m=g['K_N_m'],n=n,label=label,status='queued',kind='NEW')for g in groups()for n in LEVELS for label in LOADS]

def assignments(phase):
    if phase=='checkpoint':return [[0]]
    if phase=='remaining':return [[1,5],[2,6],[3,7],[4]]
    raise ValueError('Exactly checkpoint or remaining phase')

def cost_projection():
    # Existing measured controls only; the eight new contexts have never run.
    initialization={3:.342683,6:4.672969,12:478.432871}
    perload={3:(2.485823+3.402926+3.235110+1.793632+.003227+.005345)/2,6:(19.649688+27.348919+7.956186+7.948577+.052805+.039956)/2,12:(163.013997+238.199542+74.829085+76.705142+.908450+.908078)/2}
    group=sum(initialization.values())+11*sum(perload.values())
    return dict(observed_original_six_field_wall_s=1120.49,observed_original_peak_RSS_bytes=13802950656,observed_original_result_SHA256=OLD_RESULT_SHA,estimated_group_compute_s=group,estimated_serial_s=8*group,estimated_serial_hours=8*group/3600,ideal_four_worker_s=2*group,checkpoint_then_remaining_ideal_s=3*group,first_group_new_fields=33,all_new_fields=264,reused_fields=0,new_rest_factors=24,worker_RSS_margin=LIMITS['worker_RSS_bytes']/13802950656,uncertainty='Measured only oldE2000/K5 one-layout two-load controls; newloads/E/K/layout can change Newton,fill,stability and spatial cost. Estimates omit extra all11 spatial comparisons/checkpoint overhead. No oldLU or oldfield credit; hardguards authoritative.')

def plan():
    return dict(schema='fixture-balanced-nh-catalog-v01',resources=LIMITS,groups=groups(),worklist=worklist(),levels=list(LEVELS),loads=list(LOADS),fields=264,new_fields=264,reused_fields=0,group_load_references=88,bank_design=bank.design(),gates=dict(controls.metrics.GATES),bulk_quadrature_points=80,surface_orders=[16,20],global_wall_scope='Sum of owned checkpoint+remaining phase walltime; no resources held during rootreview gap.',maximum_Newton_iterations=40,force_tolerance_l2_N=1e-10,warm_start='Exact same-load linear equilibrium independently at each mesh andgroup; no saved nonlinear warmstart.',reference_valid='Complete original physical residual/J/strain/exactclamp/reaction/generalized-stability and surface16vs20 checks, then original commonfinest12 spatialorder/fine/medium/publicpoint20um. Excitation does NOT enter reference_valid.',excitation='Unchanged100um coordinate nonrigidRMS ANDfraction>=.3, reported for everyload includingprefix; no filtering or eligibility claim. Lowexcitation may be physically/referencevalid.',J_and_stability_scope='J sampled onvertices/edges/24+80; generalized Ritz numerical evidence with external restSPD assumption, not wholeelement/global rigorous proof.',spatial_scope='Common positiveKeast24 finest12 coordinateRMS; p>=1 conditionalfine and2delta medium20um, query16 n6to12 maxcoordinate20um separately. Indicators are not rigorous continuumbounds.',phase_assignments={p:assignments(p)for p in('checkpoint','remaining')},checkpoint='Fixedgroup0 all33NEWfields INSIDE retained264queue. No automatic expansion: all11referencevalid required before expansion; root continuation-release.json must bind completed checkpoint/rootclosure and remaining frozenresources before groups1..7.',cost_projection=cost_projection(),scope='Eight fixed INTERIOR development states, not proper-prior population/SBC. Allnegativefields/actions retained. No observations/noise/posterior/policycomparison, full660queue or general bank admission.')


def sources():
    todo=[SOURCE,TEST,AUDIT_SOURCE,AUDIT_TEST,'tests/test_fixture_balanced_held_v01.py','tests/test_active_fixture_resolution_v03.py'];seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:modules=[node.module or'']+[(node.module+'.'+v.name)for v in node.names if node.module]
            for module in modules:
                p=module.replace('.','/')+'.py'
                if(ROOT/p).is_file()and p not in seen:todo.append(p)
    return {p:sha(ROOT/p)for p in sorted(seen)}


def history_origin():
    # No arrays: the twelve old extreme controls are HISTORY, never newtruth.
    origin=controls.balanced_origin();out=dict(balanced36=origin,histories=[],reuse_fields=0)
    for stage,expected in((controls.original,OLD_RESULT_SHA),(controls,HELD_RESULT_SHA)):
        base=stage.LOCAL;digest=sha(base/'protocol.json');stage.validate(digest)
        e=json.loads((base/'execution.json').read_text());r=json.loads((base/'result-LOCAL.json').read_text());a=json.loads((base/'independent-review.json').read_text())
        if not(e.get('completed')is True and e.get('terminal')is True and e.get('worker_exit_code')==0 and e.get('cleanup_exit_code')==0 and 'failure'not in e and r.get('completed')is True and len(r.get('rows',[]))==6 and a.get('audit_passed')is True and a.get('scientific_admission')is False):raise ValueError('Exact independently audited oldcontrol history')
        if sha(base/'result-LOCAL.json')!=expected or e['result']!=record(base/'result-LOCAL.json')or a['source_result']!=e['result']or e['release']!=record(base/'release.json')or r['release']!=e['release']:raise ValueError('Old result/source/release history binding')
        for pid in(e['controller_pid'],e['worker_pid']):controls.prior.shared.old.absent(pid)
        out['histories'].append(dict(protocol=record(base/'protocol.json'),release=record(base/'release.json'),execution=record(base/'execution.json'),result=e['result'],independent_review=record(base/'independent-review.json'),fields=6,used_as_history_only=True))
    return out


def available_memory():
    return int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024

def guard(role,slot=None):
    cpu=LIMITS['controller_CPU']if role=='controller'else LIMITS['worker_CPUs'][slot]
    if set(os.sched_getaffinity(0))!={cpu}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact physicalCPU/one-thread/noGPU')
    cap=LIMITS['controller_RSS_bytes']if role=='controller'else LIMITS['worker_RSS_bytes']
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=cap:raise MemoryError('OwnroleRSS cap')
    if available_memory()<LIMITS['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<LIMITS['disk_reserve_bytes']+LIMITS['projected_writes_bytes']:raise MemoryError('30GiB RAM/disk plus projectedwrite reserve')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>LIMITS['maximum_writes_bytes']:raise OSError('Ownoutput cap')


def prepare():
    guard('controller')
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No protocolrefreeze')
    write(LOCAL/'protocol.json',dict(plan=plan(),source_SHA256=sources(),origin=history_origin(),status='FROZEN_ROOT_RELEASE_REQUIRED'))
    return record(LOCAL/'protocol.json')

def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text());r=json.loads((LOCAL/'release.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['plan']!=plan()or p['source_SHA256']!=sources():raise ValueError('Exact frozen source/design')
    controls.screen.verify_records(p['origin'])
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('origin')!=p['origin']:raise ValueError('Root release before anynewfield')
    return p


def surface_checks(f,loads,layout):
    W20,_=bank.bank.consistent_fixture(f.X,f.faces,layout,order=20)
    error=float(sparse_norm(f.W_scalar-W20)/sparse_norm(W20));ref,info=bank.load_catalog(f.n,order=20);prod=bank.load_catalog(f.n)[1]
    load_error=max(float(np.max(abs(loads[k]-ref[k])))for k in loads)
    if f.support_quadrature['normalization_error']>1e-10 or f.support_quadrature['first_moment_error_m']>1e-10 or error>1e-10 or load_error>1e-12 or any(v['net_force_error_N']>1e-12 or v['moment_error_Nm']>1e-13 or v['normalization_error']>1e-10 or v['first_moment_error_m']>1e-10 for v in list(info.values())+list(prod.values())):raise ValueError('Unchanged normalized PSDfixture/knownforce surfaceguards')
    return dict(passed=True,support=f.support_quadrature,positive_quadrature_Gram_PSD=True,relative_support16_vs20=error,maximum_load16_vs20_N=load_error,refined_known_load_checks=info)


def field_status(deformation,R,clamp,equilibrium,stability,label):
    # Frozen original physical predicate; excitation is separate, without a
    # weakprefix exemption or any new old-load-eligibility admission claim.
    old=controls.original.admissibility(deformation,R,clamp,equilibrium,stability,label)
    return dict(physical_passed=old['physical_passed'],excitation_goal_met=old['excitation_goal_met'],legacy_load_eligible=old['load_eligible'],legacy_weak_prefix_control=old['weak_prefix_control'],nonrigid_RMS_goal_m=100e-6,nonrigid_fraction_goal=.3)


def spatial_results(fields):
    out={}
    for label in LOADS:
        values=[(n,fields[n,label])for n in LEVELS];d=geometry.common_rms(values,finest=12);g=controls.spatial_gate(float(d[1,0]),float(d[2,1]));q=[geometry.field_queries(n,u,bank.measurement_points())for n,u in values]
        delta=float(np.max(abs(q[2]-q[1])));g.update(public_point_n3_n6_max_coordinate_m=float(np.max(abs(q[1]-q[0]))),public_point_n6_n12_max_coordinate_m=delta,public_point20um_passed=delta<=20e-6,spatial_development_passed=bool(g['spatial_passed']and g['medium_adequate']and delta<=20e-6),conditional_only=True)
        out[label]=g
    return out


def group_classification(rows,spatial,surfaces):
    expected={(n,l)for n in LEVELS for l in LOADS}
    complete=len(rows)==33 and{(v['n'],v['label'])for v in rows}==expected and all(v.get('completed')is True for v in rows)
    result={}
    for label in LOADS:
        members=[v for v in rows if v['label']==label]
        physics=bool(complete and all(v['physical_passed']for v in members)and set(surfaces)=={str(n)for n in LEVELS}and all(v['passed']for v in surfaces.values()))
        result[label]=dict(physical_and_surface_passed=physics,spatial_passed=bool(label in spatial and spatial[label]['spatial_development_passed']),reference_valid=bool(physics and label in spatial and spatial[label]['spatial_development_passed']),all_meshes_excitation_goal_met=bool(len(members)==3 and all(v['excitation_goal_met']for v in members)))
    return dict(all33_retained=bool(complete),references=result,all11_reference_valid=bool(complete and all(v['reference_valid']for v in result.values())),all_fields_excitation_goal_met=bool(complete and all(v['excitation_goal_met']for v in rows)),bank_admitted=False)


def group_worker(group,digest,checkpoint):
    path=LOCAL/'groups'/f'group-{group["index"]:02d}'/'result-LOCAL.json';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists()or any(path.parent.iterdir()):raise FileExistsError('No groupretry/partialreuse')
    begin=time.monotonic();fields={};out=dict(completed=False,group=group,protocol_SHA256=digest,release=record(LOCAL/'release.json'),worker_pid=os.getpid(),rows=[],meshes={},queue=[dict(n=n,label=l,status='queued')for n in LEVELS for l in LOADS],new_fields=33,reused_fields=0,bank_admitted=False)
    def save():out.update(wall_s=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(path,out);checkpoint()
    try:
        E,K=group['E_Pa'],group['K_N_m'];layout=group['layout']
        for n in LEVELS:
            start=time.monotonic();f=geometry.make_forward(n,layout);loads,info=bank.load_catalog(n);surface=surface_checks(f,loads,layout);rest=E*f.Kunit_free+K*f.W_free;directions=controls.ExactRestPCG(rest);model=controls.PreconditionedP2Reference.from_forward(f,E,K,quadrature_order=80,direction_solver=directions)
            out['meshes'][str(n)]=dict(initialization_s=time.monotonic()-start,rest_factorization_s=directions.rest_factorization_s,surface=surface,nodes=len(f.X),free_DOF=len(f.free));save()
            for label in LOADS:
                item=next(v for v in out['queue']if(v['n'],v['label'])==(n,label));item['status']='running';save();force=loads[label];start=time.monotonic();uL=np.zeros_like(f.X);uL.ravel()[f.free]=directions.rest_solve(force.ravel()[f.free]);linear_R=float(np.linalg.norm(((E*f.Kunit+K*f.W)@uL.ravel()-force.ravel())[f.free]))
                if linear_R>1e-10 or np.any(uL[f.fixed]!=0):raise ValueError('Exactsame-load linearwarmstart originalR/clamp')
                linear_s=time.monotonic()-start;start=time.monotonic();u,diag=model.solve(force,initial=uL,maximum_iterations=40,force_tolerance_N=1e-10);solve_s=time.monotonic()-start;start=time.monotonic();state=model.evaluate(u,force,tangent=True);R=float(np.linalg.norm(state['residual'].ravel()[f.free]));d=geometry.deformation(f,u,finest=12);eq=controls.reaction_diagnostics(f,u,force,E,K,state['residual']);stable=controls.generalized_stability(state['tangent'][f.free][:,f.free],rest,directions.rest_LU.solve);status=field_status(d,R,bool(np.all(u[f.fixed]==0)),eq,stable,label)
                capture=path.parent/'raw-LOCAL'/f'n{n}-{label}.npz';capture.parent.mkdir(exist_ok=True);np.savez_compressed(capture,displacement_m=u,linear_displacement_m=uL,rest_m=f.X,force_N=force,elements=f.elements,exact_constraint_nodes=f.fixed,query_points_m=bank.measurement_points(),query_displacement_m=geometry.field_queries(n,u,bank.measurement_points()))
                out['rows'].append(dict(n=n,label=label,group_index=group['index'],layout=layout,E_Pa=E,K_N_m=K,completed=True,kind='NEW',**status,linear_original_R_l2_N=linear_R,original_free_R_l2_N=R,deformation=d,equilibrium=eq,stability=stable,Newton=diag,capture=record(capture),known_load=info[label],actual_deformed_wrench=controls.prior.force_wrench(f.X,force,u),linear_warm_s=linear_s,NH80_solve_s=solve_s,independent_checks_and_capture_s=time.monotonic()-start,current_LU_fallbacks_cumulative=directions.current_fallback_factorizations));fields[n,label]=u;item['status']='completed';save()
            del f,loads,model,directions,rest,state;gc.collect()
        out['spatial']=spatial_results(fields);out.update(group_classification(out['rows'],out['spatial'],{n:v['surface']for n,v in out['meshes'].items()}));out['completed']=True;save()
    except BaseException as error:out['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),accepted_Newton_history=getattr(error,'history',None));save();raise
    finally:save()
    return record(path)


def cache_metadata(rec):
    path=ROOT/rec['path']
    if record(path)!=rec:raise ValueError('Completegroup metadata bytes beforecache')
    cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(rec['sha256']+'.json')
    if cache.exists()and cache.read_bytes()!=path.read_bytes():raise ValueError('Immutablegroup cachemismatch')
    if not cache.exists():
        cache.parent.mkdir(parents=True,exist_ok=True)
        with cache.open('xb')as f:f.write(path.read_bytes())
    return record(cache)


def worker(slot,phase,digest):
    split=assignments(phase)
    if type(slot)is not int or not 0<=slot<len(split):raise ValueError('Exact phaseworkerslot')
    p=validate(digest);guard('worker',slot)
    if phase=='remaining':continuation_gate(p,digest)
    path=LOCAL/f'worker-{phase}-{slot}.json'
    if path.exists():raise FileExistsError('No workerretry')
    begin=time.monotonic();out=dict(completed=False,pid=os.getpid(),slot=slot,phase=phase,protocol_SHA256=digest,release=record(LOCAL/'release.json'),queue=[dict(index=i,status='queued')for i in split[slot]],groups=[])
    def save():guard('worker',slot);out.update(wall_s=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(path,out)
    def deadline(*_):raise TimeoutError('Frozen group5400s deadline')
    def stop(*_):raise InterruptedError('Explicit worker interruption')
    previous={s:signal.signal(s,h)for s,h in((signal.SIGALRM,deadline),(signal.SIGTERM,stop),(signal.SIGINT,stop))}
    try:
        for item in out['queue']:
            item['status']='running';out['current_group_started_monotonic_s']=time.monotonic();save();signal.alarm(LIMITS['group_wall_s']);rec=group_worker(p['plan']['groups'][item['index']],digest,save);signal.alarm(0);out['groups'].append(dict(index=item['index'],result=rec,external_cache=cache_metadata(rec)));item['status']='completed';save()
        out['completed']=True
    except BaseException as error:out['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        signal.alarm(0)
        for sig,h in previous.items():signal.signal(sig,h)
        save()
    return out


def checkpoint_gate(p,digest):
    e=json.loads((LOCAL/'execution-checkpoint.json').read_text());c=json.loads((LOCAL/'checkpoint-root-closure.json').read_text())
    if not(e.get('terminal')is True and e.get('completed')is True and 'failure'not in e and e['protocol_SHA256']==digest and e['release']==record(LOCAL/'release.json')):raise ValueError('Completed checkpointphase required')
    expected=[e['controller_pid'],*[v['pid']for v in e['workers']]]
    if not(c.get('root_reaped')is True and c.get('root_exit_code')==0 and c.get('all_owned_PID_and_groups_absent')is True and c.get('protocol_SHA256')==digest and c.get('execution_SHA256')==sha(LOCAL/'execution-checkpoint.json')and c.get('source_SHA256')==p['source_SHA256']and len(c.get('owned_pids',[]))==len(expected)and set(c['owned_pids'])==set(expected)):raise ValueError('External checkpointrootclosure exactownedidentities')
    for pid in expected:controls.prior.shared.old.absent(pid)
    if len(e['workers'])!=1:raise ValueError('One fixed checkpointworker')
    w=e['workers'][0]
    if w.get('exit_code')!=0 or w.get('cleanup_exit_code')!=0 or w.get('reaped')is not True:raise ValueError('Checkpointworker cleanreaped0')
    value=json.loads((ROOT/w['result']['path']).read_text());controls.screen.verify_records(w['result'])
    if not(value.get('completed')is True and value['pid']==w['pid']and value['phase']=='checkpoint'and value['protocol_SHA256']==digest and value['release']==e['release']and[index['index']for index in value['groups']]==[0]):raise ValueError('Exact fixedgroup0 checkpointmembership')
    controls.screen.verify_records(value['groups']);cached=value['groups'][0];g=json.loads((ROOT/cached['result']['path']).read_text())
    if(ROOT/cached['result']['path']).read_bytes()!=(ROOT/cached['external_cache']['path']).read_bytes():raise ValueError('Immutable checkpointgroup externalcache')
    if not(g.get('completed')is True and g.get('all33_retained')is True and g.get('all11_reference_valid')is True and g['group']==p['plan']['groups'][0]and g['protocol_SHA256']==digest and g['release']==e['release']):raise ValueError('Complete33field all11referencevalid costcheckpoint; negative retained, expansion blocked')
    controls.screen.verify_records([v['capture']for v in g['rows']])
    return dict(execution=record(LOCAL/'execution-checkpoint.json'),root_closure=record(LOCAL/'checkpoint-root-closure.json'),group=value['groups'][0]['result'],worker=w['result'],wall_s=e['wall_s'],all33_retained=True,all11_reference_valid=True)


def continuation_gate(p,digest):
    checkpoint=checkpoint_gate(p,digest);r=json.loads((LOCAL/'continuation-release.json').read_text())
    if not(r.get('approved')is True and r.get('protocol_SHA256')==digest and r.get('source_SHA256')==p['source_SHA256']and r.get('checkpoint')==checkpoint and r.get('resources')==LIMITS):raise ValueError('Explicit rootcostreview/remainingseven continuationrelease')
    return checkpoint


def run(phase,digest):
    p=validate(digest);guard('controller');checkpoint=continuation_gate(p,digest)if phase=='remaining'else None
    path=LOCAL/f'execution-{phase}.json'
    if path.exists():raise FileExistsError('No controllerretry')
    total_workers=len(assignments(phase));required=total_workers*LIMITS['worker_RSS_bytes']+LIMITS['controller_RSS_bytes']+LIMITS['RAM_reserve_bytes']
    if available_memory()<required:raise MemoryError('Initial aggregateworkerbudget plus30GiBreserve')
    begin=time.monotonic();children=[];e=dict(completed=False,terminal=False,phase=phase,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),workers=[],queue=[dict(index=i,status='queued')for i in range(8)],checkpoint=checkpoint,peak_aggregate_RSS_bytes=0)
    if checkpoint:e['queue'][0]['status']='completed-in-checkpoint'
    def save():e['wall_s']=time.monotonic()-begin;write(path,e)
    def stopped(*_):raise InterruptedError('Explicit ownedcontroller interruption')
    previous={s:signal.signal(s,stopped)for s in(signal.SIGTERM,signal.SIGINT)}
    try:
        for slot in range(total_workers):
            command=['taskset','-c',str(LIMITS['worker_CPUs'][slot]),sys.executable,'-u','-m','experiments.run_fixture_balanced_nh_catalog_v01','worker','--slot',str(slot),'--phase',phase,'--sha',digest]
            with(LOCAL/f'worker-{phase}-{slot}.log').open('xb')as log:child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child);e['workers'].append(dict(pid=child.pid,slot=slot,groups=assignments(phase)[slot],command=command,started_monotonic_s=time.monotonic()));save()
        while any(c.poll()is None for c in children):
            guard('controller');rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            for child,row in zip(children,e['workers']):
                if child.poll()is None:
                    value=controls.prior.shared.owned_RSS(child);rss+=value;row['peak_RSS_bytes']=max(row.get('peak_RSS_bytes',0),value)
                    if value>=LIMITS['worker_RSS_bytes']:raise MemoryError('Ownedworker18GiB')
                    current=LOCAL/f'worker-{phase}-{row["slot"]}.json'
                    group_started=row['started_monotonic_s']
                    if current.exists():group_started=json.loads(current.read_text()).get('current_group_started_monotonic_s',group_started)
                    if time.monotonic()-group_started>LIMITS['group_wall_s']+1:raise TimeoutError('Ownedcurrentgroup hardcap')
                elif child.returncode!=0:raise RuntimeError('Workertechnicalfailure, no newgroups/retries')
            e['peak_aggregate_RSS_bytes']=max(e['peak_aggregate_RSS_bytes'],rss)
            if rss>=LIMITS['aggregate_RSS_bytes']:raise MemoryError('Ownedaggregate76GiB')
            if time.monotonic()-begin+(checkpoint['wall_s']if checkpoint else 0)>LIMITS['global_wall_s']:raise TimeoutError('Combinedcheckpoint+remaining21600s')
            save();time.sleep(.1)
        for child,row in zip(children,e['workers']):
            row['exit_code']=child.wait();row['cleanup_exit_code']=controls.prior.shared.old.cost.cleanup(child);row['reaped']=True
            if row['exit_code']!=0 or row['cleanup_exit_code']!=0:raise RuntimeError('Ownedworker wait/cleanup failure')
            target=LOCAL/f'worker-{phase}-{row["slot"]}.json';v=json.loads(target.read_text());row['result']=record(target)
            if not(v.get('completed')is True and 'failure'not in v and v['pid']==row['pid']and v['slot']==row['slot']and v['phase']==phase and v['protocol_SHA256']==digest and v['release']==e['release']and[x['index']for x in v['groups']]==row['groups']):raise ValueError('Exactworkercompletion/memberships')
            controls.screen.verify_records(v['groups'])
            for g in row['groups']:e['queue'][g]['status']='completed'
        e['completed']=True
    except BaseException as error:e['failure']=dict(type=type(error).__name__,message=str(error))
    finally:
        for child,row in zip(children,e['workers']):
            row.update(cleanup_exit_code=controls.prior.shared.old.cost.cleanup(child),exit_code=child.returncode,reaped=True)
            if row['exit_code']!=0 or row['cleanup_exit_code']!=0:e['completed']=False
        for sig,h in previous.items():signal.signal(sig,h)
        e['terminal']=True;save()
    return e

def phase_records(phase,p,digest):
    """Root-reaped process and byte barrier only; never deserialize NPZ."""
    path=LOCAL/f'execution-{phase}.json';e=json.loads(path.read_text());closure_path=LOCAL/f'{phase}-root-closure.json';c=json.loads(closure_path.read_text())
    if not(e.get('terminal')is True and e.get('completed')is True and 'failure'not in e and e['phase']==phase and e['protocol_SHA256']==digest and e['release']==record(LOCAL/'release.json')):raise ValueError('Complete clean exact phase')
    ids=[e['controller_pid'],*[w['pid']for w in e['workers']]]
    if not(c.get('root_reaped')is True and c.get('root_exit_code')==0 and c.get('all_owned_PID_and_groups_absent')is True and c.get('protocol_SHA256')==digest and c.get('execution_SHA256')==sha(path)and c.get('source_SHA256')==p['source_SHA256']and len(c.get('owned_pids',[]))==len(ids)and set(c['owned_pids'])==set(ids)):raise ValueError('Exact externalrootphaseclosure')
    for pid in ids:controls.prior.shared.old.absent(pid)
    expected=assignments(phase)
    if len(e['workers'])!=len(expected)or e['peak_aggregate_RSS_bytes']>LIMITS['aggregate_RSS_bytes']:raise ValueError('Complete phaseworkers/resourcebound')
    results=[]
    for slot,w in enumerate(e['workers']):
        if not(w['slot']==slot and w['groups']==expected[slot]and w.get('exit_code')==0 and w.get('cleanup_exit_code')==0 and w.get('reaped')is True):raise ValueError('Exact phaseworkers cleanreaped0')
        target=LOCAL/f'worker-{phase}-{slot}.json'
        if w['result']!=record(target):raise ValueError('Exact workerresult bytes')
        v=json.loads(target.read_text())
        if not(v.get('completed')is True and 'failure'not in v and v['pid']==w['pid']and v['slot']==slot and v['phase']==phase and v['protocol_SHA256']==digest and v['release']==e['release']and[x['index']for x in v['groups']]==expected[slot]and v['queue']==[dict(index=i,status='completed')for i in expected[slot]]and v['peak_RSS_bytes']<=LIMITS['worker_RSS_bytes']):raise ValueError('Exact workercontext/queue/resource')
        for item in v['groups']:
            index=item['index'];group_path=LOCAL/'groups'/f'group-{index:02d}'/'result-LOCAL.json'
            if item['result']!=record(group_path):raise ValueError('Exact groupresult bytes')
            controls.screen.verify_records(item['external_cache'])
            if group_path.read_bytes()!=(ROOT/item['external_cache']['path']).read_bytes():raise ValueError('Immutable completedgroup externalcache')
            g=json.loads(group_path.read_text());members={(n,l)for n in LEVELS for l in LOADS}
            if not(g.get('completed')is True and 'failure'not in g and g['group']==p['plan']['groups'][index]and g['protocol_SHA256']==digest and g['release']==e['release']and g['worker_pid']==w['pid']and g['new_fields']==33 and g['reused_fields']==0 and len(g['rows'])==33 and{(r['n'],r['label'])for r in g['rows']}==members and g['wall_s']<=LIMITS['group_wall_s']and g['peak_RSS_bytes']<=LIMITS['worker_RSS_bytes']):raise ValueError('All33 exact NEW groupcontexts/resource')
            for r in g['rows']:
                cap=group_path.parent/'raw-LOCAL'/f"n{r['n']}-{r['label']}.npz"
                if not(r.get('completed')is True and r['group_index']==index and r['layout']==g['group']['layout']and(r['E_Pa'],r['K_N_m'])==(g['group']['E_Pa'],g['group']['K_N_m'])and r['kind']=='NEW'and r['capture']==record(cap)):raise ValueError('Everycapture byte/context before arrayloading')
            classified=group_classification(g['rows'],g['spatial'],{n:m['surface']for n,m in g['meshes'].items()})
            if any(g[k]!=val for k,val in classified.items()):raise ValueError('Exact groupreference/excitation classification')
            results.append(dict(index=index,result=item['result'],value=g))
    return dict(execution=record(path),root_closure=record(closure_path),wall_s=e['wall_s']),results


def terminal_records(digest):
    p=validate(digest);checkpoint,left=phase_records('checkpoint',p,digest);remaining,right=phase_records('remaining',p,digest)
    if checkpoint['wall_s']+remaining['wall_s']>LIMITS['global_wall_s']:raise ValueError('Combined global21600s')
    continuation_gate(p,digest);allgroups=sorted(left+right,key=lambda v:v['index'])
    if[v['index']for v in allgroups]!=list(range(8)):raise ValueError('All8 originalgroups no replacements')
    return p,dict(checkpoint=checkpoint,remaining=remaining),allgroups


def finalize(digest):
    target=LOCAL/'summary-LOCAL.json'
    if target.exists():raise FileExistsError('No finalsummary overwrite')
    p,phases,allgroups=terminal_records(digest)
    summary=dict(schema='fixture-balanced-nh-catalog-summary-v01',completed=True,protocol=record(LOCAL/'protocol.json'),release=record(LOCAL/'release.json'),source_SHA256=p['source_SHA256'],phases=phases,groups=[dict(index=v['index'],group=v['value']['group'],result=v['result'],completed=True,reference_valid=v['value']['all11_reference_valid'],references=v['value']['references'],captures=[dict(n=r['n'],label=r['label'],capture=r['capture'])for r in v['value']['rows']])for v in allgroups],all264_retained=True,new_fields=264,reused_fields=0,group_load_references=88,all88_reference_valid=all(v['value']['all11_reference_valid']for v in allgroups),all_fields_excitation_goal_met=all(v['value']['all_fields_excitation_goal_met']for v in allgroups),bank_admitted=False,scope=p['plan']['scope'])
    write(target,summary);return record(target)


def completed_catalog_gate():
    """Downstream metadata admission: no RNG, truth array or solver calls."""
    digest=sha(LOCAL/'protocol.json');p,phases,allgroups=terminal_records(digest);s=json.loads((LOCAL/'summary-LOCAL.json').read_text());a=json.loads((LOCAL/'independent-review.json').read_text())
    if not(s.get('completed')is True and s.get('all264_retained')is True and s.get('all88_reference_valid')is True and s.get('new_fields')==264 and s.get('reused_fields')==0 and len(s.get('groups',[]))==8 and s['protocol']==record(LOCAL/'protocol.json')and s['release']==record(LOCAL/'release.json')and s['source_SHA256']==p['source_SHA256']and s['phases']==phases):raise ValueError('Complete all88referencevalid admission; excitation is separate')
    if not(a.get('audit_passed')is True and a.get('all264_retained')is True and a.get('all88_reference_valid')is True and a.get('source_SHA256')==p['source_SHA256']and a.get('source_summary')==record(LOCAL/'summary-LOCAL.json')):raise ValueError('Independent pre-pinned all264 savedfieldaudit required')
    expected_audit={name:sha(ROOT/name)for name in(AUDIT_SOURCE,AUDIT_TEST)}
    if any(p['source_SHA256'].get(name)!=digest for name,digest in expected_audit.items())or a.get('audit_source')!=record(ROOT/AUDIT_SOURCE)or a.get('audit_test')!=record(ROOT/AUDIT_TEST):raise ValueError('Exact prospectively pinned independent auditprocedure/source/test')
    controls.screen.verify_records(s['groups'])
    return dict(protocol=record(LOCAL/'protocol.json'),release=record(LOCAL/'release.json'),phases=phases,summary=record(LOCAL/'summary-LOCAL.json'),independent_review=record(LOCAL/'independent-review.json'),groups=[dict(index=v['index'],group=v['value']['group'],result=v['result'],rows=[dict(n=r['n'],label=r['label'],capture=r['capture'])for r in v['value']['rows']])for v in allgroups],new_fields=264,reused_fields=0,all88_reference_valid=True,scope='Eight fixed interior development groups only; no populationcalibration or policyadvantage admission.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','worker','guard','finalize']);ap.add_argument('--phase',choices=['checkpoint','remaining']);ap.add_argument('--sha');ap.add_argument('--slot',type=int);ap.add_argument('--role',choices=['controller','worker']);a=ap.parse_args()
    if a.action=='guard':guard(a.role,a.slot);print('guard-PASS; no model/data')
    elif a.action=='prepare':print(json.dumps(prepare()))
    elif a.action=='finalize':print(json.dumps(finalize(a.sha)))
    elif a.action=='worker':print(json.dumps(worker(a.slot,a.phase,a.sha)))
    else:r=run(a.phase,a.sha);print(json.dumps(r));raise SystemExit(0 if r['completed']else 1)
