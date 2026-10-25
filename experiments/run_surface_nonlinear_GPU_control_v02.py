"""New canonical-label loading control; previous technical failure preserved.

Preparation is metadata-only. Explicit root release and closed qualified CPU
cost are mandatory before archives, new control noise, or GPU initialization.
"""
import argparse,ast,json,os,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_surface_nonlinear_posterior_cost_v03 as cost
from experiments import run_surface_nonlinear_GPU_control_v01 as previous
from experiments import run_surface_full_atlas_feasibility_v01 as full
from experiments import run_surface_p2_study_v04 as study
from experiments.campaign_nonlinear_conditional_reference_v03 import NonlinearConditionalReferenceV03
from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas
from experiments.campaign_nonlinear_residual_atlas_fast_v01 import FastResidualAtlas
from experiments.campaign_surface_conditional_amplitude_v02 import AmplitudePWLPlan
ROOT=cost.ROOT;LOCAL=ROOT/'runs/operator-dev/surface-nonlinear-GPU-control-v02'
SOURCE='experiments/run_surface_nonlinear_GPU_control_v02.py';TEST='tests/test_surface_nonlinear_GPU_control_v02.py'
CPU_COST_SHA='3a98be21beb6e11fccd06224fe9d2809a69cb293be3ad074076f435ac583543b'
PREFIX='normal_x180';AXIAL='paired_axial_x180';GPU_UUIDS=study.GPU_UUIDS
RESOURCES=dict(worker_CPUs=[0,1,2],controller_CPU=16,worker_wall_s=1800,global_wall_s=2100,worker_RSS_bytes=4*1024**3,controller_RSS_bytes=4*1024**3,aggregate_RSS_bytes=16*1024**3,torch_allocated_bytes=512*1024**2,owned_VRAM_bytes_per_GPU=2*1024**3,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=512*1024**2)
CASES=[dict(slot=0,unit=[.5,.5],labels=[PREFIX],noise_seeds=[],saved_prefix=True,CPU_parity=True),dict(slot=1,unit=[.5,.5],labels=[PREFIX,PREFIX],noise_seeds=[2026103001],saved_prefix=True,CPU_parity=False),dict(slot=2,unit=[0.,1.],labels=[PREFIX,AXIAL],noise_seeds=[2026103002,2026103003],saved_prefix=False,CPU_parity=False)]
sha,record,write=cost.sha,cost.record,cost.write


def sources():
    pending=[SOURCE,TEST];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:modules=[node.module or '']+[(node.module+'.'+x.name)for x in node.names if node.module]
            for module in modules:
                path=module.replace('.','/')+'.py'
                if(ROOT/path).is_file()and path not in seen:pending.append(path)
    return {name:sha(ROOT/name)for name in sorted(seen)}


def guard(slot=None):
    cpu=16 if slot is None else slot;device=''if slot is None else GPU_UUIDS[slot]
    if slot is not None and slot not in range(3):raise ValueError('Exactly three fixed slots')
    if set(os.sched_getaffinity(0))!={cpu}or os.environ.get('CUDA_VISIBLE_DEVICES')!=device or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))or(slot is not None and os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):raise ValueError('Exact owned CPU/thread/GPU UUID guard')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3:raise MemoryError('Own4GiB RSS')
    if study.available_memory()<RESOURCES['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise MemoryError('30GiB RAM/disk reserve')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise OSError('Own output cap')


V01_PROTOCOL_SHA='d86d0e9e6c9d4eb4989b93dbcb2bd2d56ad72ae1451978dc9afd9ee2c5da988f'


def failed_v01_gate():
    """Bind the frozen technical failure/closure before any new execution."""
    previous.validate(V01_PROTOCOL_SHA);previous.release_gate(V01_PROTOCOL_SHA)
    path=previous.LOCAL/'execution.json';e=json.loads(path.read_text())
    if e.get('terminal')is not True or e.get('completed')is not False or e.get('qualified')is not False or len(e.get('workers',[]))!=3 or sorted(row['slot']for row in e['workers'])!=[0,1,2]or e.get('protocol_SHA256')!=V01_PROTOCOL_SHA or e.get('release')!=record(previous.LOCAL/'release.json'):raise ValueError('Frozen v01 technical negative required')
    cost.old.absent(e['controller_pid']);results=[]
    for row in e['workers']:
        cost.old.absent(row['pid']);target=previous.LOCAL/f"worker-{row['slot']}/result-LOCAL.json"
        if row.get('reaped')is not True or row.get('exit_code')!=1 or row.get('result')!=record(target):raise ValueError('All old workers reaped1/exact failure records')
        w=json.loads(target.read_text())
        if w.get('pid')!=row['pid']or w.get('protocol_SHA256')!=V01_PROTOCOL_SHA or w.get('release')!=e['release']or w.get('completed')is not False or w.get('passed')is not False or w.get('failure',{}).get('message')!='Admitted n9 atlas fingerprint'or'observations'in w or (target.parent/'observations-LOCAL.npz').exists():raise ValueError('Exact pre-observation fingerprint failure/no noise archive')
        results.append(record(target))
    return dict(protocol=record(previous.LOCAL/'protocol.json'),release=record(previous.LOCAL/'release.json'),execution=record(path),results=results,source_SHA256=previous.sources(),no_control_observations_generated=True)


def admitted_atlas(origin):
    """Canonical Python labels preserve the ORIGINAL admitted fingerprint."""
    path=ROOT/origin['full_atlas']['archive']['path']
    if sha(path)!=origin['full_atlas']['archive']['sha256']:raise ValueError('Exact admitted atlas raw bytes')
    with np.load(path,allow_pickle=False)as z:
        labels=tuple(str(v)for v in z['labels'])
        base=NonlinearResidualAtlas(z['delta'],labels,provenance_fingerprint=origin['full_protocol_SHA256'])
        if not np.array_equal(z['axis'],base.axis):raise ValueError('Exact admitted axis')
    if base.fingerprint!=origin['atlas_fingerprint']or list(base.labels)!=origin['labels']:raise ValueError('Canonical admitted n9 atlas fingerprint/labels')
    return base


def prerequisite():
    receipt=cost.completed_cost_gate(CPU_COST_SHA);p,_=cost.validate(CPU_COST_SHA)
    center=cost.old.truth_gate()
    corner={}
    for label in(PREFIX,AXIAL):
        matches=list(full.LOCAL.glob('worker-*/point-008-'+label+'.json'))
        if len(matches)!=1:raise ValueError('Unique admitted corner record')
        path=matches[0];row=json.loads(path.read_text())
        if row.get('index')!=8 or row.get('label')!=label or row.get('passed')is not True or row.get('completed')is not True or row.get('protocol_SHA256')!=p['full_protocol_SHA256']or sha(ROOT/row['raw']['path'])!=row['raw']['sha256']:raise ValueError('Admitted corner point008 original context/hash')
        corner[label]=dict(metadata=record(path),raw=row['raw'])
    moments=json.loads((cost.LOCAL/'worker-result-LOCAL.json').read_text())['atlas_results']['9']['posterior_moments']
    if sha(ROOT/moments['path'])!=moments['sha256']:raise ValueError('Qualified CPU n9 moments')
    return dict(negative_GPU_v01=failed_v01_gate(),CPU_cost=receipt,center=center,corner=corner,CPU_moments=moments,full_atlas=p['origin_v01']['full_atlas'],full_protocol_SHA256=p['full_protocol_SHA256'],atlas_fingerprint=p['full_atlas_fingerprint'],labels=p['all_target_labels'],saved_observation=p['origin_v01']['observation'])


def prepare():
    guard();origin=prerequisite()
    if study.available_memory()<RESOURCES['aggregate_RSS_bytes']+RESOURCES['RAM_reserve_bytes']:raise MemoryError('16GiB owned +30GiB reserve')
    p=dict(schema='nonlinear-three-GPU-control-v02',source_SHA256=sources(),origin=origin,resources=RESOURCES,cases=CASES,GPU_UUIDs=list(GPU_UUIDS),plan=dict(inner=[128,256,512],outer=[128,256,512],moment_chunk=1024),new_PDE=False,fresh64_access=False,noise_scope='Only prospective control seeds3001/3002/3003 after release; independent prefix/reset draws',gates=dict(original_qualifier_and_adaptive=True,additional_absolute_refinement_m=2e-6,CPU_predictive_parity_m=2e-12,CPU_CDF_parity=1e-8))
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['origin']!=prerequisite()or p['resources']!=RESOURCES or p['cases']!=CASES or p['GPU_UUIDs']!=list(GPU_UUIDS):raise ValueError('Frozen source/protocol/cost/fields/resources')
    return p


def release_gate(digest):
    p=validate(digest);r=json.loads((LOCAL/'release.json').read_text())
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']:raise ValueError('Root release required before arrays/noise/GPU')
    return p


def observations(slot,origin,fwd):
    """Only called after release; new noise is independent reset noise."""
    from experiments.campaign_surface_p2_static_v02 import point_query_matrix
    from experiments.campaign_surface_p2_study_design_v01 import observation_points
    def load(rec):
        path=ROOT/rec['path']
        if sha(path)!=rec['sha256']:raise ValueError('Bound input raw bytes')
        return np.load(path,allow_pickle=False)
    if slot in(0,1):
        with load(origin['saved_observation'])as z:prefix=z['observation_m'].copy()
        result=[(PREFIX,prefix)]
        if slot==1:
            with load(origin['center']['capture'])as z:u=z['displacement_m']
            Q=point_query_matrix(fwd,observation_points());direct=np.asarray(Q@u.ravel()[fwd.free])
            result.append((PREFIX,direct+np.random.default_rng(2026103001).normal(0,20e-6,24)))
    else:
        result=[]
        for label,seed in zip((PREFIX,AXIAL),(2026103002,2026103003)):
            with load(origin['corner'][label]['raw'])as z:
                if not np.array_equal(z['unit'],[0.,1.]):raise ValueError('Original point008 must be unit(0,1)')
                direct=z['response_m'].copy()
            result.append((label,direct+np.random.default_rng(seed).normal(0,20e-6,24)))
    if any(y.shape!=(24,)or not np.isfinite(y).all()for _,y in result):raise ValueError('Finite independent24channel observations')
    return result


def parity(final,baseline):
    errors={key:float(np.max(abs(final[key]-baseline[key])))for key in('predictive_mean','predictive_SD','unit_CDF')}
    return dict(passed=bool(errors['predictive_mean']<=2e-12 and errors['predictive_SD']<=2e-12 and errors['unit_CDF']<=1e-8),maximum_errors=errors)


def exact_ratio_callback(model):
    """Planned study path: CPU singleton/oracle, FP64 GPU batched >=16 rows."""
    CPU=study.RatioModal(model.modal,model.labels)
    def evaluate(z,labels):return model.ratio_select(labels)(z)if hasattr(model,'ratio_select')and len(z)>=16 else CPU(np.exp(z),labels)
    return study.ExactRatioMemo(evaluate,model_fingerprint=study.model_fingerprint(model),maximum_bytes=256*1024**2,minimum_rows=1)


def worker(slot,digest):
    guard(slot);p=release_gate(digest);directory=LOCAL/f'worker-{slot}';directory.mkdir(exist_ok=False)
    start=time.monotonic();state=dict(slot=slot,pid=os.getpid(),completed=False,passed=False,protocol_SHA256=digest,release=record(LOCAL/'release.json'),new_PDE_calls=0,fresh64_draws=0)
    def save():state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(directory/'result-LOCAL.json',state)
    def stop(*_):raise TimeoutError('Prospective1800s GPU control worker cap')
    previous=signal.signal(signal.SIGALRM,stop);signal.alarm(1800)
    try:
        fwd,model=study.make_model(device_mode='gpu',slot=slot)
        memo=exact_ratio_callback(model)
        origin=p['origin'];base=admitted_atlas(origin)
        atlas=FastResidualAtlas(base);obs=observations(slot,origin,fwd);np.savez_compressed(directory/'observations-LOCAL.npz',**{f'observation_{i}':y for i,(_,y)in enumerate(obs)})
        state['observations']=record(directory/'observations-LOCAL.npz');state['observation_labels']=[label for label,_ in obs];save()
        engine=NonlinearConditionalReferenceV03(memo,None,obs,origin['labels'],residual_predict=atlas.predict,retain_hypotheses=False,evaluation_chunk=1024)
        result=engine.qualify(plan=AmplitudePWLPlan(moment_chunk=1024));checks=cost.engine_gates(result);state.update(engine_qualified=bool(result['qualified']),all_engine_checks=checks,diagnostics=result['diagnostics'])
        final=result['local_final']
        if final is not None:
            keys=['unit_mean','unit_covariance','unit_CDF','unit_quantiles','predictive_mean','predictive_SD'];np.savez_compressed(directory/'posterior-LOCAL.npz',**{key:final[key]for key in keys});state['posterior']=record(directory/'posterior-LOCAL.npz')
            if slot==0:
                with np.load(ROOT/origin['CPU_moments']['path'],allow_pickle=False)as z:state['CPU_parity']=parity(final,{key:z[key]for key in keys})
        state.update(completed=True,passed=bool(checks['passed']and(slot!=0 or state.get('CPU_parity',{}).get('passed',False))),memo=memo.diagnostics(),torch_peak_allocated_bytes=model.torch.cuda.max_memory_allocated(model.device))
        if state['torch_peak_allocated_bytes']>RESOURCES['torch_allocated_bytes']:raise MemoryError('Owned Torch512MiB cap')
        guard(slot);save()
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());save();raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous);save()
    return state


def worker_command(slot,digest,*,guard_only=False):return ['taskset','-c',str(slot),sys.executable,'-u','-m','experiments.run_surface_nonlinear_GPU_control_v02','worker-guard'if guard_only else'worker','--slot',str(slot),'--sha',digest]


def run(digest):
    guard();p=release_gate(digest)
    if(LOCAL/'execution.json').exists():raise FileExistsError('No implicit retry')
    start=time.monotonic();children=[];logs=[];state=dict(completed=False,qualified=False,terminal=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),workers=[])
    def save():state['wall_s']=time.monotonic()-start;write(LOCAL/'execution.json',state)
    def stopped(*_):raise InterruptedError('Explicit owned controller stop')
    previous={sig:signal.signal(sig,stopped)for sig in(signal.SIGTERM,signal.SIGINT)}
    try:
        for slot in range(3):
            env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU_UUIDS[slot],'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'};log=(LOCAL/f'worker-{slot}.log').open('xb');logs.append(log);child=subprocess.Popen(worker_command(slot,digest),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(child);state['workers'].append(dict(slot=slot,pid=child.pid,reaped=False,started_s=time.monotonic()-start));save()
        while any(child.poll()is None for child in children):
            guard();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            for child,row in zip(children,state['workers']):
                if child.poll()is not None:row.update(reaped=True,exit_code=child.wait());continue
                own=full.owned_RSS(child);rss+=own
                if own>RESOURCES['worker_RSS_bytes']or time.monotonic()-start-row['started_s']>1800:raise TimeoutError('WorkerRSS/deadline')
            state['maximum_aggregate_RSS_bytes']=max(state.get('maximum_aggregate_RSS_bytes',0),rss)
            if rss>16*1024**3 or time.monotonic()-start>2100:raise TimeoutError('AggregateRSS/globaldeadline')
            state['owned_GPU_usage']=study.GPU_usage([c.pid for c in children],cap_bytes=RESOURCES['owned_VRAM_bytes_per_GPU']);save();time.sleep(.2)
        for child,row in zip(children,state['workers']):row.update(reaped=True,exit_code=child.wait())
        if any(c.returncode!=0 for c in children):raise RuntimeError('Worker execution failed; retain negatives')
        state['completed']=True
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error))
    finally:
        for child,row in zip(children,state['workers']):
            row.update(reaped=True,exit_code=cost.old.cost.cleanup(child));cost.old.absent(child.pid)
            path=LOCAL/f"worker-{row['slot']}/result-LOCAL.json"
            if path.exists():row['result']=record(path)
        state['qualified']=bool(state['completed']and'failure'not in state and all(row.get('exit_code')==0 and row.get('result')and json.loads((ROOT/row['result']['path']).read_text()).get('passed')is True for row in state['workers']))
        for log in logs:log.close()
        for sig,handler in previous.items():signal.signal(sig,handler)
        state['terminal']=True;save()
    return state


def completed_GPU_control_gate(digest):
    p=validate(digest);release_gate(digest);e=json.loads((LOCAL/'execution.json').read_text())
    if not all(e.get(k)is True for k in('completed','qualified','terminal'))or'failure'in e or e.get('protocol_SHA256')!=digest or e.get('release')!=record(LOCAL/'release.json')or len(e.get('workers',[]))!=3 or sorted(r['slot']for r in e['workers'])!=[0,1,2]:raise ValueError('Closed qualified three-GPU numerical control required')
    cost.old.absent(e['controller_pid'])
    for row in e['workers']:
        cost.old.absent(row['pid']);path=LOCAL/f"worker-{row['slot']}/result-LOCAL.json"
        if row.get('reaped')is not True or row.get('exit_code')!=0 or row.get('result')!=record(path):raise ValueError('Every owned worker reaped0/exact result')
        w=json.loads(path.read_text())
        if w.get('completed')is not True or w.get('passed')is not True or'failure'in w or w.get('protocol_SHA256')!=digest or w.get('pid')!=row['pid']or w.get('release')!=e['release']or not cost.engine_gates(dict(qualified=w['engine_qualified'],diagnostics=w['diagnostics']))['passed']or w['all_engine_checks']!=cost.engine_gates(dict(qualified=w['engine_qualified'],diagnostics=w['diagnostics']))or(row['slot']==0 and not w['CPU_parity']['passed']):raise ValueError('All original/adaptive/absolute/CPU parity gates')
        for key in('observations','posterior'):
            if sha(ROOT/w[key]['path'])!=w[key]['sha256']:raise ValueError('Immutable worker archive hash')
    return dict(protocol=record(LOCAL/'protocol.json'),release=record(LOCAL/'release.json'),execution=record(LOCAL/'execution.json'),source_SHA256=p['source_SHA256'],cases=CASES,all_checks_passed=True,wall_s=e['wall_s'],maximum_aggregate_RSS_bytes=e['maximum_aggregate_RSS_bytes'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','run','worker-guard']);parser.add_argument('--sha');parser.add_argument('--slot',type=int);args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare()))
    elif args.action=='worker':r=worker(args.slot,args.sha);print(json.dumps(dict(completed=r['completed'],qualified=r['passed'])))
    elif args.action=='worker-guard':guard(args.slot);print('guard passed')
    else:r=run(args.sha);print(json.dumps(r));raise SystemExit(0 if r['completed']and r['qualified']else 1)
