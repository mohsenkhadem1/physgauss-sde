"""Root-released same-failed-observation 1024-panel numerical control.

Never reads true parameters, held responses, PIT, coverage or forecast error.
Original fresh64 run, failures and draws remain immutable. These are three controls,
not continuation/admission of the original cohort. No RNG or PDE entry point.
"""
from __future__ import annotations
import argparse,ast,json,os,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_fresh64_nonlinear_calibration_v01 as fresh
from experiments import run_surface_nonlinear_posterior_cost_v03 as cost
from experiments.campaign_nonlinear_conditional_reference_v04 import NonlinearConditionalReferenceV04,ExtendedAmplitudePWLPlan
ROOT=fresh.ROOT
LOCAL=ROOT/'runs/operator-dev/fresh64-inner-resolution-control-v01'
SOURCE='experiments/run_fresh64_inner_resolution_control_v01.py'
TEST='tests/test_fresh64_inner_resolution_control_v01.py'
PRIMARY_SHA='ab1ecddf49342ec337401f2629d681ec88dc4f1424986872f9ed173dcbd20323'
PREFIX='normal_x180'
CASES=(dict(id='object05-nonlinear-normal-r0-inner',index=5,phase='normal-r0',kind='nonlinear',axis='inner'),dict(id='object01-nonlinear-axial-r0-outer',index=1,phase='axial-r0',kind='nonlinear',axis='outer'),dict(id='object01-linear-axial-r0-outer',index=1,phase='axial-r0',kind='linear',axis='outer'))
RESOURCES=dict(worker_CPU=17,controller_CPU=18,threads=1,GPU=False,worker_RSS_cap_bytes=4*1024**3,controller_RSS_cap_bytes=4*1024**3,aggregate_RSS_cap_bytes=8*1024**3,wall_cap_s=2400,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=512*1024**2)
def case_plan(case):
    return ExtendedAmplitudePWLPlan(inner=(128,256,512,1024)if case['axis']=='inner'else(128,256,512),outer=(128,256,512,1024)if case['axis']=='outer'else(128,256,512),moment_chunk=1024)


def canonical_cases():
    return [dict(case,plan=dict(inner=list(case_plan(case).inner),outer=list(case_plan(case).outer),moment_chunk=1024))for case in CASES]
PROJECTION=dict(per_case={case['id']:dict(final_outer_rows=3*case_plan(case).outer[-1],atoms_per_row=(4 if case['kind']=='nonlinear'else 2)*case_plan(case).inner[-1],maximum_hypotheses=3*case_plan(case).outer[-1]*(4 if case['kind']=='nonlinear'else 2)*case_plan(case).inner[-1])for case in CASES},maximum_hypotheses=6291456,target_chunk=1024,target_channels=96,target_chunk_bytes=786432,retain_hypotheses=False,full_hypothesis_by_target_array_allocated=False,actual_4GiB_per_process_guard_authoritative=True)
sha,record,write=cost.sha,cost.record,cost.write


def sources():
    pending=[SOURCE,TEST,'tests/test_nonlinear_inner_extension_v01.py'];seen=set()
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
                if(ROOT/path).is_file()and path not in seen:pending.append(path)
    return {name:sha(ROOT/name)for name in sorted(seen)}


def guard(role):
    if role not in('worker','controller'):raise ValueError('Declared process role')
    if set(os.sched_getaffinity(0))!={RESOURCES[role+'_CPU']}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact CPU17worker/18controller/one-thread/noGPU')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES[role+'_RSS_cap_bytes']:raise MemoryError('Per-process4GiB cap')
    if fresh.oldstudy.available_memory()<RESOURCES['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise MemoryError('30GiB RAM/disk reserves')
    if LOCAL.exists()and sum(f.stat().st_size for f in LOCAL.rglob('*')if f.is_file())>RESOURCES['maximum_writes_bytes']:raise OSError('Own512MiB output cap')


def failed_context(row,case=CASES[0]):
    """Read only numerical admission and hash context; never calibration fields."""
    c=row.get('context',{});d=row.get('diagnostics',{})
    backend='nonlinear_conditional_reference_v03_MIS'if case['kind']=='nonlinear'else'conditional_amplitude_PWL_v02'
    schema='nonlinear-conditional-reference-v03'if case['kind']=='nonlinear'else'conditional-amplitude-v02'
    if row.get('qualified')is not False or row.get('physical_valid')is not True or row.get('kind')!=case['kind']or row.get('numerical_backend')!=backend:raise ValueError('Exact original physical-valid numerical negative')
    if c.get('protocol_SHA256')!=PRIMARY_SHA or c.get('index')!=case['index']or c.get('phase')!=case['phase']or d.get('schema')!=schema:raise ValueError('Exact original failed context')
    if d.get('qualified')is not False or not all(r.get('status')=='computed'for r in d.get('levels',[]))or len(d.get('levels',[]))!=5:raise ValueError('All five original levels computed')
    if not row.get('raw_SHA256')or not all(c.get(k)for k in('draws_SHA256','observations_SHA256','model_fingerprint','recipe_SHA256')):raise ValueError('Original numerical archive/observation/model hashes')
    return dict(context=c,diagnostics=d,raw_SHA256=row['raw_SHA256'])


def origin_gate():
    """Hash/commit metadata only. No primary terminal requirement during control."""
    p=fresh.release_gate(PRIMARY_SHA);rows={}
    for case in CASES:
        index=case['index'];directory=fresh.LOCAL/f'object-{index:02d}'
        path=directory/(case['kind']+'-'+case['phase']+'-posterior-LOCAL.json')
        row=failed_context(json.loads(path.read_text()),case);raw=path.with_suffix('.npz')
        if sha(raw)!=row['raw_SHA256']:raise ValueError('Original failed archive checksum; never deserialize')
        draws=directory/'draws-LOCAL.npz'
        fresh.verify(draws,dict(protocol_SHA256=PRIMARY_SHA,index=index,kind='freshparameter_and_independentnoise'))
        truths={}
        for label in(PREFIX,)if case['phase']=='normal-r0'else(PREFIX,'paired_axial_x180'):
            truth=directory/(label+'-truth-LOCAL.npz')
            fresh.verify(truth,dict(protocol_SHA256=PRIMARY_SHA,index=index,label=label,draws_SHA256=sha(draws)))
            truths[label]=dict(archive=record(truth),commit=record(truth.with_suffix('.commit.json')))
        if row['context']['draws_SHA256']!=sha(draws)or row['context']['model_fingerprint']!=p['cost_receipt']['memo']['model_fingerprint']:raise ValueError('Original data and qualified modal identity')
        rows[case['id']]=dict(failed_record=record(path),failed_raw=record(raw),draws=record(draws),draws_commit=record(draws.with_suffix('.commit.json')),truth_responses=truths,context=row['context'],original_diagnostics=row['diagnostics'])
    return p,dict(primary_protocol=record(fresh.LOCAL/'protocol.json'),primary_release=record(fresh.LOCAL/'release.json'),cases=rows)


def prepare():
    guard('controller');p,origin=origin_gate()
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    recipe=dict(schema='fresh64-inner-resolution-control-v01',status='FROZEN_ROOT_RELEASE_REQUIRED',source_SHA256=sources(),origin=origin,resources=RESOURCES,resource_projection=PROJECTION,cases=canonical_cases(),atlas=p['full_atlas'],atlas_fingerprint=p['atlas_fingerprint'],targets=p['target_labels'],sigma_m=20e-6,new_RNG_calls=0,new_PDE_calls=0,true_parameters_or_held_truth_read=False,old_primary_negative_preserved=True,scope='Three unchanged failed observations, one-axis numerical refinement only; no cohort continuation/admission, no scientific metrics')
    write(LOCAL/'protocol.json',recipe);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text());fp,origin=origin_gate()
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['origin']!=origin or p['resources']!=RESOURCES or p['resource_projection']!=PROJECTION:raise ValueError('Exact new frozen source/protocol/origin/resources')
    if p['cases']!=canonical_cases()or p['atlas']!=fp['full_atlas']or p['atlas_fingerprint']!=fp['atlas_fingerprint']or p['targets']!=fp['target_labels']or p['sigma_m']!=20e-6 or p['new_RNG_calls']!=0 or p['new_PDE_calls']!=0 or p['true_parameters_or_held_truth_read']is not False:raise ValueError('Only declared panel extension, no truth/noise/physical change')
    return p,fp


def release_gate(p,digest):
    r=json.loads((LOCAL/'release.json').read_text())
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('origin')!=p['origin']:raise ValueError('Explicit root same-observation release before archives')


def reconstruct(response,prefix_noise,probe_noise,expected_sha,*,probe_response=None,probe_label=PREFIX,family=0):
    response=np.asarray(response,float);pn=np.asarray(prefix_noise,float);qn=np.asarray(probe_noise,float)
    probe=response if probe_response is None else np.asarray(probe_response,float)
    if response.shape!=(24,)or probe.shape!=(24,)or pn.shape!=(24,)or qn.shape!=(2,3,24)or family not in(0,1)or not all(np.isfinite(v).all()for v in(response,probe,pn,qn)):raise ValueError('Only saved acquired responses and paired noises')
    obs=[(PREFIX,response+pn),(probe_label,probe+qn[family,0])]
    if fresh.oldstudy.observation_fingerprint(obs)!=expected_sha:raise ValueError('EXACT original observation binding before any inference')
    return obs


def load_observations(origin,case=CASES[0]):
    # Only acquired response/noise members; no unit/material/held arrays.
    response_records=origin['truth_responses']
    with np.load(ROOT/response_records[PREFIX]['archive']['path'],allow_pickle=False)as z:response=z['response_m'].copy()
    probe=response;label=PREFIX;family=0
    if case['phase']=='axial-r0':
        label='paired_axial_x180';family=1
        with np.load(ROOT/response_records[label]['archive']['path'],allow_pickle=False)as z:probe=z['response_m'].copy()
    with np.load(ROOT/origin['draws']['path'],allow_pickle=False)as z:prefix=z['prefix_noise'].copy();probes=z['probe_noise'].copy()
    return reconstruct(response,prefix,probes,origin['context']['observations_SHA256'],probe_response=probe,probe_label=label,family=family)


def old512_parity(current,saved):
    """Implementation reproduction only; original inference gates unchanged."""
    errors={}
    for key,oldkey,shape in(('predictive_mean','mean',(96,)),('predictive_SD','SD',(96,)),('unit_CDF','unit_CDF',(2,33))):
        x=np.asarray(current[key],float);y=np.asarray(saved[oldkey],float)
        if x.shape!=shape or y.shape!=shape or not np.isfinite(x).all()or not np.isfinite(y).all():raise ValueError('Exact old512 finite sufficient-statistic shapes')
        errors[key]=float(np.max(abs(x-y)))
    return dict(passed=errors['predictive_mean']<=2e-12 and errors['predictive_SD']<=2e-12 and errors['unit_CDF']<=1e-8,maximum_absolute_errors=errors,mean_SD_tolerance_m=2e-12,CDF_tolerance=1e-8,scope='Reproduction of frozen512 result onCPU; implementation/data-binding check, no new posterior qualification gate')


def load_old512(origin):
    path=ROOT/origin['failed_raw']['path']
    if record(path)!=origin['failed_raw']:raise ValueError('Original failed sufficient-statistic archive hash')
    with np.load(path,allow_pickle=False)as z:return {key:z[key].copy()for key in('mean','SD','unit_CDF')}


def numerical_checks(result,case=CASES[0]):
    d=result.get('diagnostics',{});plan=case_plan(case)
    schema='nonlinear-conditional-reference-v04'if case['kind']=='nonlinear'else'conditional-amplitude-v02'
    if d.get('schema')!=schema or d.get('gates')!=dict(logZ=.01,CDF=.005,quantile=.002,unit_mean=.001,target_absolute_m=2e-6,target_relative_SD=.01):raise ValueError('Original gates and explicit schema')
    requested=list(dict.fromkeys([(plan.outer[-1],n)for n in plan.inner]+[(n,plan.inner[-1])for n in plan.outer]))
    if [(r.get('outer'),r.get('inner'))for r in d.get('levels',[])]!=requested:raise ValueError('All six extended levels retained')
    for axis,orders in(('inner',plan.inner),('outer',plan.outer)):
        if [(r.get('coarse'),r.get('fine'))for r in d.get('successive',{}).get(axis,[])]!=list(zip(orders,orders[1:])):raise ValueError('Original and extended successive pairs')
    return cost.engine_gates(result)


def completed_control_gate(digest):
    """Pure metadata/hash prerequisite; never opens a posterior or truth archive."""
    p,_=validate(digest);release_gate(p,digest)
    e=json.loads((LOCAL/'execution.json').read_text());w=json.loads((LOCAL/'worker-result-LOCAL.json').read_text())
    if not e.get('terminal')or not all(cost.phase_completion(e,w).values())or e.get('protocol_SHA256')!=digest or w.get('protocol_SHA256')!=digest or e.get('worker_pid')!=w.get('pid'):raise ValueError('All3 qualified controls and clean terminal closure')
    cost.old.absent(e['controller_pid']);cost.old.absent(e['worker_pid'])
    if e['result']!=record(LOCAL/'worker-result-LOCAL.json')or e['release']!=record(LOCAL/'release.json')or w['release']!=e['release']or (ROOT/e['external_cache']['path']).read_bytes()!=(LOCAL/'worker-result-LOCAL.json').read_bytes():raise ValueError('Exact release/result/external cache')
    if set(w.get('cases',{}))!={case['id']for case in CASES}or w.get('new_RNG_calls')!=0 or w.get('new_PDE_calls')!=0 or w.get('true_parameters_or_held_truth_read')is not False:raise ValueError('All3 fixed cases/no forbidden reads or calls')
    for case in CASES:
        row=w['cases'][case['id']]
        checks=numerical_checks(dict(qualified=row['engine_qualified'],diagnostics=row['diagnostics']),case)
        if not row.get('completed')or not row.get('passed')or not row.get('old512_CPU_parity',{}).get('passed')or not checks['passed']or checks!=row['all_engine_checks']or row['observations_SHA256']!=p['origin']['cases'][case['id']]['context']['observations_SHA256']or sha(ROOT/row['posterior_moments']['path'])!=row['posterior_moments']['sha256']:raise ValueError('All original/strict/adaptive gates and exact observation/raw checks')
    expected={row['context']['model_fingerprint']for row in p['origin']['cases'].values()}
    if expected!={w['memo']['model_fingerprint']}:raise ValueError('Qualified canonical exact modal identity')
    return dict(protocol=record(LOCAL/'protocol.json'),release=record(LOCAL/'release.json'),execution=record(LOCAL/'execution.json'),result=record(LOCAL/'worker-result-LOCAL.json'),origin=p['origin'],source_SHA256=p['source_SHA256'],cases=p['cases'],memo=w['memo'],wall_s=e['wall_s'],worker_wall_s=w['wall_s'],all_checks_passed=True)


def worker(digest):
    guard('worker');p,fp=validate(digest);release_gate(p,digest)
    if(LOCAL/'worker-result-LOCAL.json').exists():raise FileExistsError('No actual retry')
    start=time.monotonic();state=dict(completed=False,passed=False,pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),new_RNG_calls=0,new_PDE_calls=0,true_parameters_or_held_truth_read=False,cases={})
    def save():state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'worker-result-LOCAL.json',state)
    def timed(*_):raise TimeoutError('Frozen2400s numerical control cap')
    def interrupted(*_):raise InterruptedError('Explicit owned supervisor interruption')
    previous={s:signal.signal(s,h)for s,h in((signal.SIGALRM,timed),(signal.SIGTERM,interrupted))};signal.alarm(2400)
    try:
        observations_by_case={case['id']:load_observations(p['origin']['cases'][case['id']],case)for case in CASES}
        atlas=fresh.load_atlas(fp)
        from experiments.campaign_surface_p2_static_v02 import SparseStaticForward,point_query_matrix
        from experiments.campaign_surface_p2_force_v03 import load_catalog
        from experiments.campaign_surface_p2_study_design_v01 import observation_points
        from experiments.campaign_surface_p2_inference_v01 import LabelledModal
        from experiments.campaign_surface_conditional_amplitude_v01 import RatioModal,ExactRatioMemo
        begin=time.monotonic();fwd=SparseStaticForward(4);linear=LabelledModal.from_forward(fwd,loads=load_catalog(4),points=observation_points(),condensed=True,query_matrix=point_query_matrix)
        fingerprint=fresh.oldstudy.model_fingerprint(linear)
        if any(fingerprint!=row['context']['model_fingerprint']for row in p['origin']['cases'].values()):raise ValueError('Canonical model fingerprint BEFORE inference')
        ratio=RatioModal(linear.modal,linear.labels);memo=ExactRatioMemo(lambda z,labels:ratio(np.exp(z),labels),model_fingerprint=fingerprint,maximum_bytes=256*1024**2,minimum_rows=1)
        state['exact_linear_construction_s']=time.monotonic()-begin;save();guard('worker')
        from experiments.campaign_surface_conditional_amplitude_v02 import ConditionalAmplitudeReferenceV02
        for case in CASES:
            begin=time.monotonic();origin=p['origin']['cases'][case['id']];observations=observations_by_case[case['id']];guard('worker')
            if case['kind']=='nonlinear':engine=NonlinearConditionalReferenceV04(memo,None,observations,p['targets'],residual_predict=atlas.predict,retain_hypotheses=False,evaluation_chunk=1024)
            else:engine=ConditionalAmplitudeReferenceV02(None,observations,p['targets'],log_ratio_predict=memo,retain_hypotheses=False)
            original512=engine.integrate(outer=512,inner=512,moment_chunk=1024)
            parity=old512_parity(original512,load_old512(origin))
            if not parity['passed']:raise ValueError('Old512 CPU sufficient-statistic reproduction failed: '+repr(parity))
            del original512
            result=engine.qualify(plan=case_plan(case));checks=numerical_checks(result,case);last=result['local_final']
            row=dict(completed=True,passed=checks['passed'],old512_CPU_parity=parity,engine_qualified=result['qualified'],all_engine_checks=checks,diagnostics=result['diagnostics'],observations_SHA256=fresh.oldstudy.observation_fingerprint(observations),seconds=time.monotonic()-begin)
            if last is not None:
                target=LOCAL/(case['id']+'-posterior-moments-LOCAL.npz')
                np.savez_compressed(target,**{k:last[k]for k in('predictive_mean','predictive_SD','unit_mean','unit_covariance','unit_CDF','unit_quantiles')});row['posterior_moments']=record(target)
            state['cases'][case['id']]=row;save();guard('worker')
        state.update(completed=True,passed=all(row['passed']for row in state['cases'].values())and len(state['cases'])==3,memo=memo.diagnostics());save()

    except BaseException as e:state.update(completed=False,passed=False,failure=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));save();raise
    finally:
        signal.alarm(0)
        for s,h in previous.items():signal.signal(s,h)
        save()
    return state


def run(digest):
    guard('controller');p,_=validate(digest);release_gate(p,digest)
    if(LOCAL/'execution.json').exists():raise FileExistsError('No actual retry')
    start=time.monotonic();child=None;state=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'))
    def interrupted(*_):raise InterruptedError('Explicit controller interruption')
    previous={s:signal.signal(s,interrupted)for s in(signal.SIGTERM,signal.SIGINT)}
    try:
        with(LOCAL/'worker.log').open('xb')as log:
            child=subprocess.Popen(['taskset','-c','17',sys.executable,'-u','-m','experiments.run_fresh64_inner_resolution_control_v01','worker','--sha',digest],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['worker_pid']=child.pid;write(LOCAL/'execution.json',state)
            while child.poll()is None:
                guard('controller')
                if cost.owned_RSS(child)>RESOURCES['worker_RSS_cap_bytes']or time.monotonic()-start>2400:raise TimeoutError('Owned worker wall/RSS cap')
                time.sleep(.1)
            state['worker_exit_code']=child.wait()
    except BaseException as e:state['failure']=repr(e)
    finally:
        if child is not None:state['cleanup_exit_code']=cost.old.cost.cleanup(child)
        for s,h in previous.items():signal.signal(s,h)
        f=LOCAL/'worker-result-LOCAL.json'
        if f.exists():
            w=json.loads(f.read_text());state.update(**cost.phase_completion(state,w),result=record(f));cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(sha(f)+'.json');cache.parent.mkdir(exist_ok=True,parents=True)
            if cache.exists()and cache.read_bytes()!=f.read_bytes():raise ValueError('External immutable cache')
            if not cache.exists():cache.write_bytes(f.read_bytes())
            state['external_cache']=record(cache)
        state.update(terminal=True,wall_s=time.monotonic()-start);write(LOCAL/'execution.json',state)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','run']);parser.add_argument('--sha');args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare()))
    elif args.action=='worker':r=worker(args.sha);print(json.dumps(dict(completed=r['completed'],qualified=r['passed'])))
    else:r=run(args.sha);print(json.dumps(r));raise SystemExit(0 if r['completed']and r.get('qualified')else 1)
