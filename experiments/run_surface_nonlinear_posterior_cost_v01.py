"""DRAFT admitted-full-atlas nonlinear posterior cost control; no automatic data.

Preparation remains closed until full10 atlas admission exists and is bound.
Worker additionally requires explicit root release. No original16 object data,
physical rerun, noise draw or inference occurs during source preparation/tests.
"""
from __future__ import annotations
import argparse,ast,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_surface_nh80_cost_v01 as cost
from experiments.run_surface_full_atlas_feasibility_v01 import owned_RSS
ROOT=cost.ROOT;LOCAL=ROOT/'runs/operator-dev/surface-nonlinear-posterior-cost-v01'
SOURCE='experiments/run_surface_nonlinear_posterior_cost_v01.py';TEST='tests/test_surface_nonlinear_posterior_cost_v01.py'
UNIT=(.5,.5);NOISE_SEED=2026102591;SIGMA=20e-6
COST_PROTOCOL_SHA='ddc96294c33ac0cb104c9da257d97f1cbe197be60924574b417044f0660cdc33'
RESOURCES=dict(worker_CPU=17,controller_CPU=16,threads=1,GPU=False,worker_RSS_cap_bytes=4*1024**3,wall_cap_s=1800,cleanup_grace_s=3,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=512*1024**2)
sha=cost.sha;record=cost.record;write=cost.write


def sources():
    pending=[SOURCE,TEST];seen=set()
    while pending:
        f=pending.pop()
        if f in seen:continue
        seen.add(f)
        for node in ast.walk(ast.parse((ROOT/f).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:modules=[node.module or '']+[(node.module+'.'+v.name)for v in node.names if node.module]
            for name in modules:
                path=name.replace('.','/')+'.py'
                if (ROOT/path).is_file()and path not in seen:pending.append(path)
    return {f:sha(ROOT/f)for f in sorted(seen)}


def absent(pid):
    if type(pid)is not int or pid<=1:raise ValueError('Concrete positive owned PID')
    for check in(os.kill,os.killpg):
        try:check(pid,0)
        except ProcessLookupError:pass
        else:raise ValueError('Prerequisite PID/group still exists')


def full_atlas_gate(summary_path,digest):
    """Metadata/hash-only barrier; never opens NPZ or draws observations."""
    from experiments import run_surface_full_atlas_feasibility_v01 as full
    path=Path(summary_path)
    if path!=full.LOCAL/'summary-LOCAL.json' or sha(path)!=digest:raise ValueError('Exact complete full10 summary record required')
    a=json.loads(path.read_text());protocol_digest=a.get('protocol_SHA256');p,_=full.validate(protocol_digest);e=json.loads((full.LOCAL/'execution.json').read_text())
    required={'completed':True,'qualified':True,'all1130_retained':True,'physical_passed':1130,'reused_prefix_states':113,'new_state_count':1017,'channels':24,'target_mesh_n':4}
    if any(a.get(k)!=v for k,v in required.items()):raise ValueError('All1130 states/full10 admission required; prefix-only forbidden')
    if a.get('labels')!=p['labels'] or len(a['labels'])!=10 or len(set(a['labels']))!=10:raise ValueError('Exact ordered ten force labels')
    if not all(a['perforce_validation'][label]['passed']for label in a['labels'])or not a['joint_validation_levels'][-1]['passed']or a['joint_validation_levels'][-1]['n']!=9 or not a['joint_validation_levels'][-1]['all72_joint_cases_retained']:raise ValueError('Every force and full48-channel joint validation required')
    if not all(e.get(k)is True for k in ['completed','qualified','terminal'])or e['summary']!=record(path)or e['protocol_SHA256']!=protocol_digest or len(e['workers'])!=16 or sorted(r['slot']for r in e['workers'])!=list(range(16)):raise ValueError('Full16-worker terminal exact summary binding')
    for r in e['workers']:
        if not r['reaped']or r['exit_code']!=0:raise ValueError('All fullatlas workers reaped0')
        absent(r['pid'])
    absent(e['controller_pid']);archive=ROOT/a['atlas']['path']
    if sha(archive)!=a['atlas']['sha256']:raise ValueError('Admitted atlas archive hash')
    return a,p,dict(summary=record(path),protocol=record(full.LOCAL/'protocol.json'),execution=record(full.LOCAL/'execution.json'),archive=record(archive))


def truth_gate():
    cost.validate(COST_PROTOCOL_SHA);e=json.loads((cost.LOCAL/'execution.json').read_text());a=json.loads((cost.LOCAL/'result.json').read_text())
    if not e['terminal']or not e['completed']or e['worker_exit_code']!=0 or e['cleanup_exit_code']!=0 or not a['completed']or not a['passed']or e['protocol_sha256']!=COST_PROTOCOL_SHA or e['result']!=record(cost.LOCAL/'result.json'):raise ValueError('Closed six-state directNH80 source required')
    absent(e['controller_pid']);absent(e['worker_pid']);r=a['cases'][1]
    if r['index']!=1 or r['E_Pa']!=4000. or r['K_N_m']!=float(np.sqrt(150.))or not r['field_parity']['passed']or not r['branches']['exact_linear_warm']['passed']:raise ValueError('Exact public synthetic center direct field')
    raw=r['branches']['exact_linear_warm']['capture']
    if sha(ROOT/raw['path'])!=raw['sha256']:raise ValueError('Direct field source hash')
    return dict(protocol=record(cost.LOCAL/'protocol.json'),result=record(cost.LOCAL/'result.json'),execution=record(cost.LOCAL/'execution.json'),capture=raw,E_Pa=4000.,K_N_m=float(np.sqrt(150.)),unit=list(UNIT),purpose='Deterministic public cost control, not an old16 object or population calibration draw')


def absolute_engine_witnesses(result):
    rows={};ok=result.get('qualified')is True
    for axis in('inner','outer'):
        history=result.get('diagnostics',{}).get('successive',{}).get(axis,[])
        row=history[-1]if history else {};errors={key:row.get('maximum_'+key+'_difference_m')for key in('predictive_mean','predictive_SD')}
        passed=bool(row.get('passed')is True and all(isinstance(v,(float,int))and np.isfinite(v)and 0<=v<=2e-6 for v in errors.values()))
        rows[axis]=dict(passed=passed,actual_absolute_errors_m=errors,tolerance_m=2e-6,coarse=row.get('coarse'),fine=row.get('fine'));ok=ok and passed
    return dict(passed=bool(ok),final_axis_witnesses=rows,engine_qualified=result.get('qualified')is True,strict_absolute_rule_is_additional=True)


def cross_atlas_checks(coarse,fine):
    from experiments.campaign_surface_inverse_reference_v01 import compare_grids
    check=compare_grids(coarse,fine);mean=float(np.max(abs(np.asarray(coarse['unit_mean'])-np.asarray(fine['unit_mean']))));absolute={key:float(np.max(abs(np.asarray(coarse[key])-np.asarray(fine[key]))))for key in('predictive_mean','predictive_SD')}
    passed=check['passed']and mean<=.001 and all(np.isfinite(v)and v<=2e-6 for v in absolute.values())
    return dict(passed=bool(passed),original_checks=check,maximum_unit_mean_difference=mean,strict_maximum_absolute_predictive_errors_m=absolute,absolute_tolerance_m=2e-6)


def guard(role):
    shared=('worker_CPU','controller_CPU','threads','GPU','worker_RSS_cap_bytes','RAM_reserve_bytes','disk_reserve_bytes')
    if any(RESOURCES[k]!=cost.LIMITS[k]for k in shared):raise ValueError('Delegated original CPU/RSS/noGPU guard configuration mismatch')
    cost.guard(role)
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise OSError('Own512MiB output cap')


def phase_completion(execution,worker):
    closed=bool('failure'not in execution and execution.get('worker_exit_code')==0 and execution.get('cleanup_exit_code')==0)
    completed=bool(closed and worker.get('completed')is True)
    return dict(completed=completed,qualified=bool(completed and worker.get('passed')is True))


def prepare(summary_path,digest):
    guard('controller');a,fp,atlas_record=full_atlas_gate(summary_path,digest);truth=truth_gate()
    p=dict(schema='nonlinear-full10-posterior-cost-v01',status='FROZEN_ROOT_RELEASE_REQUIRED',source_SHA256=sources(),resources=RESOURCES,full_atlas=atlas_record,full_protocol_SHA256=a['protocol_SHA256'],full_atlas_fingerprint=a['atlas_fingerprint'],direct_truth=truth,unit=list(UNIT),noise_seed=NOISE_SEED,sigma_m=SIGMA,prefix='normal_x180',all_target_labels=a['labels'],atlas_levels=[5,9],integration=dict(inner=[128,256,512],outer=[128,256,512],moment_chunk=1024,evaluation_chunk=1024,retain_hypotheses=False,independent_adaptive_epsrel=[1e-6,1e-8]),memo=dict(maximum_bytes=256*1024**2,minimum_rows=1,exact_bytes_only=True,all_linear_modes_retained=True),additional_gates=dict(actual_absolute_target_mean_SD_m=2e-6,within_each_engine_final_inner_and_outer=True,cross_atlas_logZ=.01,cross_atlas_CDF=.005,cross_atlas_quantile=.002,cross_atlas_unit_mean=.001),scope='One deterministic synthetic numerical cost control; same observation for5/9 full-likelihood models, no grouped nonlinear variance, no original16 read, no calibration or active policy. No extra PDE or noise before protocol+release.')
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=RESOURCES:raise ValueError('Frozen new source/protocol/resources required')
    a,_,records=full_atlas_gate(ROOT/p['full_atlas']['summary']['path'],p['full_atlas']['summary']['sha256'])
    if records!=p['full_atlas']or a['atlas_fingerprint']!=p['full_atlas_fingerprint']or truth_gate()!=p['direct_truth']or p['unit']!=list(UNIT)or p['noise_seed']!=NOISE_SEED or p['sigma_m']!=SIGMA or p['atlas_levels']!=[5,9]or p['all_target_labels']!=a['labels']:raise ValueError('Exact admitted model/truth/new independent observation recipe')
    return p,a


def release_gate(p,digest):
    r=json.loads((LOCAL/'release.json').read_text())
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('full_atlas')!=p['full_atlas']:raise ValueError('Explicit root full-atlas/protocol/source release before arrays/noise')


def completed_cost_gate(digest):
    """Future prepare prerequisite: all scalar checks/closure, no array reads."""
    p,_=validate(digest);release_gate(p,digest)
    e=json.loads((LOCAL/'execution.json').read_text());w=json.loads((LOCAL/'worker-result-LOCAL.json').read_text())
    if not e.get('terminal')or not all(phase_completion(e,w).values())or e['protocol_SHA256']!=digest or w['protocol_SHA256']!=digest or e['worker_pid']!=w['pid']:raise ValueError('Completed qualified root-released numerical cost control required')
    absent(e['controller_pid']);absent(e['worker_pid'])
    result=record(LOCAL/'worker-result-LOCAL.json');release=record(LOCAL/'release.json')
    if e['result']!=result or e['release']!=release or w['release']!=release or (ROOT/e['external_cache']['path']).read_bytes()!=(LOCAL/'worker-result-LOCAL.json').read_bytes():raise ValueError('Exact cost metadata/release/cache binding')
    if set(w['atlas_results'])!={'5','9'}or not w['cross_atlas']['passed']:raise ValueError('Both atlas5/9 and cross-model checks required')
    for row in w['atlas_results'].values():
        strict=absolute_engine_witnesses(dict(qualified=row['engine_qualified'],diagnostics=row['diagnostics']))
        if not row['completed']or not strict['passed']or strict!=row['strict_absolute_checks']:raise ValueError('Original engine and actual2um finalaxis checks required')
        if sha(ROOT/row['posterior_moments']['path'])!=row['posterior_moments']['sha256']:raise ValueError('Cost posterior moments archive hash')
    cross=w['cross_atlas'];checks=cross['original_checks']
    bounds={'log_evidence_difference':.01,'maximum_CDF_difference':.005,'maximum_unit_quantile_difference':.002}
    if not checks['passed']or any(not np.isfinite(checks[k])or not 0<=checks[k]<=v for k,v in bounds.items())or not 0<=cross['maximum_unit_mean_difference']<=.001 or any(not np.isfinite(v)or not 0<=v<=2e-6 for v in cross['strict_maximum_absolute_predictive_errors_m'].values()):raise ValueError('Explicit cross-atlas original and absolute gates')
    return dict(protocol=record(LOCAL/'protocol.json'),release=release,execution=record(LOCAL/'execution.json'),result=result,full_atlas=p['full_atlas'],source_SHA256=p['source_SHA256'],wall_s=e['wall_s'],worker_wall_s=w['wall_s'],exact_linear_construction_s=w['exact_linear_construction_s'],atlas_seconds={n:r['seconds']for n,r in w['atlas_results'].items()},memo=w['memo'],all_checks_passed=True)


def worker(digest):
    guard('worker');p,a=validate(digest);release_gate(p,digest)
    if(LOCAL/'worker-result-LOCAL.json').exists()or(LOCAL/'observation-LOCAL.npz').exists():raise FileExistsError('No actual restart/noise redraw')
    start=time.monotonic();state=dict(completed=False,passed=False,pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),atlas_results={},old16_data_read=False)
    def save():state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'worker-result-LOCAL.json',state)
    def stop(*_):raise TimeoutError('Frozen1800s numerical cost cap')
    def interrupted(*_):raise InterruptedError('Explicit supervisor SIGTERM cleanup')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,stop);signal.alarm(1800)
    try:
        from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas
        from experiments.campaign_surface_p2_static_v02 import SparseStaticForward,point_query_matrix
        from experiments.campaign_surface_p2_force_v03 import load_catalog
        from experiments.campaign_surface_p2_study_design_v01 import observation_points
        from experiments.campaign_surface_p2_inference_v01 import LabelledModal
        from experiments.campaign_surface_conditional_amplitude_v01 import RatioModal,ExactRatioMemo
        from experiments.campaign_surface_response_cache_v01 import model_fingerprint
        from experiments.campaign_nonlinear_conditional_reference_v01 import NonlinearConditionalReference
        from experiments.campaign_surface_conditional_amplitude_v02 import AmplitudePWLPlan
        # Every truth-bearing array and the single noise draw are below all gates.
        with np.load(ROOT/p['full_atlas']['archive']['path'],allow_pickle=False)as z:
            labels=tuple(str(v)for v in z['labels']);atlas=NonlinearResidualAtlas(z['delta'],labels,provenance_fingerprint=p['full_protocol_SHA256'])
            if not np.array_equal(z['axis'],atlas.axis)or atlas.delta.shape!=(9,9,10,24)or list(labels)!=p['all_target_labels']or atlas.fingerprint!=p['full_atlas_fingerprint']:raise ValueError('Exact admitted atlas shapes/axes/labels/fingerprint')
        t=time.monotonic();fwd=SparseStaticForward(4);loads=load_catalog(4);points=observation_points();Q=point_query_matrix(fwd,points);linear=LabelledModal.from_forward(fwd,loads=loads,points=points,condensed=True,query_matrix=point_query_matrix);ratio=RatioModal(linear.modal,linear.labels);memo=ExactRatioMemo(lambda z,labels:ratio(np.exp(z),labels),model_fingerprint=model_fingerprint(linear),maximum_bytes=256*1024**2,minimum_rows=1);state['exact_linear_construction_s']=time.monotonic()-t;state['linear_modes']=linear.modal.diagnostics;save()
        with np.load(ROOT/p['direct_truth']['capture']['path'],allow_pickle=False)as z:
            if not np.array_equal(z['rest_m'],fwd.X)or not np.array_equal(z['elements'],fwd.elements)or not np.array_equal(z['force_N'],loads[p['prefix']])or not np.array_equal(z['exact_constraint_nodes'],fwd.fixed):raise ValueError('Direct NH80 prefix geometry/force/clamp layout')
            u=z['displacement_m'];truth=np.asarray(Q@u.ravel()[fwd.free]);direct_linear=np.asarray(Q@z['linear_displacement_m'].ravel()[fwd.free])
        modal_center=linear.select([p['prefix']])(np.array([UNIT]))[0,0]
        if np.max(abs(modal_center-direct_linear))>2e-12:raise ValueError('All-mode exact linear/direct query parity at synthetic center')
        noise=np.random.default_rng(p['noise_seed']).normal(0,p['sigma_m'],size=24);observation=truth+noise;np.savez_compressed(LOCAL/'observation-LOCAL.npz',unit=np.array(UNIT),direct_NH80_prefix_m=truth,noise_m=noise,observation_m=observation);state['observation']=record(LOCAL/'observation-LOCAL.npz');save()
        final={}
        for n in p['atlas_levels']:
            begin=time.monotonic();selected=atlas.coarsen(n);engine=NonlinearConditionalReference(memo,None,[(p['prefix'],observation)],labels,residual_predict=selected.predict,retain_hypotheses=False,evaluation_chunk=1024);result=engine.qualify(plan=AmplitudePWLPlan(inner=(128,256,512),outer=(128,256,512),moment_chunk=1024));strict=absolute_engine_witnesses(result);row=dict(completed=True,engine_qualified=bool(result['qualified']),strict_absolute_checks=strict,diagnostics=result['diagnostics'],seconds=time.monotonic()-begin,memo=memo.diagnostics());state['atlas_results'][str(n)]=row
            if result['local_final']is not None:
                last=result['local_final'];final[n]=last
                if 'grouped'in last:raise ValueError('Nonlinear moments cannot use grouped linear amplitude variance')
                np.savez_compressed(LOCAL/f'atlas{n}-posterior-LOCAL.npz',**{k:last[k]for k in ['unit_mean','unit_covariance','unit_CDF','unit_quantiles','predictive_mean','predictive_SD']});row['posterior_moments']=record(LOCAL/f'atlas{n}-posterior-LOCAL.npz')
            save();guard('worker')
        cross=cross_atlas_checks(final[5],final[9])if set(final)=={5,9}else dict(passed=False,reason='Missing retained final intrinsic state')
        state.update(completed=True,passed=bool(all(r['strict_absolute_checks']['passed']for r in state['atlas_results'].values())and cross['passed']),cross_atlas=cross,memo=memo.diagnostics());save()
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());save();raise
    finally:signal.alarm(0);save()
    return state


def run(digest):
    guard('controller');p,_=validate(digest);release_gate(p,digest)
    if(LOCAL/'execution.json').exists():raise FileExistsError('No actual retry')
    start=time.monotonic();child=None;state=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'))
    def stopped(*_):raise InterruptedError('Controller stopped')
    old={sig:signal.signal(sig,stopped)for sig in(signal.SIGTERM,signal.SIGINT)}
    try:
        with(LOCAL/'worker.log').open('xb')as log:
            child=subprocess.Popen(['taskset','-c','17',sys.executable,'-u','-m','experiments.run_surface_nonlinear_posterior_cost_v01','worker','--sha',digest],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['worker_pid']=child.pid;write(LOCAL/'execution.json',state)
            while child.poll()is None:
                guard('controller')
                rss=owned_RSS(child)
                if rss>RESOURCES['worker_RSS_cap_bytes']or time.monotonic()-start>1800:raise TimeoutError('Independent parent1800s/4GiB cap')
                time.sleep(.1)
            state['worker_exit_code']=child.wait()
    except BaseException as error:state['failure']=repr(error)
    finally:
        if child is not None:state['cleanup_exit_code']=cost.cleanup(child)
        for sig,handler in old.items():signal.signal(sig,handler)
        f=LOCAL/'worker-result-LOCAL.json'
        if f.exists():
            a=json.loads(f.read_text());state.update(**phase_completion(state,a),result=record(f));cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(sha(f)+'.json');cache.parent.mkdir(parents=True,exist_ok=True)
            if cache.exists()and cache.read_bytes()!=f.read_bytes():raise ValueError('Exact immutable metadata cache')
            if not cache.exists():cache.write_bytes(f.read_bytes())
            state['external_cache']=record(cache)
        state.update(terminal=True,wall_s=time.monotonic()-start);write(LOCAL/'execution.json',state)
    return state

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','run']);parser.add_argument('--sha');parser.add_argument('--full-atlas-summary');parser.add_argument('--full-atlas-sha');args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare(Path(args.full_atlas_summary),args.full_atlas_sha)))
    elif args.action=='worker':a=worker(args.sha);print(json.dumps(dict(completed=a['completed'],qualified=a['passed'])))
    else:a=run(args.sha);print(json.dumps(a));raise SystemExit(0 if a['completed']and a['qualified']else 1)
