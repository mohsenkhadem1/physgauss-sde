"""New bounded prior-floor MIS-v03 cost qualification using the EXACT saved v01 observation.

No RNG/PDE or cohort access. Preparation is metadata only; root release is
required before any archive is opened. Frozen v01 failures remain prerequisites.
"""
from __future__ import annotations
import argparse,ast,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_surface_nonlinear_posterior_cost_v01 as old
from experiments import run_surface_nonlinear_posterior_cost_v02 as previous
ROOT=old.ROOT;LOCAL=ROOT/'runs/operator-dev/surface-nonlinear-posterior-cost-v03'
SOURCE='experiments/run_surface_nonlinear_posterior_cost_v03.py';TEST='tests/test_surface_nonlinear_posterior_cost_v03.py'
BACKENDS=('experiments/campaign_nonlinear_conditional_reference_v03.py','experiments/campaign_nonlinear_mis_pwl_v01.py','experiments/campaign_nonlinear_residual_atlas_fast_v01.py')
ORIGIN_SHA='8e48a46387dae1936e47650ccabd9b74f40f99a66ddc4481607bb7154f1091ee'
DIAGNOSTIC=ROOT/'runs/operator-dev/surface-tilted-cdf-diagnostic-v01/summary.json'
DIAGNOSTIC_SHA='cd0c6a2f8b098963164bfef0d2697b983aa38e0f974e1a89b4d7c013b84e5a51'
RESOURCES=dict(old.RESOURCES)
RESOURCE_PROJECTION=dict(final_outer_rows=1536,atoms_per_row=2048,maximum_weighted_hypotheses=3145728,relative_to_v02_hypothesis_count=2,target_chunk=1024,target_labels=10,channels=24,target_chunk_output_bytes=1966080,full_hypothesis_by_target_array_allocated=False,retain_hypotheses=False,actual_per_process_4GiB_guard_authoritative=True)
sha,record,write=old.sha,old.record,old.write
owned_RSS=old.owned_RSS
absolute_engine_witnesses=old.absolute_engine_witnesses
cross_atlas_checks=old.cross_atlas_checks
phase_completion=old.phase_completion


def sources():
    pending=[SOURCE,TEST,*BACKENDS];seen=set()
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


V02_PROTOCOL_SHA='ee27581ec076dbc611e25e8ea3506423ebf1717b812af9087a85008198038b8e'
V02_REVIEW_SHA='602c65bc061171cea5e725b43d776196eee8089dbdbb33ed1488066ddcfadb3f'
V02_SUMMARY_SHA='5c1b5506cf5f52fc17011766148dffdbfec92d239c5e0a90f917a52509839b7d'


def failed_v02_gate():
    """Bind the second clean numerical failure, never overwrite/read its arrays."""
    p,_=previous.validate(V02_PROTOCOL_SHA);previous.release_gate(p,V02_PROTOCOL_SHA)
    e=json.loads((previous.LOCAL/'execution.json').read_text());w=json.loads((previous.LOCAL/'worker-result-LOCAL.json').read_text())
    if not e.get('terminal')or not e.get('completed')or e.get('qualified')is not False or 'failure'in e or e.get('worker_exit_code')!=0 or e.get('cleanup_exit_code')!=0 or not w.get('completed')or w.get('passed')is not False or 'failure'in w:raise ValueError('Closed numerical-negative v02 required')
    old.absent(e['controller_pid']);old.absent(e['worker_pid'])
    if e['protocol_SHA256']!=V02_PROTOCOL_SHA or w['protocol_SHA256']!=V02_PROTOCOL_SHA or e['result']!=record(previous.LOCAL/'worker-result-LOCAL.json')or e['release']!=record(previous.LOCAL/'release.json')or w['release']!=e['release']or w['observation']!=p['origin_v01']['observation']or (ROOT/e['external_cache']['path']).read_bytes()!=(previous.LOCAL/'worker-result-LOCAL.json').read_bytes():raise ValueError('Exact v02 source/result/release/cache/same-observation binding')
    review=previous.LOCAL/'independent-negative-review.json';summary=previous.LOCAL/'summary.json'
    if sha(review)!=V02_REVIEW_SHA or sha(summary)!=V02_SUMMARY_SHA:raise ValueError('Immutable v02 negative review/summary pins')
    return dict(protocol=record(previous.LOCAL/'protocol.json'),source_SHA256=p['source_SHA256'],execution=record(previous.LOCAL/'execution.json'),result=e['result'],release=e['release'],external_cache=e['external_cache'],independent_review=record(review),summary=record(summary))


def origin_gate():
    p,a,origin=previous.origin_gate()
    return p,a,dict(origin,negative_cost_v02=failed_v02_gate())


def guard(role):
    if RESOURCES!=old.RESOURCES:raise ValueError('Original separate4GiB/core/thread/noGPU resource configuration')
    old.guard(role)
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise OSError('Own512MiB output cap')


def engine_gates(result):
    """Require original final-axis/adaptive checks AND strict actual2um errors."""
    strict=absolute_engine_witnesses(result);d=result.get('diagnostics',{});a=d.get('independent_adaptive',{});ok=bool(result.get('qualified')is True and strict['passed']and a.get('passed')is True and a.get('epsrel')==[1e-6,1e-8]and a.get('CDF_checks_per_epsrel')==6)
    bounds={'log_evidence_difference':.01,'maximum_CDF_error':.005,'adaptive_refinement_log_evidence_difference':.01,'adaptive_refinement_maximum_CDF_difference':.005}
    for k,b in bounds.items():
        v=a.get(k);ok=ok and isinstance(v,(float,int))and np.isfinite(v)and 0<=v<=b
    errors=a.get('maximum_QUADPACK_relative_error_estimates',{})
    ok=ok and set(errors)=={'1e-06','1e-08'}and all(isinstance(v,(float,int))and np.isfinite(v)and 0<=v<=1e-5 for v in errors.values())
    for axis in('inner','outer'):
        rows=d.get('successive',{}).get(axis,[]);r=rows[-1]if rows else {}
        for k,b in {'log_evidence_difference':.01,'maximum_CDF_difference':.005,'maximum_unit_quantile_difference':.002,'maximum_unit_mean_difference':.001}.items():
            v=r.get(k);ok=ok and r.get('passed')is True and isinstance(v,(float,int))and np.isfinite(v)and 0<=v<=b
    return dict(passed=bool(ok),strict_absolute_checks=strict,original_adaptive_checks=a,requires_all_original_gates=True)


def prepare():
    guard('controller');p,a,origin=origin_gate()
    recipe=dict(schema='nonlinear-full10-posterior-cost-v03',status='FROZEN_ROOT_RELEASE_REQUIRED',source_SHA256=sources(),resources=RESOURCES,resource_projection=RESOURCE_PROJECTION,origin_v01=origin,full_protocol_SHA256=p['full_protocol_SHA256'],full_atlas_fingerprint=p['full_atlas_fingerprint'],prefix=p['prefix'],all_target_labels=a['labels'],sigma_m=p['sigma_m'],saved_observation_only=True,new_RNG_or_PDE=False,atlas_levels=[5,9],integration=p['integration'],memo=p['memo'],additional_gates=p['additional_gates'],backend=dict(module='experiments.campaign_nonlinear_conditional_reference_v03',class_name='NonlinearConditionalReferenceV03',only_change='Fixed half Gaussian plus half normalized log-a MIS proposal; qMix balance weights preserve original likelihood/prior, no Gaussian kernel-mass multiplier, linear_peak added once. Original qualifier/oracle gates unchanged.'),atlas_evaluator=dict(class_name='FastResidualAtlas',table_cache_bytes=8*1024**2,same_original_barycentric_polynomial=True,no_parameter_prediction_cache=True),scope='One SAME-observation numerical repair qualification; all v01/v02/diagnostic failures retained, no population or old16 access. Worker4GiB and controller4GiB are separate per-process bounds, not a combined4GiB claim.')
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    write(LOCAL/'protocol.json',recipe);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text());op,a,origin=origin_gate()
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=RESOURCES or p['resource_projection']!=RESOURCE_PROJECTION or p['origin_v01']!=origin:raise ValueError('Exact frozen v03 source/protocol/resources/origin')
    for k in('full_protocol_SHA256','full_atlas_fingerprint','prefix','all_target_labels','sigma_m','integration','memo','additional_gates'):
        if p[k]!=op[k]:raise ValueError('Unchanged original numerical/physical recipe '+k)
    if p['atlas_levels']!=[5,9]or not p['saved_observation_only']or p['new_RNG_or_PDE']:raise ValueError('Same observation/no draws/PDE and5+9 qualifiers')
    return p,a


def release_gate(p,digest):
    r=json.loads((LOCAL/'release.json').read_text())
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('origin_v01')!=p['origin_v01']:raise ValueError('Explicit root v03 source/protocol/saved-observation release required')


def completed_cost_gate(digest):
    """Future cohort prerequisite. Metadata/hash only; no array or RNG access."""
    p,_=validate(digest);release_gate(p,digest)
    e=json.loads((LOCAL/'execution.json').read_text());w=json.loads((LOCAL/'worker-result-LOCAL.json').read_text())
    if not e.get('terminal')or not all(phase_completion(e,w).values())or e.get('protocol_SHA256')!=digest or w.get('protocol_SHA256')!=digest or e.get('worker_pid')!=w.get('pid'):raise ValueError('Completed qualified v03 cost control and clean closure required')
    old.absent(e['controller_pid']);old.absent(e['worker_pid'])
    result=record(LOCAL/'worker-result-LOCAL.json');release=record(LOCAL/'release.json')
    if e['result']!=result or e['release']!=release or w['release']!=release or w['observation']!=p['origin_v01']['observation']or (ROOT/e['external_cache']['path']).read_bytes()!=(LOCAL/'worker-result-LOCAL.json').read_bytes():raise ValueError('Exact v03 metadata/cache/release/observation binding')
    if set(w['atlas_results'])!={'5','9'}or not w['cross_atlas']['passed']:raise ValueError('Both5/9 qualifiers and cross-model checks required')
    for row in w['atlas_results'].values():
        checks=engine_gates(dict(qualified=row['engine_qualified'],diagnostics=row['diagnostics']))
        if not row['completed']or not checks['passed']or checks!=row['all_engine_checks']:raise ValueError('All original/adaptive/strict2um engine gates')
        if sha(ROOT/row['posterior_moments']['path'])!=row['posterior_moments']['sha256']:raise ValueError('Saved posterior moment archive hash')
    c=w['cross_atlas'];bounds={'log_evidence_difference':.01,'maximum_CDF_difference':.005,'maximum_unit_quantile_difference':.002}
    if not c['original_checks']['passed']or any(not np.isfinite(c['original_checks'][k])or not 0<=c['original_checks'][k]<=v for k,v in bounds.items())or not 0<=c['maximum_unit_mean_difference']<=.001 or any(not np.isfinite(v)or not 0<=v<=2e-6 for v in c['strict_maximum_absolute_predictive_errors_m'].values()):raise ValueError('Cross-atlas original and absolute2um gates')
    return dict(protocol=record(LOCAL/'protocol.json'),release=release,execution=record(LOCAL/'execution.json'),result=result,origin_v01=p['origin_v01'],source_SHA256=p['source_SHA256'],wall_s=e['wall_s'],worker_wall_s=w['wall_s'],exact_linear_construction_s=w['exact_linear_construction_s'],atlas_seconds={n:r['seconds']for n,r in w['atlas_results'].items()},memo=w['memo'],all_checks_passed=True)


def worker(digest):
    guard('worker');p,a=validate(digest);release_gate(p,digest)
    if(LOCAL/'worker-result-LOCAL.json').exists():raise FileExistsError('No actual restart/retry')
    start=time.monotonic();state=dict(completed=False,passed=False,pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),observation=p['origin_v01']['observation'],atlas_results={},old16_data_read=False,new_noise_draws=0,new_PDE_calls=0)
    def save():state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'worker-result-LOCAL.json',state)
    def stop(*_):raise TimeoutError('Frozen1800s v03 numerical cost cap')
    def interrupted(*_):raise InterruptedError('Explicit supervisor SIGTERM cleanup')
    previous={sig:signal.signal(sig,handler)for sig,handler in((signal.SIGTERM,interrupted),(signal.SIGALRM,stop))};signal.alarm(1800)
    try:
        from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas
        from experiments.campaign_surface_p2_static_v02 import SparseStaticForward,point_query_matrix
        from experiments.campaign_surface_p2_force_v03 import load_catalog
        from experiments.campaign_surface_p2_study_design_v01 import observation_points
        from experiments.campaign_surface_p2_inference_v01 import LabelledModal
        from experiments.campaign_surface_conditional_amplitude_v01 import RatioModal,ExactRatioMemo
        from experiments.campaign_surface_response_cache_v01 import model_fingerprint
        from experiments.campaign_nonlinear_conditional_reference_v03 import NonlinearConditionalReferenceV03
        from experiments.campaign_nonlinear_residual_atlas_fast_v01 import FastResidualAtlas
        from experiments.campaign_surface_conditional_amplitude_v02 import AmplitudePWLPlan
        # Root gate precedes every archive; no truth field is needed or opened.
        with np.load(ROOT/p['origin_v01']['full_atlas']['archive']['path'],allow_pickle=False)as z:
            labels=tuple(str(v)for v in z['labels']);atlas=NonlinearResidualAtlas(z['delta'],labels,provenance_fingerprint=p['full_protocol_SHA256'])
            if not np.array_equal(z['axis'],atlas.axis)or atlas.delta.shape!=(9,9,10,24)or list(labels)!=p['all_target_labels']or atlas.fingerprint!=p['full_atlas_fingerprint']:raise ValueError('Exact admitted full10 atlas')
        atlas=FastResidualAtlas(atlas,table_cache_bytes=p['atlas_evaluator']['table_cache_bytes'])
        with np.load(ROOT/p['origin_v01']['observation']['path'],allow_pickle=False)as z:observation=z['observation_m'].copy()
        if observation.shape!=(24,)or not np.isfinite(observation).all()or record(ROOT/p['origin_v01']['observation']['path'])!=state['observation']:raise ValueError('Exact saved finite24-channel observation')
        t=time.monotonic();fwd=SparseStaticForward(4);linear=LabelledModal.from_forward(fwd,loads=load_catalog(4),points=observation_points(),condensed=True,query_matrix=point_query_matrix);ratio=RatioModal(linear.modal,linear.labels);memo=ExactRatioMemo(lambda z,labels:ratio(np.exp(z),labels),model_fingerprint=model_fingerprint(linear),maximum_bytes=256*1024**2,minimum_rows=1);state['exact_linear_construction_s']=time.monotonic()-t;state['linear_modes']=linear.modal.diagnostics;save()
        final={}
        for n in p['atlas_levels']:
            begin=time.monotonic();selected=atlas.coarsen(n);engine=NonlinearConditionalReferenceV03(memo,None,[(p['prefix'],observation)],labels,residual_predict=selected.predict,retain_hypotheses=False,evaluation_chunk=1024);result=engine.qualify(plan=AmplitudePWLPlan(inner=(128,256,512),outer=(128,256,512),moment_chunk=1024));checks=engine_gates(result);row=dict(completed=True,engine_qualified=bool(result['qualified']),all_engine_checks=checks,diagnostics=result['diagnostics'],seconds=time.monotonic()-begin,memo=memo.diagnostics());state['atlas_results'][str(n)]=row
            if result['local_final']is not None:
                last=result['local_final'];final[n]=last
                if 'grouped'in last:raise ValueError('No grouped linear variance for nonlinear predictions')
                np.savez_compressed(LOCAL/f'atlas{n}-posterior-LOCAL.npz',**{k:last[k]for k in ['unit_mean','unit_covariance','unit_CDF','unit_quantiles','predictive_mean','predictive_SD']});row['posterior_moments']=record(LOCAL/f'atlas{n}-posterior-LOCAL.npz')
            save();guard('worker')
        cross=cross_atlas_checks(final[5],final[9])if set(final)=={5,9}else dict(passed=False,reason='Missing retained final intrinsic state')
        state.update(completed=True,passed=bool(all(r['all_engine_checks']['passed']for r in state['atlas_results'].values())and cross['passed']),cross_atlas=cross,memo=memo.diagnostics());save()
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());save();raise
    finally:
        signal.alarm(0)
        for sig,handler in previous.items():signal.signal(sig,handler)
        save()
    return state


def run(digest):
    guard('controller');p,_=validate(digest);release_gate(p,digest)
    if(LOCAL/'execution.json').exists():raise FileExistsError('No actual retry')
    start=time.monotonic();child=None;state=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'))
    def stopped(*_):raise InterruptedError('Controller explicit interruption')
    previous={sig:signal.signal(sig,stopped)for sig in(signal.SIGTERM,signal.SIGINT)}
    try:
        with(LOCAL/'worker.log').open('xb')as log:
            child=subprocess.Popen(['taskset','-c','17',sys.executable,'-u','-m','experiments.run_surface_nonlinear_posterior_cost_v03','worker','--sha',digest],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['worker_pid']=child.pid;write(LOCAL/'execution.json',state)
            while child.poll()is None:
                guard('controller');rss=owned_RSS(child)
                if rss>RESOURCES['worker_RSS_cap_bytes']or time.monotonic()-start>1800:raise TimeoutError('Independent parent1800s/worker4GiB cap')
                time.sleep(.1)
            state['worker_exit_code']=child.wait()
    except BaseException as error:state['failure']=repr(error)
    finally:
        if child is not None:state['cleanup_exit_code']=old.cost.cleanup(child)
        for sig,handler in previous.items():signal.signal(sig,handler)
        f=LOCAL/'worker-result-LOCAL.json'
        if f.exists():
            w=json.loads(f.read_text());state.update(**phase_completion(state,w),result=record(f));cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(sha(f)+'.json');cache.parent.mkdir(parents=True,exist_ok=True)
            if cache.exists()and cache.read_bytes()!=f.read_bytes():raise ValueError('Exact immutable completed metadata cache')
            if not cache.exists():cache.write_bytes(f.read_bytes())
            state['external_cache']=record(cache)
        state.update(terminal=True,wall_s=time.monotonic()-start);write(LOCAL/'execution.json',state)
    return state

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','run']);parser.add_argument('--sha');args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare()))
    elif args.action=='worker':r=worker(args.sha);print(json.dumps(dict(completed=r['completed'],qualified=r['passed'])))
    else:r=run(args.sha);print(json.dumps(r));raise SystemExit(0 if r['completed']and r.get('qualified')else 1)
