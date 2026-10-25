"""Same saved prefix; qualified continuous category references then ONEaction cost.

This source-only prospective ladder cannot admit all8 action ranks or a policy.
No redraw, generating-case response, held truth, new FE solve or source edit.
"""
import argparse,ast,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments import run_layout_mixture_cost_pilot_v01 as old
from experiments import campaign_grouped_linear_decision_v01 as grouped
from experiments import audit_layout_mixture_cost_pilot_v01 as negative_audit
from experiments.campaign_surface_conditional_amplitude_v01 import RatioModal
from experiments.campaign_surface_conditional_amplitude_v02 import ConditionalAmplitudeReferenceV02,AmplitudePWLPlan
ROOT=old.ROOT;LOCAL=ROOT/'runs/operator-dev/layout-grouped-reference-cost-v01'
SOURCE='experiments/run_layout_grouped_reference_cost_v01.py';TEST='tests/test_layout_grouped_reference_cost_v01.py'
OLD_SHA='f4527a99103ee1230f069aa1e857dde0b404d9d37ed15981ccdf94e74f2e6e6e'
RESOURCES=dict(CPU=17,threads=1,GPU=False,RSS_bytes=4*1024**3,external_wall_s=600,internal_wall_s=550,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=128*1024**2,row_chunk=64,outcome_chunk=8,maximum_outcome_nodes=200000)
DECISION_LEVELS=((512,8),(512,16),(512,32),(512,64),(256,64));ACTION=0
sha,record,read,write=old.sha,old.record,old.read,old.write

def sources():
    pending=[SOURCE,TEST,negative_audit.SOURCE,negative_audit.TEST,'experiments/campaign_grouped_linear_decision_v01.py','tests/test_grouped_linear_decision_v01.py',*old.sources()];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            mods=[]
            if isinstance(node,ast.Import):mods=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:mods=[node.module or '']+[(node.module+'.'+a.name)for a in node.names if node.module]
            for mod in mods:
                name2=mod.replace('.','/')+'.py'
                if(ROOT/name2).is_file()and name2 not in seen:pending.append(name2)
    return {name:sha(ROOT/name)for name in sorted(seen)}

def origin_gate():
    old.validate(OLD_SHA);w=read(old.LOCAL/'execution.json');e=read(old.LOCAL/'root-execution.json');a=read(old.LOCAL/'independent-negative-review.json')
    if not(w.get('completed')is True and w.get('qualified')is False and a.get('integrity_passed')is True and a.get('policy_use_blocked')is True and e.get('completed')is True and e.get('exit_code')==0 and e.get('reaped')is True and e.get('cleanup_exit_code')==0):raise ValueError('Closed original finitebank negative retained')
    if a.get('source_SHA256')!={name:sha(ROOT/name)for name in(negative_audit.SOURCE,negative_audit.TEST)}:raise ValueError('Exact independent negative-audit source procedure')
    for pid in(e['controller_pid'],e['child_pid']):old.mechanical_screen.shared.old.absent(pid)
    if record(ROOT/w['prefix_observation']['path'])!=w['prefix_observation']:raise ValueError('Same saved prefix observation bytes')
    return dict(protocol=record(old.LOCAL/'protocol.json'),release=record(old.LOCAL/'release.json'),execution=record(old.LOCAL/'execution.json'),external=record(old.LOCAL/'root-execution.json'),independent_review=record(old.LOCAL/'independent-negative-review.json'),prefix_observation=w['prefix_observation'],source_SHA256=old.sources())

def prepare():
    origin=origin_gate()
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No implicit refreeze/retry')
    p=dict(schema='layout-grouped-reference-cost-v01',source_SHA256=sources(),origin=origin,resources=RESOURCES,prefix_reference_plan=dict(inner=[128,256,512],outer=[128,256,512],moment_chunk=1024),decision_levels=[list(x)for x in DECISION_LEVELS],utility_worklist=[dict(outer=o,amplitude=n,action=ACTION)for o,n in((512,32),(256,64),(512,64))],outcome_orders=[8,16],all8_action_ranks='PENDING_NOT_ADMITTED',no_RNG=True,no_generating_case_or_held_truth_reads=True,no_nonlinear_PDE=True,scope='ONEaction same-data LINEAR development/reference and resource proof only; physical/nonlinear/calibration/active-advantage admission separate')
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')

def validate(digest):
    p=read(LOCAL/'protocol.json')
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=RESOURCES or p['decision_levels']!=[list(x)for x in DECISION_LEVELS]or p['utility_worklist']!=[dict(outer=o,amplitude=n,action=ACTION)for o,n in((512,32),(256,64),(512,64))]or p['outcome_orders']!=[8,16]or p['prefix_reference_plan']!=dict(inner=[128,256,512],outer=[128,256,512],moment_chunk=1024):raise ValueError('Exact frozen grouped pilot sources/config')
    old.verify_mechanics_receipt(p['origin'])
    e=read(ROOT/p['origin']['external']['path'])
    for pid in(e['controller_pid'],e['child_pid']):old.mechanical_screen.shared.old.absent(pid)
    r=read(LOCAL/'release.json')
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('origin')!=p['origin']or r.get('external_watchdog_600s_4GiB')is not True:raise ValueError('Exact rootrelease/external600s4GiB bound')
    return p

def guard():
    old.guard()
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise MemoryError('Own128MiB writes')

def reference_check(result):
    original=old.mechanical_screen.shared.engine_gates(result);strict=old.mechanical_screen.shared.absolute_engine_witnesses(result)
    return dict(passed=bool(original['passed']and strict['passed']),original=original,strict_absolute_checks=strict)

def risk_check(c,f):
    error=abs(c['expected_target_variance']-f['expected_target_variance'])+f['omitted_risk_bound'];return dict(passed=bool(error<=1e-4*f['prior_target_variance']),error=error,gate_relative_to_prior=1e-4)

def worker(digest):
    p=validate(digest);guard()
    if(LOCAL/'execution.json').exists():raise FileExistsError('No grouped pilot retry')
    state=dict(completed=False,qualified=False,pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),prefix_categories=[dict(layout=s['id'],status='pending')for s in old.mechanics.LAYOUTS],decision_levels=[],all_utility_worklist=p['utility_worklist'],utility=[],all8_action_ranks='PENDING_NOT_ADMITTED',new_RNG_calls=0,nonlinear_PDE_solves=0,generating_or_held_truth_reads=0);start=time.monotonic()
    def checkpoint():state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'execution.json',state)
    def timeout(*_):raise TimeoutError('550s internal groupedcost cap; preserve pending worklist')
    def stop(*_):raise InterruptedError('External owned groupedpilot stop')
    previous={s:signal.signal(s,h)for s,h in((signal.SIGALRM,timeout),(signal.SIGTERM,stop),(signal.SIGINT,stop))};signal.alarm(550)
    try:
        with np.load(ROOT/p['origin']['prefix_observation']['path'],allow_pickle=False)as z:observation=z['observed_prefix'].copy()
        if observation.shape!=(4,)or not np.isfinite(observation).all():raise ValueError('Only same saved prefix4 data')
        loads,_=old.mechanics.load_catalog(2);refs=[];finals=[];names=old.labels()
        for k,layout in enumerate(old.mechanics.LAYOUTS):
            f=old.mechanics.make_forward(2,layout['id']);model=old.mechanics.make_measurement_modal(f,loads);ratio=RatioModal(model.modal,model.labels);ref=ConditionalAmplitudeReferenceV02(ratio,list(zip(names['prefix'],observation[:,None])),names['target'],retain_hypotheses=False);result=ref.qualify(plan=AmplitudePWLPlan(moment_chunk=1024));check=reference_check(result);path=LOCAL/f'prefix-layout-{k}-diagnostics-LOCAL.json';write(path,dict(qualified=result['qualified'],diagnostics=result['diagnostics'],strict=check));state['prefix_categories'][k]=dict(layout=layout['id'],status='qualified'if check['passed']else'unresolved',record=record(path));refs.append(ref);finals.append(result['local_final']);guard();checkpoint()
        if not all(x['status']=='qualified'for x in state['prefix_categories']):raise ValueError('Every category must independently qualify; no category dropping')
        for outer,n in DECISION_LEVELS:
            bank,metadata=grouped.decision_bank(refs,names['candidate'],names['target'],outer=outer,amplitude_order=n);check=grouped.reference_moment_checks(metadata,finals);path=LOCAL/f'decision-{outer}-{n}-LOCAL.json';write(path,dict(reference_checks=check,metadata={k:v for k,v in metadata.items()if k!='category_moments'}));state['decision_levels'].append(dict(outer=outer,amplitude=n,checks=check,record=record(path)));guard();checkpoint()
        if not all(x['checks']['passed']for x in state['decision_levels']if(x['outer'],x['amplitude'])in((512,32),(256,64),(512,64))):raise ValueError('Final decision quadrature must match each full reference evidence/48moments')
        for job in p['utility_worklist']:
            bank,metadata=grouped.decision_bank(refs,names['candidate'],names['target'],outer=job['outer'],amplitude_order=job['amplitude']);levels=[grouped.expected_risk(bank,ACTION,old.mechanics.SIGMA_M,order=order,row_chunk=64,outcome_chunk=8)for order in(8,16)];check=risk_check(*levels);path=LOCAL/f"utility-{job['outer']}-{job['amplitude']}-LOCAL.json";write(path,dict(job=job,levels=levels,outcome_check=check));state['utility'].append(dict(job=job,levels=levels,check=check,record=record(path)));guard();checkpoint()
        risk={ (v['job']['outer'],v['job']['amplitude']):v['levels'][-1]for v in state['utility']};fine=risk[512,64];comparisons={}
        for axis,key in(('amplitude',(512,32)),('outer',(256,64))):
            error=abs(risk[key]['expected_target_variance']-fine['expected_target_variance']);comparisons[axis]=dict(error=error,relative_to_prior=error/fine['prior_target_variance'],passed=bool(error<=1e-3*fine['prior_target_variance']))
        state.update(completed=True,decision_risk_refinements=comparisons,qualified=all(v['check']['passed']for v in state['utility'])and all(v['passed']for v in comparisons.values()),one_action_only=True,not_all8_policy_admission=True)
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        signal.alarm(0)
        for s,h in previous.items():signal.signal(s,h)
        checkpoint()
    return state

def run(digest):
    """External owned-child600s/aggregate4GiB watchdog; no physics in controller."""
    validate(digest)
    if set(os.sched_getaffinity(0))!={18}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('External controllerCPU18/one-thread/noGPU')
    if(LOCAL/'root-execution.json').exists():raise FileExistsError('No external controller retry')
    state=dict(completed=False,qualified=False,controller_pid=os.getpid(),protocol_SHA256=digest,wall_cap_s=600,RSS_cap_bytes=4*1024**3);child=None;start=time.monotonic()
    def stop(*_):raise InterruptedError('External groupedcost controller stop')
    previous={s:signal.signal(s,stop)for s in(signal.SIGTERM,signal.SIGINT)}
    try:
        with(LOCAL/'root-worker.log').open('xb')as log:
            command=['taskset','-c','17',sys.executable,'-u','-m','experiments.run_layout_grouped_reference_cost_v01','worker','--sha',digest];child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state.update(child_pid=child.pid,command=command);write(LOCAL/'root-execution.json',state)
            while child.poll()is None:
                total=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+old.mechanical_screen.shared.owned_RSS(child);state['peak_RSS_bytes']=max(state.get('peak_RSS_bytes',0),total)
                if total>=4*1024**3:raise MemoryError('External aggregate4GiB cap')
                if time.monotonic()-start>600:raise TimeoutError('External600s cap')
                time.sleep(.1)
            state.update(exit_code=child.wait(),reaped=True)
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error))
    finally:
        if child is not None:state.update(cleanup_exit_code=old.mechanical_screen.shared.old.cost.cleanup(child),reaped=True)
        for sig,handler in previous.items():signal.signal(sig,handler)
        path=LOCAL/'execution.json'
        if path.exists():
            w=read(path);state.update(worker=record(path),completed=bool('failure'not in state and state.get('exit_code')==0 and state.get('cleanup_exit_code')==0 and w.get('completed')is True),qualified=bool('failure'not in state and state.get('exit_code')==0 and state.get('cleanup_exit_code')==0 and w.get('completed')is True and w.get('qualified')is True))
        state['wall_s']=time.monotonic()-start;write(LOCAL/'root-execution.json',state)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','worker-guard','run']);parser.add_argument('--sha');a=parser.parse_args()
    if a.action=='worker-guard':guard();print(json.dumps(dict(passed=True,no_model_or_data=True)))
    else:
        result=prepare()if a.action=='prepare'else(run(a.sha)if a.action=='run'else worker(a.sha));print(json.dumps(result))
        if a.action=='run'and not result['completed']:raise SystemExit(1)
