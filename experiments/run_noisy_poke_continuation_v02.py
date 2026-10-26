"""Versioned resource-only continuation: no new noise, choices or physics.

Only previously unexecuted trials1..31 are evaluated. Frozen v01 source and
trial0 remain unchanged. Every new producer explicitly binds this protocol.
"""
import argparse, gc, json, os, resource, shutil, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
from experiments import run_noisy_poke_benchmark_v01 as parent
ROOT=parent.ROOT
LOCAL=ROOT/'runs/operator-dev/noisy-poke-continuation-v02'
SOURCE='experiments/run_noisy_poke_continuation_v02.py'
TEST='tests/test_noisy_poke_continuation_v02.py'
AUDIT_SOURCE='experiments/audit_noisy_poke_continuation_v02.py'
AUDIT_TEST='tests/test_audit_noisy_poke_continuation_v02.py'
RESOURCES={**parent.RESOURCES,'evaluation_wall_s':18000,'global_wall_s':24000}
sha,read,write,record=parent.sha,parent.read,parent.write,parent.record
base,design=parent.base,parent.design
infer,load_common,query_truth=parent.infer,parent.load_common,parent.query_truth
snapshot_metrics,observations_fingerprint=parent.snapshot_metrics,parent.observations_fingerprint
all_seals_gate=parent.all_seals_gate

def sources():
    return {**parent.sources(),**{name:sha(ROOT/name) for name in (SOURCE,TEST,AUDIT_SOURCE,AUDIT_TEST)}}

def exact(rec):
    if record(ROOT/rec['path'])!=rec:raise ValueError('Changed exact continuation commitment')
    return ROOT/rec['path']

def checkpoint_cost(digest):
    phase=parent.completed_phase('checkpoint',digest)
    worker=read(exact(phase['workers'][0]['result']))
    if not(worker.get('completed')is True and [v['index']for v in worker['trials']]==[0]):raise ValueError('Exact fixed original trial0')
    trial=read(exact(worker['trials'][0]['record']))
    if not(trial.get('qualified')is True and len(trial['posteriors'])==8 and [v['action']for v in trial['posteriors']]==list(range(8))and all(v.get('qualified')is True for v in trial['posteriors'])):raise ValueError('All8 checkpoint densities must qualify')
    walls=[read(exact(v['record']))['wall_s']for v in trial['posteriors']]
    if not all(np.isfinite(v)and v>=0 for v in walls):raise ValueError('Finite nonnegative measured costs')
    projection=3*max(walls)*8*11
    if projection>RESOURCES['evaluation_wall_s']:raise ValueError('Unchanged safety3 cost projection exceeds revised budget')
    return dict(slowest_posterior_wall_s=max(walls),safety_factor=3,maximum_trials_per_worker=11,actions=8,projected_remaining_wall_s=projection,original_budget_s=parent.RESOURCES['evaluation_wall_s'],revised_budget_s=RESOURCES['evaluation_wall_s'])

def untouched_original_remaining():
    if (parent.LOCAL/'execution-remaining.json').exists():raise FileExistsError('Original remaining phase was already attempted')
    for index in range(1,32):
        directory=parent.LOCAL/f'trial-{index:02d}'/'evaluation'
        if directory.exists()and any(directory.iterdir()):raise FileExistsError('Original remaining trial already has evaluation artifacts')

def origin(digest):
    untouched_original_remaining()
    p=parent.validate(digest);seals=parent.all_seals_gate(digest);parent.inputs_gate(digest)
    parent.completed_phase('choices',digest)
    cost=checkpoint_cost(digest)
    if len(seals)!=32:raise ValueError('Same all32 presealed choices')
    return dict(parent_protocol=record(parent.LOCAL/'protocol.json'),parent_release=record(parent.LOCAL/'release.json'),inputs=record(parent.LOCAL/'inputs.json'),choice_execution=record(parent.LOCAL/'execution-choices.json'),choice_closure=record(parent.LOCAL/'root-closure-choices.json'),checkpoint_execution=record(parent.LOCAL/'execution-checkpoint.json'),checkpoint_closure=record(parent.LOCAL/'root-closure-checkpoint.json'),all32_choice_seals=seals,cost=cost)

def prepare(digest):
    guard('controller');o=origin(digest)
    if (LOCAL/'protocol.json').exists():raise FileExistsError('No continuation refreeze')
    value=dict(schema='noisy-poke-resource-continuation-v02',parent_protocol_SHA256=digest,source_SHA256=sources(),origin=o,resources=RESOURCES,only_changes=['evaluation_wall_s10800to18000','global_wall_s14400to24000','new execution namespace and explicit provenance'],unchanged='Original32 choices/noise/observations,prior,models,inference and accuracy gates; retained trial0 plus untouched1..31; no RNG or PDE',remaining=parent.read(parent.LOCAL/'protocol.json')['evaluation_remaining'])
    write(LOCAL/'protocol.json',value);return record(LOCAL/'protocol.json')

def validate(digest):
    p=parent.validate(digest);v=read(LOCAL/'protocol.json');r=read(LOCAL/'release.json')
    if not(v.get('schema')=='noisy-poke-resource-continuation-v02' and v['parent_protocol_SHA256']==digest and v['source_SHA256']==sources() and v['resources']==RESOURCES and v['remaining']==p['evaluation_remaining']):raise ValueError('Exact versioned continuation source/resources/worklist')
    # Hash checks only; origin() additionally proves original phase closure on launch.
    for key in ('parent_protocol','parent_release','inputs','choice_execution','choice_closure','checkpoint_execution','checkpoint_closure'):exact(v['origin'][key])
    if not(r.get('approved')is True and r.get('protocol')==record(LOCAL/'protocol.json')and r.get('source_SHA256')==sources()and r.get('resources')==RESOURCES and r.get('parent_protocol_SHA256')==digest):raise ValueError('Exact ROOT resource-only continuation release')
    return p

def execution_context():
    return dict(execution_protocol=record(LOCAL/'protocol.json'),execution_release=record(LOCAL/'release.json'),execution_source_SHA256=sources())

def guard(role,slot=None):
    # Identical memory/disk/UUID/affinity guards; only namespace and wall caps differ.
    cpu=RESOURCES['worker_CPUs'][slot]if role=='worker'else RESOURCES[role+'_CPU']
    if set(os.sched_getaffinity(0))!={cpu}or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact continuation affinity/one thread')
    if role=='worker':base.old.parity.GPU.pin_config(base.old.parity.GPU.ALLOWED_UUIDS[slot])
    elif os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Controller no CUDA')
    limit=RESOURCES['controller_RSS_bytes']if role=='controller'else RESOURCES['worker_RSS_bytes']
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=limit:raise MemoryError('Unchanged continuation RSS cap')
    if parent.available_memory()<RESOURCES['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']+RESOURCES['maximum_writes_bytes']:raise MemoryError('Unchanged RAM/disk reserves')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise MemoryError('Unchanged own writes cap')

available_memory=parent.available_memory

def continuation_gate(digest):
    p=validate(digest)
    if read(LOCAL/'protocol.json')['origin']!=origin(digest):raise ValueError('Changed original checkpoint/seals/noise or projection')
    return p

def evaluate_worker(slot,phase,digest):
    if phase!='remaining':raise ValueError('Only untouched remaining31 trials')
    p=validate(digest);guard('worker',slot);seals=all_seals_gate(digest);m=parent.inputs_gate(digest)
    # No candidate/held array, model or noise access above the complete seal barrier.
    models,callbacks,names=load_common(p,slot);indices=p['evaluation_remaining'][slot];start=time.monotonic();state=dict(**execution_context(),completed=False,pid=os.getpid(),slot=slot,phase=phase,physical_UUID=p['devices'][slot],protocol_SHA256=digest,release=record(LOCAL/'release.json'),all32_choice_seals=seals,trials=[]);path=LOCAL/f'evaluation-{phase}-worker-{slot}.json'
    if path.exists():raise FileExistsError('No evaluation worker retry')
    with np.load(ROOT/m['poke_noise']['path'],allow_pickle=False)as z:noise=z['poke_noise'].copy()
    if noise.shape!=(4,8,8)or not np.isfinite(noise).all():raise ValueError('Exact fourreplicate noise archive')
    def checkpoint():guard('worker',slot);state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(path,state)
    try:
        for index in indices:
            trial=p['trials'][index];i,r=trial['object_index'],trial['replicate'];directory=LOCAL/f'trial-{index:02d}'/'evaluation';directory.mkdir(parents=True,exist_ok=True)
            if any(directory.iterdir()):raise FileExistsError('No evaluation partial reuse')
            seal=read(ROOT/seals[index]['seal']['path']);prefix=read(ROOT/seal['prefix_result']['path']);values,warm,captures=query_truth(p,i);parameters=np.array([p['objects'][i]['E_Pa'],p['objects'][i]['K_N_m']]);unit=(np.log(parameters)-np.log([2000.,5.]))/np.log([4.,6.]);layout=next(k for k,v in enumerate(design.physics.LAYOUTS)if v['id']==p['objects'][i]['layout']);oracle=models[layout].select(names['target'])(unit[None])[0,:,0];truth=values[12:]
            truth_path=directory/'truth-LOCAL.npz';np.savez_compressed(truth_path,NH_target_m=truth,n12_linear_target_m=warm,n6_true_parameter_oracle_m=oracle,clean_candidate_m=values[4:12]);out=dict(**execution_context(),index=index,object_index=i,replicate=r,protocol_SHA256=digest,choice_seal=seals[index]['seal'],prefix_result=seal['prefix_result'],truth=record(truth_path),source_captures=captures,posteriors=[],qualified=False,oracle_metrics=dict(n12_linear_vs_n12_NH_MSE_m2=float(np.mean((warm-truth)**2)),n6_modal_vs_n12_linear_MSE_m2=float(np.mean((oracle-warm)**2)),n6_modal_vs_n12_NH_MSE_m2=float(np.mean((oracle-truth)**2))))
            with np.load(ROOT/m['prefixes'][index]['record']['path'],allow_pickle=False)as z:observed=design.prefix_only(z['observed_prefix'])
            before=None
            if prefix.get('qualified')is True:before=snapshot_metrics(ROOT/prefix['snapshot']['path'],truth,parameters);out['before']=before
            after=[]
            for action in range(8):
                action_dir=directory/f'action-{action}';action_dir.mkdir();obs=list(zip(names['prefix'],observed[:,None]))+[(names['candidate'][action],np.array([values[4+action]+noise[r,i,action]]))];obs_path=action_dir/'observations-LOCAL.npz';np.savez_compressed(obs_path,labels=np.array([v[0]for v in obs]),values=np.array([v[1]for v in obs]));begin=time.monotonic();result,refs,finals,_=infer(obs,callbacks,names,action_dir,checkpoint);result.update(**execution_context(),index=index,action=action,protocol_SHA256=digest,choice_seal=seals[index]['seal'],observations=record(obs_path),observations_fingerprint=observations_fingerprint(obs),prefix_plus_ONE_reset_poke=True,wall_s=time.monotonic()-begin);target=action_dir/'result-LOCAL.json';write(target,result);entry=dict(action=action,qualified=result['qualified'],record=record(target))
                if result['qualified']:entry['metrics']=snapshot_metrics(ROOT/result['snapshot']['path'],truth,parameters);after.append(entry['metrics']['field'])
                else:after.append(None)
                out['posteriors'].append(entry);del refs,finals;gc.collect();checkpoint()
            out['qualified']=bool(before is not None and seal['qualified']and all(v['qualified']for v in out['posteriors']))
            if out['qualified']:out['policies']=design.compare_policies(before['field'],after,int(seal['adaptive_action']),int(seal['fixed_action']));out['material_policies']=design.compare_material(before['material'],[v['metrics']['material']for v in out['posteriors']],int(seal['adaptive_action']),int(seal['fixed_action']))
            target=directory/'result-LOCAL.json';write(target,out);state['trials'].append(dict(index=index,qualified=out['qualified'],record=record(target)));checkpoint()
        state['completed']=True
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:checkpoint()
    return state

def run(phase,digest):
    p=validate(digest);guard('controller')
    if phase!='remaining':raise ValueError('Fixed threephase worklist')
    continuation_gate(digest)
    worker_count=3
    if available_memory()<worker_count*RESOURCES['worker_RSS_bytes']+RESOURCES['controller_RSS_bytes']+RESOURCES['RAM_reserve_bytes']:raise MemoryError('Initial owned worker/controller budgets plus30GiBreserve')
    execution=LOCAL/f'execution-{phase}.json'
    if execution.exists():raise FileExistsError('No phase retry/overwrite')
    prior=sum(read(parent.LOCAL/f'execution-{v}.json')['wall_s']for v in('choices','checkpoint'));start=time.monotonic();children=[];state=dict(**execution_context(),completed=False,controller_pid=os.getpid(),phase=phase,protocol_SHA256=digest,release=record(LOCAL/'release.json'),workers=[])
    def save():state['wall_s']=time.monotonic()-start;write(execution,state)
    def stop(*_):raise InterruptedError('Owned benchmark controller stop')
    previous={s:signal.signal(s,stop)for s in(signal.SIGTERM,signal.SIGINT)}
    try:
        for slot in([0]if phase=='checkpoint'else range(3)):
            action='choose-worker'if phase=='choices'else'evaluate-worker';command=['taskset','-c',str(slot),sys.executable,'-u','-m','experiments.run_noisy_poke_continuation_v02',action,'--slot',str(slot),'--sha',digest]
            if phase!='choices':command+=['--phase',phase]
            with(LOCAL/f'{phase}-worker-{slot}.log').open('xb')as log:child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':p['devices'][slot],'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child);state['workers'].append(dict(slot=slot,pid=child.pid,physical_UUID=p['devices'][slot],command=command));save()
        while any(c.poll()is None for c in children):
            total=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            for child,row in zip(children,state['workers']):
                if child.poll()is None:
                    rss=base.old.parity.old.old.mechanical_screen.shared.owned_RSS(child);vram=base.old.parity.owned_VRAM(child.pid,row['physical_UUID']);total+=rss;row['peak_RSS_bytes']=max(row.get('peak_RSS_bytes',0),rss);row['peak_owned_VRAM_bytes']=max(row.get('peak_owned_VRAM_bytes',0),vram)
                    if rss>=RESOURCES['worker_RSS_bytes']or vram>=RESOURCES['owned_VRAM_bytes']:raise MemoryError('Owned worker RSS/VRAM cap')
                elif child.returncode!=0:raise RuntimeError('Owned stage failed; preserve all planned trials')
            limit=RESOURCES['choice_wall_s']if phase=='choices'else RESOURCES['evaluation_wall_s']
            if time.monotonic()-start>limit or prior+time.monotonic()-start>RESOURCES['global_wall_s']:raise TimeoutError('Frozen stage/global benchmark wall cap')
            if total>=RESOURCES['aggregate_RSS_bytes']:raise MemoryError('Own aggregate28GiB cap')
            state['peak_aggregate_RSS_bytes']=max(state.get('peak_aggregate_RSS_bytes',0),total);guard('controller');save();time.sleep(.5)
        for child,row in zip(children,state['workers']):
            base.finalize_child(child,row);path=LOCAL/(f"choices-worker-{row['slot']}.json"if phase=='choices'else f"evaluation-{phase}-worker-{row['slot']}.json");row['result']=record(path);value=read(path)
            if value.get('completed')is not True or value['pid']!=row['pid']or value['slot']!=row['slot']or value['protocol_SHA256']!=digest or value['release']!=record(LOCAL/'release.json')or value['phase']!=phase or value['physical_UUID']!=row['physical_UUID']:raise ValueError('Exact launched worker completion/context')
            expected=p['choice_split'][row['slot']]if phase=='choices'else(p['evaluation_checkpoint']if phase=='checkpoint'else p['evaluation_remaining'][row['slot']])
            if[v['index']for v in value['trials']]!=expected:raise ValueError('Every exact assigned trial retained at worker completion')
        state['completed']=True
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error))
    finally:
        for child,row in zip(children,state['workers']):row.update(cleanup_exit_code=base.old.parity.old.old.mechanical_screen.shared.old.cost.cleanup(child),exit_code=child.returncode,reaped=True)
        for s,h in previous.items():signal.signal(s,h)
        state['terminal']=True;save()
    return state

def external_watchdog(e,c,digest):
    operator=ROOT/'runs/operator-dev/noisy-poke-recovery-operator-v02'
    approval_path=operator/'root-preapproval.json';stage_path=operator/'stages/remaining/execution.json'
    if c.get('operator_preapproval')!=record(approval_path)or c.get('operator_stage_execution')!=record(stage_path):raise ValueError('Canonical external watchdog commitments')
    approval=read(exact(c['operator_preapproval']));stage=read(exact(c['operator_stage_execution']))
    caps=[18,18060,RESOURCES['aggregate_RSS_bytes']]
    pins=approval.get('source_SHA256',{})
    required={**sources(),**{name:sha(ROOT/name)for name in('experiments/supervise_noisy_poke_recovery_v02.py','tests/test_supervise_noisy_poke_recovery_v02.py')}}
    if not(approval.get('approved')is True and approval.get('schema')=='noisy-poke-recovery-operator-preapproval-v02'and all(pins.get(k)==v for k,v in required.items())and approval.get('stage_caps',{}).get('remaining')==caps):raise ValueError('Source-pinned external watchdog approval/caps')
    command=['taskset','-c','18',sys.executable,'-u','-m','experiments.run_noisy_poke_continuation_v02','run','--sha',digest]
    if not(stage.get('completed')is True and stage.get('reaped')is True and stage.get('exit_code')==0 and stage.get('cleanup_exit_code')==0 and 'failure'not in stage and stage.get('phase')=='remaining'and stage.get('command')==command and stage.get('parent_protocol_SHA256')==digest and stage.get('source_SHA256')==pins and stage.get('operator_preapproval')==c['operator_preapproval']and stage.get('resources')==dict(zip(('CPU','wall_s','RSS_bytes'),caps))and stage.get('outputs')==[record(LOCAL/'execution-remaining.json')]and stage.get('child_identity',{}).get('pid')==e['controller_pid']):raise ValueError('Exact successful external watchdog stage')
    if not(0<=stage.get('wall_s',float('inf'))<=caps[1]and 0<=stage.get('peak_RSS_bytes',float('inf'))<caps[2]):raise ValueError('External watchdog measured limits')
    observed={v['pid']for v in stage.get('observed_owned',[])}
    if not set(c['owned_pids'])<=observed:raise ValueError('Every controller/worker observed by external watchdog')
    for pid in observed:base.absent(pid)

def completed_remaining(digest):
    p=validate(digest);e=read(LOCAL/'execution-remaining.json');context=execution_context()
    if not(e.get('completed')is True and e.get('terminal')is True and 'failure'not in e and e['phase']=='remaining'and e['protocol_SHA256']==digest and e['release']==record(LOCAL/'release.json')and all(e.get(k)==v for k,v in context.items())):raise ValueError('Closed new remaining execution/source')
    if [v['slot']for v in e['workers']]!=[0,1,2]:raise ValueError('All3 exact continuation slots')
    owned=[e['controller_pid']]
    for row in e['workers']:
        slot=row['slot'];owned.append(row['pid']);path=LOCAL/f'evaluation-remaining-worker-{slot}.json';w=read(path)
        if not(row.get('exit_code')==0 and row.get('cleanup_exit_code')==0 and row.get('reaped')is True and row['physical_UUID']==p['devices'][slot]and row['result']==record(path)):raise ValueError('Exact newworker closed receipt')
        if not(w.get('completed')is True and 'failure'not in w and w['pid']==row['pid']and w['slot']==slot and w['phase']=='remaining'and w['protocol_SHA256']==digest and w['release']==record(LOCAL/'release.json')and w['physical_UUID']==p['devices'][slot]and all(w.get(k)==v for k,v in context.items())and [v['index']for v in w['trials']]==p['evaluation_remaining'][slot]):raise ValueError('Exact unchanged remaining worklist/context')
        parent.verify(w['trials'])
    c=read(LOCAL/'root-closure-remaining.json')
    if not(c.get('root_reaped')is True and c.get('root_exit_code')==0 and c.get('all_owned_PID_and_groups_absent')is True and c.get('execution')==record(LOCAL/'execution-remaining.json')and c.get('protocol')==record(LOCAL/'protocol.json')and c.get('source_SHA256')==sources()and c.get('owned_pids')==owned):raise ValueError('External versioned remaining root closure')
    external_watchdog(e,c,digest)
    for pid in owned:base.absent(pid)
    return e

def evaluation_directory(index):
    return (parent.LOCAL if index==0 else LOCAL)/f'trial-{index:02d}'/'evaluation'

def metadata_barrier(digest):
    p=continuation_gate(digest)
    phases={name:parent.completed_phase(name,digest)for name in('choices','checkpoint')}
    phases['remaining']=completed_remaining(digest)
    seals=parent.all_seals_gate(digest);inputs=parent.inputs_gate(digest)
    if sum(v['wall_s']for v in phases.values())>RESOURCES['global_wall_s']or phases['remaining']['wall_s']>RESOURCES['evaluation_wall_s']:raise ValueError('Revised global/evaluation wall caps')
    entries=[]
    for name in ('checkpoint','remaining'):
        for worker in phases[name]['workers']:
            for row in read(exact(worker['result']))['trials']:
                if row['record']!=record(evaluation_directory(row['index'])/'result-LOCAL.json'):raise ValueError('Exact mixed old0/new1..31 path')
                entries.append(row)
    entries.sort(key=lambda r:r['index'])
    if [v['index']for v in entries]!=list(range(32)):raise ValueError('All32 exact mixed memberships; no exclusions')
    return p,phases,seals,inputs,entries

def summarize(digest):
    p,phases,seals,inputs,entries=metadata_barrier(digest)
    rows=[read(exact(v['record']))for v in entries]
    target=LOCAL/'summary-LOCAL.json'
    if target.exists():raise FileExistsError('No summary overwrite')
    value=dict(**execution_context(),parent_protocol=record(parent.LOCAL/'protocol.json'),protocol_SHA256=digest,summary=design.aggregate(rows),all32_trial_records=entries,phases=[record((parent.LOCAL if name!='remaining'else LOCAL)/f'execution-{name}.json')for name in('choices','checkpoint','remaining')],scope='Original fixed32 design; trial0 reused,31 resource-only continuation; all numerical accuracy gates unchanged')
    write(target,value);return record(target)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','evaluate-worker','summarize']);ap.add_argument('--sha',required=True);ap.add_argument('--slot',type=int);ap.add_argument('--phase',default='remaining');a=ap.parse_args()
    result=prepare(a.sha)if a.action=='prepare'else(run(a.phase,a.sha)if a.action=='run'else(evaluate_worker(a.slot,a.phase,a.sha)if a.action=='evaluate-worker'else summarize(a.sha)))
    print(json.dumps(result))
    if a.action=='run'and not result['completed']:raise SystemExit(1)
