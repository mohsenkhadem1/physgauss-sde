"""All eight actions, SAME saved old LINEAR case; numerical ranking only."""
import argparse,ast,json,os,resource,signal,subprocess,sys,time,shutil
from pathlib import Path
import numpy as np
from experiments import run_layout_grouped_gpu_parity_v01 as parity
from experiments import audit_layout_grouped_gpu_parity_v01 as audit
ROOT=parity.ROOT;LOCAL=ROOT/'runs/operator-dev/layout-grouped-allactions-v01'
SOURCE='experiments/run_layout_grouped_allactions_v01.py';TEST='tests/test_layout_grouped_allactions_v01.py'
PARITY_SHA=audit.EXPECTED_PROTOCOL
VARIANTS=((512,32),(256,64),(512,64));SPLIT=((0,3,6),(1,4,7),(2,5))
RESOURCES=dict(parity.RESOURCES)
sha,read,write,record=parity.sha,parity.read,parity.write,parity.record

def sources():
    out=parity.sources();out.update({s:sha(ROOT/s)for s in(SOURCE,TEST,audit.SOURCE,audit.TEST)});return dict(sorted(out.items()))
def origin_gate():
    parity.validate(PARITY_SHA);review=read(parity.LOCAL/'independent-terminal-review.json');closure=read(parity.LOCAL/'root-closure.json');execution=read(parity.LOCAL/'execution.json')
    if review['summary'].get('all3_GPU_parity_qualified')is not True or closure.get('root_exit_code')!=0 or closure.get('root_reaped')is not True or closure['execution_SHA256']!=sha(parity.LOCAL/'execution.json')or closure['protocol_SHA256']!=PARITY_SHA:raise ValueError('Qualified independently audited samecase GPU parity')
    if review['source_SHA256']!={s:sha(ROOT/s)for s in(audit.SOURCE,audit.TEST)}:raise ValueError('Exact independent audit procedure')
    for pid in(execution['controller_pid'],*[r['pid']for r in execution['devices']]):parity.old.old.mechanical_screen.shared.old.absent(pid)
    return dict(parity_protocol=record(parity.LOCAL/'protocol.json'),parity_release=record(parity.LOCAL/'release.json'),parity_execution=record(parity.LOCAL/'execution.json'),parity_review=record(parity.LOCAL/'independent-terminal-review.json'),root_closure=record(parity.LOCAL/'root-closure.json'),original=read(parity.LOCAL/'protocol.json')['origin'],decision_records=[record(parity.old.LOCAL/f'decision-{o}-{a}-LOCAL.json')for o,a in VARIANTS])
def worklist():return [dict(outer=o,amplitude=a,action=i,order=n)for i in range(8)for o,a in VARIANTS for n in(8,16)]
def prepare():
    origin=origin_gate()
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze/retry')
    p=dict(schema='layout-grouped-allactions-v01',source_SHA256=sources(),origin=origin,resources=RESOURCES,devices=list(parity.GPU.ALLOWED_UUIDS),split=[list(x)for x in SPLIT],worklist=worklist(),gates=dict(outcome_relative=1e-4,latent_relative=1e-3),tie_rule='Unique winner only if risk margin strictly exceeds sum of final-action outcome+amplitude+outer refinement indicators; otherwise numerical tie. Indicators are not rigorous error bounds.',same_saved_prefix=True,no_RNG=True,no_PDE=True,no_generating_or_held_truth=True,development_only=True)
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')
def validate(digest):
    p=read(LOCAL/'protocol.json');r=read(LOCAL/'release.json')
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=RESOURCES or p['devices']!=list(parity.GPU.ALLOWED_UUIDS)or p['split']!=[list(x)for x in SPLIT]or p['worklist']!=worklist()or p['gates']!=dict(outcome_relative=1e-4,latent_relative=1e-3):raise ValueError('Exact source/config/worklist')
    parity.old.old.verify_mechanics_receipt(p['origin'])
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('origin')!=p['origin']:raise ValueError('Explicit root allactions release')
    e=read(ROOT/p['origin']['parity_execution']['path'])
    for pid in(e['controller_pid'],*[x['pid']for x in e['devices']]):parity.old.old.mechanical_screen.shared.old.absent(pid)
    return p

def guard(role,uuid=None):
    cpu=RESOURCES['worker_CPU']if role=='worker'else RESOURCES['controller_CPU']
    if set(os.sched_getaffinity(0))!={cpu}or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact own worker16/controller19 one-thread affinity')
    if role=='worker':parity.GPU.pin_config(uuid)
    elif os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Controller CUDA hidden')
    available=int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024
    if available<RESOURCES['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise MemoryError('Own30GiB RAM/disk reserve')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=RESOURCES['aggregate_host_RSS_bytes']:raise MemoryError('OwnhostRSS cap')
    if LOCAL.exists()and sum(x.stat().st_size for x in LOCAL.rglob('*')if x.is_file())>RESOURCES['maximum_writes_bytes']:raise MemoryError('Ownwrite cap')

def make_refs(observation):
    names=parity.old.old.labels();loads,_=parity.old.old.mechanics.load_catalog(2);refs=[]
    for layout in parity.old.old.mechanics.LAYOUTS:
        f=parity.old.old.mechanics.make_forward(2,layout['id']);model=parity.old.old.mechanics.make_measurement_modal(f,loads);refs.append(parity.ConditionalAmplitudeReferenceV02(parity.RatioModal(model.modal,model.labels),list(zip(names['prefix'],observation[:,None])),names['target'],retain_hypotheses=False))
    return refs,names

def classify(rows,gates):
    if len(rows)!=48 or{tuple(r['job'][k]for k in('outer','amplitude','action','order'))for r in rows}!={(o,a,i,n)for i in range(8)for o,a in VARIANTS for n in(8,16)}:raise ValueError('All48 unique jobs required')
    for row in rows:
        value=row['result']
        if any(not np.isfinite(value[k])or value[k]<0 for k in('predictive_mass','omitted_mass_bound','omitted_risk_bound'))or abs(value['predictive_mass']-1)>1e-10+value['omitted_mass_bound']:raise ValueError('Every outcome integral requires finite positive predictive mass and tail bounds')
    table={(r['job']['outer'],r['job']['amplitude'],r['job']['action'],r['job']['order']):r['result']for r in rows};actions=[]
    for i in range(8):
        errors=[];outcome_ok=True
        for o,a in VARIANTS:
            coarse,fine=table[o,a,i,8],table[o,a,i,16];prior=fine['prior_target_variance'];error=abs(coarse['expected_target_variance']-fine['expected_target_variance'])+fine['omitted_risk_bound'];errors.append(error);outcome_ok=outcome_ok and np.isfinite(error)and error<=gates['outcome_relative']*prior
        f=table[512,64,i,16];risk=f['expected_target_variance'];prior=f['prior_target_variance'];amp=abs(table[512,32,i,16]['expected_target_variance']-risk);outer=abs(table[256,64,i,16]['expected_target_variance']-risk);valid=bool(outcome_ok and np.isfinite([risk,prior,amp,outer]).all()and prior>0 and 0<=risk<=prior and amp<=gates['latent_relative']*prior and outer<=gates['latent_relative']*prior)
        actions.append(dict(action=i,qualified=valid,risk=risk,uncertainty=errors[-1]+amp+outer,outcome_errors=errors,amplitude_error=amp,outer_error=outer))
    all_ok=all(a['qualified']for a in actions);best=min(actions,key=lambda a:(a['risk'],a['action']));ties=[a['action']for a in actions if a['risk']-best['risk']<=a['uncertainty']+best['uncertainty']]
    return dict(qualified=all_ok,actions=actions,numerical_ties=ties,unique_best_action=best['action']if all_ok and len(ties)==1 else None,all8_retained=True,scope='SAME saved LINEAR development case only; refinement indicators are not rigorous error bounds; no active advantage or new physics admission')

def worker(slot,digest):
    p=validate(digest)
    if type(slot)is not int or slot not in range(3):raise ValueError('Fixed slot')
    uuid=p['devices'][slot];guard('worker',uuid);directory=LOCAL/f'worker-{slot}';directory.mkdir(parents=True,exist_ok=True);path=directory/'result-LOCAL.json'
    if path.exists():raise FileExistsError('No worker retry')
    start=time.monotonic();state=dict(completed=False,qualified=False,pid=os.getpid(),slot=slot,physical_UUID=uuid,protocol_SHA256=digest,release=record(LOCAL/'release.json'),jobs=[],bank_replay=[])
    try:
        with np.load(ROOT/p['origin']['original']['prefix_observation']['path'],allow_pickle=False)as z:observation=z['observed_prefix'].copy()
        refs,names=make_refs(observation)
        for outer,amp in VARIANTS:
            bank,meta=parity.CPU.decision_bank(refs,names['candidate'],names['target'],outer=outer,amplitude_order=amp);saved=read(parity.old.LOCAL/f'decision-{outer}-{amp}-LOCAL.json')['metadata'];le=float(np.max(abs(np.asarray(meta['log_evidence'])-saved['log_evidence'])));pe=float(np.max(abs(np.asarray(meta['layout_probability'])-saved['layout_probability'])))
            if le>1e-8 or pe>1e-10 or meta['weighted_hypotheses']!=saved['weighted_hypotheses']:raise ValueError('Same admitted decision bank replay')
            state['bank_replay'].append(dict(outer=outer,amplitude=amp,logZ_error=le,layout_probability_error=pe));engine=parity.GPU.Float64GroupedGPU(bank,physical_UUID=uuid,outcome_chunk=8,allocated_cap_bytes=RESOURCES['torch_allocated_bytes'])
            for action in p['split'][slot]:
                for order in(8,16):
                    begin=time.monotonic();result=engine.expected_risk(action,parity.old.old.mechanics.SIGMA_M,order=order);engine.torch.cuda.synchronize();state['jobs'].append(dict(job=dict(outer=outer,amplitude=amp,action=action,order=order),result=result,wall_s=time.monotonic()-begin));guard('worker',uuid);write(path,state)
            del engine
        state.update(completed=True,qualified=True)
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(path,state)
    return state

def worker_context(value,row,p,digest):
    if value.get('completed')is not True or value.get('qualified')is not True or value.get('protocol_SHA256')!=digest or value.get('release')!=record(LOCAL/'release.json')or value.get('slot')!=row['slot']or value.get('pid')!=row['pid']or value.get('physical_UUID')!=row['physical_UUID']or row['physical_UUID']!=p['devices'][row['slot']]:raise ValueError('Exact launched worker protocol/release/slot/PID/physicalUUID context')
    expected=[j for j in p['worklist']if j['action']in p['split'][row['slot']]]
    actual=[v['job']for v in value['jobs']]
    if len(actual)!=len(expected)or{tuple(v[k]for k in('outer','amplitude','action','order'))for v in actual}!={tuple(v[k]for k in('outer','amplitude','action','order'))for v in expected}:raise ValueError('Exact assigned per-device jobs')
    return value

def run(digest):
    p=validate(digest);guard('controller')
    if(LOCAL/'execution.json').exists():raise FileExistsError('No controller retry')
    state=dict(completed=False,qualified=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),devices=[dict(slot=i,physical_UUID=u,status='pending')for i,u in enumerate(p['devices'])]);start=time.monotonic();child=None
    def checkpoint():state['wall_s']=time.monotonic()-start;write(LOCAL/'execution.json',state)
    def stop(*_):raise InterruptedError('Owned controller interruption')
    previous={s:signal.signal(s,stop)for s in(signal.SIGTERM,signal.SIGINT)}
    try:
        for slot,uuid in enumerate(p['devices']):
            guard('controller');row=state['devices'][slot];begin=time.monotonic()
            with(LOCAL/f'worker-{slot}.log').open('xb')as log:
                command=['taskset','-c','16',sys.executable,'-u','-m','experiments.run_layout_grouped_allactions_v01','worker','--slot',str(slot),'--sha',digest];child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row.update(pid=child.pid,status='running',command=command);checkpoint()
                while child.poll()is None:
                    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+parity.old.old.mechanical_screen.shared.owned_RSS(child);vram=parity.owned_VRAM(child.pid,uuid);row['peak_host_RSS_bytes']=max(row.get('peak_host_RSS_bytes',0),rss);row['peak_owned_VRAM_bytes']=max(row.get('peak_owned_VRAM_bytes',0),vram)
                    if rss>=RESOURCES['aggregate_host_RSS_bytes']or vram>=RESOURCES['owned_VRAM_bytes']:raise MemoryError('Owned memory cap')
                    if time.monotonic()-begin>120 or time.monotonic()-start>420:raise TimeoutError('Owned fixed wall caps')
                    guard('controller');time.sleep(.2)
                row.update(exit_code=child.wait(),reaped=True,cleanup_exit_code=parity.old.old.mechanical_screen.shared.old.cost.cleanup(child));path=LOCAL/f'worker-{slot}/result-LOCAL.json'
                if path.exists():row['result']=record(path)
                if row['exit_code']!=0 or row['cleanup_exit_code']!=0 or not path.exists()or read(path).get('qualified')is not True:raise RuntimeError('Unresolved planned action worklist; no retry')
                worker_context(read(path),row,p,digest);row['status']='qualified';checkpoint();child=None
        rows=[job for row in state['devices']for job in read(ROOT/row['result']['path'])['jobs']];state['ranking']=classify(rows,p['gates']);state.update(completed=True,qualified=state['ranking']['qualified'])
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error))
    finally:
        if child is not None:
            state['devices'][slot].update(cleanup_exit_code=parity.old.old.mechanical_screen.shared.old.cost.cleanup(child),reaped=True,status='unresolved');path=LOCAL/f'worker-{slot}/result-LOCAL.json'
            if path.exists():state['devices'][slot]['result']=record(path)
        for s,h in previous.items():signal.signal(s,h)
        state['terminal']=True;checkpoint()
    return state
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','worker','worker-guard']);ap.add_argument('--sha');ap.add_argument('--slot',type=int);a=ap.parse_args()
    if a.action=='worker-guard':guard('worker',parity.GPU.ALLOWED_UUIDS[a.slot]);print('guard passed; no model/CUDA')
    else:
        result=prepare()if a.action=='prepare'else(worker(a.slot,a.sha)if a.action=='worker'else run(a.sha));print(json.dumps(result))
        if a.action=='run'and not result['completed']:raise SystemExit(1)
