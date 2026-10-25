"""Three pinned-device SAME qualified bank replays; no new inference or data.

SequentialGPU0–2, ownCPU16worker/CPU19controller. All3 devices remain in the
fixed worklist on failure. No physical/new-layout/policy admission is implied.
"""
import argparse,ast,json,os,resource,signal,subprocess,sys,time,shutil
from pathlib import Path
import numpy as np
from experiments import run_layout_grouped_reference_cost_v01 as old
from experiments import audit_layout_grouped_reference_cost_v01 as independent
from experiments import campaign_grouped_linear_gpu_v01 as GPU
from experiments import campaign_grouped_linear_decision_v01 as CPU
from experiments.campaign_surface_conditional_amplitude_v01 import RatioModal
from experiments.campaign_surface_conditional_amplitude_v02 import ConditionalAmplitudeReferenceV02
ROOT=old.ROOT;LOCAL=ROOT/'runs/operator-dev/layout-grouped-gpu-parity-v01'
SOURCE='experiments/run_layout_grouped_gpu_parity_v01.py';TEST='tests/test_layout_grouped_gpu_parity_v01.py'
ORIGIN_SHA=independent.EXPECTED_PROTOCOL
RESOURCES=dict(worker_CPU=16,controller_CPU=19,threads=1,worker_wall_s=120,global_wall_s=420,aggregate_host_RSS_bytes=4*1024**3,torch_allocated_bytes=512*1024**2,owned_VRAM_bytes=2*1024**3,outcome_chunk=8,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=64*1024**2)
sha,record,read,write=old.sha,old.record,old.read,old.write

def sources():
    todo=[SOURCE,TEST,independent.SOURCE,independent.TEST,'experiments/campaign_grouped_linear_gpu_v01.py','tests/test_grouped_linear_gpu_v01.py',*old.sources()];seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            mods=[]
            if isinstance(node,ast.Import):mods=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:mods=[node.module or '']+[(node.module+'.'+a.name)for a in node.names if node.module]
            for mod in mods:
                path=mod.replace('.','/')+'.py'
                if(ROOT/path).is_file()and path not in seen:todo.append(path)
    return {name:sha(ROOT/name)for name in sorted(seen)}

def origin_gate():
    old.validate(ORIGIN_SHA);review=read(old.LOCAL/'independent-terminal-review.json');s=review['summary'];e=read(old.LOCAL/'root-execution.json');w=read(old.LOCAL/'execution.json')
    if s.get('integrity_passed')is not True or s.get('execution_clean')is not True or s.get('continuous_categories_qualified')!=4 or s.get('oneaction_numerically_qualified')is not True or s.get('all8_ranks_pending')is not True or review['source_SHA256']!={name:sha(ROOT/name)for name in(independent.SOURCE,independent.TEST)}:raise ValueError('Exact independently qualified old oneaction bank required')
    if e.get('completed')is not True or e.get('qualified')is not True or e.get('reaped')is not True or e.get('exit_code')!=0 or e.get('cleanup_exit_code')!=0 or w.get('qualified')is not True:raise ValueError('Closed original cost execution')
    for pid in(e['controller_pid'],e['child_pid']):old.old.mechanical_screen.shared.old.absent(pid)
    receipt=dict(protocol=record(old.LOCAL/'protocol.json'),release=record(old.LOCAL/'release.json'),execution=record(old.LOCAL/'execution.json'),external=record(old.LOCAL/'root-execution.json'),review=record(old.LOCAL/'independent-terminal-review.json'),prefix_observation=read(old.LOCAL/'protocol.json')['origin']['prefix_observation'],decision=record(old.LOCAL/'decision-512-64-LOCAL.json'),utility=record(old.LOCAL/'utility-512-64-LOCAL.json'))
    old.old.verify_mechanics_receipt(receipt);return receipt

def prepare():
    origin=origin_gate()
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No parity refreeze/retry')
    p=dict(schema='layout-grouped-gpu-parity-v01',source_SHA256=sources(),origin=origin,resources=RESOURCES,devices=list(GPU.ALLOWED_UUIDS),bank=dict(outer=512,amplitude_order=64,action=0,outcome_orders=[8,16]),tolerances=dict(risk_absolute_m2=1e-20,risk_relative=1e-10,predictive_mass=2e-12,bank_logZ=1e-8,bank_layout_probability=1e-10,conditional_log_density=1e-10),same_saved_prefix=True,no_reference_requalification=True,no_generating_or_held_truth=True,no_RNG=True,all8_ranks='PENDING_NOT_ADMITTED')
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')

def validate(digest):
    p=read(LOCAL/'protocol.json')
    expected=dict(risk_absolute_m2=1e-20,risk_relative=1e-10,predictive_mass=2e-12,bank_logZ=1e-8,bank_layout_probability=1e-10,conditional_log_density=1e-10)
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=RESOURCES or p['devices']!=list(GPU.ALLOWED_UUIDS)or p['bank']!=dict(outer=512,amplitude_order=64,action=0,outcome_orders=[8,16])or p['tolerances']!=expected:raise ValueError('Exact frozen parity source/config/devices')
    old.old.verify_mechanics_receipt(p['origin']);e=read(ROOT/p['origin']['external']['path'])
    for pid in(e['controller_pid'],e['child_pid']):old.old.mechanical_screen.shared.old.absent(pid)
    r=read(LOCAL/'release.json')
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('origin')!=p['origin']:raise ValueError('Explicit root same-bank GPU parity release')
    return p

def guard(role,uuid=None):
    cpu=RESOURCES['worker_CPU']if role=='worker'else RESOURCES['controller_CPU']
    if set(os.sched_getaffinity(0))!={cpu}or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact one-thread worker16/controller19 affinity')
    if role=='worker':GPU.pin_config(uuid)
    elif os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Controller has no visible CUDA device')
    available=int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024
    if available<RESOURCES['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise MemoryError('30GiB host/disk reserve')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=RESOURCES['aggregate_host_RSS_bytes']:raise MemoryError('Owned host RSS')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise MemoryError('64MiB owned writes')

def replay_bank(observation):
    """No true layout/parameters and no future observation/target data input."""
    names=old.old.labels();loads,_=old.old.mechanics.load_catalog(2);refs=[]
    for layout in old.old.mechanics.LAYOUTS:
        f=old.old.mechanics.make_forward(2,layout['id']);model=old.old.mechanics.make_measurement_modal(f,loads);refs.append(ConditionalAmplitudeReferenceV02(RatioModal(model.modal,model.labels),list(zip(names['prefix'],observation[:,None])),names['target'],retain_hypotheses=False))
    return CPU.decision_bank(refs,names['candidate'],names['target'],outer=512,amplitude_order=64)

def compare(saved,actual,tolerances):
    error=abs(saved['expected_target_variance']-actual['expected_target_variance']);prior=abs(saved['prior_target_variance']-actual['prior_target_variance']);bound=tolerances['risk_absolute_m2']+tolerances['risk_relative']*abs(saved['prior_target_variance']);mass=abs(saved['predictive_mass']-actual['predictive_mass'])
    return dict(passed=bool(error<=bound and prior<=bound and mass<=tolerances['predictive_mass']and saved['outcome_nodes']==actual['outcome_nodes']),risk_error_m2=error,prior_variance_error_m2=prior,mass_error=mass,risk_tolerance_m2=bound)

def worker(slot,digest):
    p=validate(digest)
    if type(slot)is not int or not 0<=slot<3:raise ValueError('Fixed GPU worklist slot')
    uuid=p['devices'][slot];guard('worker',uuid);directory=LOCAL/f'worker-{slot}';path=directory/'result-LOCAL.json'
    if path.exists():raise FileExistsError('No GPU parity retry')
    directory.mkdir(parents=True,exist_ok=True);start=time.monotonic();state=dict(completed=False,qualified=False,pid=os.getpid(),slot=slot,physical_UUID=uuid,protocol_SHA256=digest,release=record(LOCAL/'release.json'),oneaction_only=True,all8_ranks_pending=True,no_new_RNG=True,no_new_observation=True,no_reference_requalification=True)
    try:
        with np.load(ROOT/p['origin']['prefix_observation']['path'],allow_pickle=False)as z:observation=z['observed_prefix'].copy()
        begin=time.monotonic();bank,metadata=replay_bank(observation);state['model_and_bank_replay_s']=time.monotonic()-begin;saved=read(ROOT/p['origin']['decision']['path'])['metadata'];tol=p['tolerances'];le=float(np.max(abs(np.asarray(metadata['log_evidence'])-saved['log_evidence'])));pe=float(np.max(abs(np.asarray(metadata['layout_probability'])-saved['layout_probability'])));state['bank_replay_errors']=dict(logZ=le,layout_probability=pe)
        if le>tol['bank_logZ']or pe>tol['bank_layout_probability']or metadata['weighted_hypotheses']!=saved['weighted_hypotheses']:raise ValueError('Qualified decision bank replay mismatch before GPU')
        begin=time.monotonic();engine=GPU.Float64GroupedGPU(bank,physical_UUID=uuid,outcome_chunk=8,allocated_cap_bytes=RESOURCES['torch_allocated_bytes']);engine.torch.cuda.synchronize();state['GPU_initialization_s']=time.monotonic()-begin
        lower,upper=bank.response_bounds(bank.candidate_ratio[:,0,None]);ys=np.array([lower[0],(lower[0]+upper[0])/2,upper[0]]);lp,v=CPU.outcome_risk(bank,0,ys,old.old.mechanics.SIGMA_M);lg,vg=engine.outcome_risk(0,ys,old.old.mechanics.SIGMA_M);local_risk=float(np.max(abs(v-vg)));local_lp=float(np.max(abs(lp-lg)));bound=tol['risk_absolute_m2']+tol['risk_relative']*bank.prior_moments()['total_variance'];state['conditional_parity']=dict(passed=bool(local_risk<=bound and local_lp<=tol['conditional_log_density']),risk_error_m2=local_risk,log_density_error=local_lp)
        oldlevels=read(ROOT/p['origin']['utility']['path'])['levels'];state['levels']=[];state['utility_wall_s']={}
        for order,s in zip((8,16),oldlevels):
            begin=time.monotonic();r=engine.expected_risk(0,old.old.mechanics.SIGMA_M,order=order);engine.torch.cuda.synchronize();state['utility_wall_s'][str(order)]=time.monotonic()-begin;state['levels'].append(dict(order=order,parity=compare(s,r,tol),result=r));guard('worker',uuid)
        state.update(completed=True,qualified=state['conditional_parity']['passed']and all(v['parity']['passed']for v in state['levels']),torch_peak_allocated_bytes=int(engine.torch.cuda.max_memory_allocated()),bank_hypotheses=bank.hypotheses)
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(path,state)
    return state

def owned_VRAM(pid,uuid):
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True);total=0
    for line in result.stdout.splitlines():
        parts=[s.strip()for s in line.split(',')]
        if len(parts)==3 and parts[0]==str(pid):
            if parts[1]!=uuid:raise ValueError('Owned worker CUDA allocation on wrong physical device')
            total+=int(parts[2])*1024**2
    return total

def run(digest):
    p=validate(digest);guard('controller')
    if(LOCAL/'execution.json').exists():raise FileExistsError('No parity controller retry')
    start=time.monotonic();state=dict(completed=False,qualified=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),devices=[dict(slot=i,physical_UUID=u,status='pending')for i,u in enumerate(p['devices'])]);child=None
    def checkpoint():state['wall_s']=time.monotonic()-start;write(LOCAL/'execution.json',state)
    def stop(*_):raise InterruptedError('Owned parity controller stop')
    previous={s:signal.signal(s,stop)for s in(signal.SIGTERM,signal.SIGINT)}
    try:
        for slot,uuid in enumerate(p['devices']):
            guard('controller');begin=time.monotonic();row=state['devices'][slot]
            with(LOCAL/f'worker-{slot}.log').open('xb')as log:
                command=['taskset','-c','16',sys.executable,'-u','-m','experiments.run_layout_grouped_gpu_parity_v01','worker','--slot',str(slot),'--sha',digest];child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row.update(pid=child.pid,status='running',command=command);checkpoint()
                while child.poll()is None:
                    host=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+old.old.mechanical_screen.shared.owned_RSS(child);vram=owned_VRAM(child.pid,uuid);row['peak_host_RSS_bytes']=max(row.get('peak_host_RSS_bytes',0),host);row['peak_owned_VRAM_bytes']=max(row.get('peak_owned_VRAM_bytes',0),vram)
                    if host>=RESOURCES['aggregate_host_RSS_bytes']or vram>=RESOURCES['owned_VRAM_bytes']:raise MemoryError('Owned host/VRAM cap')
                    if time.monotonic()-begin>120 or time.monotonic()-start>420:raise TimeoutError('Owned sequential120s/420s GPU parity cap')
                    guard('controller');time.sleep(.2)
                row.update(exit_code=child.wait(),reaped=True,cleanup_exit_code=old.old.mechanical_screen.shared.old.cost.cleanup(child));path=LOCAL/f'worker-{slot}/result-LOCAL.json'
                if path.exists():row['result']=record(path)
                if row['exit_code']!=0 or row['cleanup_exit_code']!=0 or not path.exists()or read(path).get('qualified')is not True:raise RuntimeError('GPU parity unresolved; all other planned devices remain pending')
                row.update(status='qualified',wall_s=time.monotonic()-begin);checkpoint();child=None
        state.update(completed=True,qualified=True,all3_devices_retained=True,all8_ranks_pending=True)
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error))
    finally:
        if child is not None:
            state['devices'][slot].update(cleanup_exit_code=old.old.mechanical_screen.shared.old.cost.cleanup(child),reaped=True,status='unresolved')
            path=LOCAL/f'worker-{slot}/result-LOCAL.json'
            if path.exists():state['devices'][slot]['result']=record(path)
        for s,h in previous.items():signal.signal(s,h)
        state['terminal']=True;checkpoint()
    return state

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker','run','worker-guard']);ap.add_argument('--sha');ap.add_argument('--slot',type=int);a=ap.parse_args()
    if a.action=='worker-guard':guard('worker',GPU.ALLOWED_UUIDS[a.slot]);print(json.dumps(dict(passed=True,no_models_or_CUDA=True)))
    else:
        result=prepare()if a.action=='prepare'else(worker(a.slot,a.sha)if a.action=='worker'else run(a.sha));print(json.dumps(result))
        if a.action=='run'and not result['completed']:raise SystemExit(1)
