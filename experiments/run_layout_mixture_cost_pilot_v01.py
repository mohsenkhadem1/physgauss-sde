"""One deterministic DEVELOPMENT case; no random draws or future truth scores.

All3 parameter-grid levels/all8 actions are a frozen worklist. An unfinished
120s externally supervised pilot remains a retained negative, never a filtered
or smaller successful sweep. No actual prepare/run is performed on import.
"""
import argparse,ast,hashlib,json,os,resource,signal,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from experiments import campaign_active_fixture_bank_v01 as mechanics
from experiments import campaign_layout_mixture_decision_v01 as decision
from experiments import run_active_fixture_mechanics_screen_v01 as mechanical_screen
ROOT=Path(__file__).resolve().parents[1];LOCAL=ROOT/'runs/operator-dev/layout-mixture-cost-pilot-v01'
SOURCE='experiments/run_layout_mixture_cost_pilot_v01.py';TEST='tests/test_layout_mixture_cost_pilot_v01.py'
ORDERS=(9,17,33);OUTCOME_ORDERS=(8,16)
RESOURCES=dict(CPU=17,threads=1,GPU=False,RSS_bytes=4*1024**3,external_wall_s=120,internal_wall_s=110,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=64*1024**2,latent_chunk=4096,outcome_chunk=16,maximum_outcome_nodes=200000)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');os.replace(tmp,path)
def sources():
    pending=[SOURCE,TEST,*mechanical_screen.sources()];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names]
            if isinstance(node,ast.ImportFrom)and node.level==0:modules=[node.module or '']+[(node.module+'.'+a.name)for a in node.names if node.module]
            for mod in modules:
                p=mod.replace('.','/')+'.py'
                if(ROOT/p).is_file()and p not in seen:pending.append(p)
    return {name:sha(ROOT/name)for name in sorted(seen)}
def labels():
    specs=mechanics.measurement_specs();return {role:tuple(s['id']for s in specs if s['role']==role)for role in('prefix','candidate','target')}
def positive_grid(order):
    if order not in ORDERS:raise ValueError('Frozen9/17/33 positive unit-prior Gauss grid')
    x,w=np.polynomial.legendre.leggauss(order);a,b=np.meshgrid((x+1)/2,(x+1)/2,indexing='ij');weights=np.outer(w/2,w/2).ravel()
    return np.column_stack((a.ravel(),b.ravel())),weights

def prefix_only(predict_prefix):
    """The sole deterministic generating-case interface: four observed channels."""
    y=np.asarray(predict_prefix(np.array([[.5,.5]])),float)
    if y.shape!=(1,4,1)or not np.isfinite(y).all():raise ValueError('Only canonical n2 linear prefix4; no added noise')
    return y[0,:,0].copy()

def posterior_bank(models,order,observation):
    """Policy receives prefix data only, never generating layout/parameters."""
    names=labels();y=np.asarray(observation,float)
    if len(models)!=4 or y.shape!=(4,)or not np.isfinite(y).all():raise ValueError('All four layouts and same finite prefix4')
    unit,prior=positive_grid(order);log_evidence=[];conditional=[];candidates=[];targets=[]
    all_labels=names['prefix']+names['candidate']+names['target']
    for model in models:
        pred=np.asarray(model.select(all_labels)(unit),float)
        if pred.shape!=(len(unit),60,1)or not np.isfinite(pred).all():raise ValueError('Exact60channel hypothetical response map')
        ll=-.5*np.sum(((pred[:,:4,0]-y)/mechanics.SIGMA_M)**2,axis=1)-4*np.log(mechanics.SIGMA_M*np.sqrt(2*np.pi));logs=np.log(prior)+ll;z=logsumexp(logs)
        log_evidence.append(float(z));conditional.append(np.exp(logs-z));candidates.append(pred[:,4:12,0]);targets.append(pred[:,12:,0])
    bank=decision.combine_layouts(log_evidence,[.25]*4,conditional,candidates,targets)
    return bank,dict(layout_probability=decision.layout_probabilities(log_evidence,[.25]*4).tolist(),log_evidence=log_evidence,hypotheses=len(bank.probability),parameter_order=order)

def action_record(bank,action):
    levels=[decision.expected_risk(bank,action,mechanics.SIGMA_M,order=n,latent_chunk=RESOURCES['latent_chunk'],outcome_chunk=RESOURCES['outcome_chunk'],maximum_outcome_nodes=RESOURCES['maximum_outcome_nodes'])for n in OUTCOME_ORDERS]
    prior=levels[-1]['prior_target_variance'];error=abs(levels[-1]['expected_target_variance']-levels[-2]['expected_target_variance'])+levels[-1]['omitted_risk_bound']
    return dict(action=action,levels=levels,successive_error=error,qualified=bool(error<=1e-4*prior if prior>0 else error==0))

def level_summary(bank,metadata,actions):
    if len(actions)!=8 or {a['action']for a in actions}!=set(range(8)):raise ValueError('All8 actions retained before any ranking')
    risks=np.array([a['levels'][-1]['expected_target_variance']for a in actions]);errors=np.array([a['successive_error']for a in actions]);best=int(np.argmin(risks));ties=np.flatnonzero(risks-risks[best]<=errors+errors[best]).tolist();parts=decision.variance_decomposition(bank)
    return dict(**metadata,outcome_qualified=all(a['qualified']for a in actions),expected_target_risk=risks.tolist(),risk_error_witness=errors.tolist(),unique_best_action=best if len(ties)==1 and all(a['qualified']for a in actions)else None,ties=ties,catalog_order_tie_selection=min(ties)if all(a['qualified']for a in actions)else None,fixed_action_regret=(risks-risks[best]).tolist(),uniform_random_expected_regret=float(np.mean(risks)-risks[best]),between_layout_variance_fraction=parts['between_layout']/parts['total']if parts['total']>0 else 0.,total_target_variance=parts['total'],maximum_layout_probability=max(metadata['layout_probability']),scope='One deterministic finite linear DEVELOPMENT posterior; regret is predicted Bayes risk, not observed prediction error or adaptive test advantage.')

def parameter_witness(coarse,fine):
    scale=max(coarse['total_target_variance'],fine['total_target_variance']);error=float(np.max(abs(np.asarray(coarse['expected_target_risk'])-fine['expected_target_risk'])));tv=float(np.sum(abs(np.asarray(coarse['layout_probability'])-fine['layout_probability']))/2)
    return dict(coarse=coarse['parameter_order'],fine=fine['parameter_order'],maximum_risk_difference=error,relative_risk_gate=1e-3,layout_probability_TV=tv,layout_TV_gate=.01,passed=bool(error<=1e-3*scale and tv<=.01),scope='Finite-grid development convergence witness only, not continuous posterior qualification.')

def guard():
    if set(os.sched_getaffinity(0))!={17}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('CPU17/one-thread/noGPU only')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=RESOURCES['RSS_bytes']:raise MemoryError('Owned4GiB RSS')
    available=int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024
    if available<RESOURCES['RAM_reserve_bytes']:raise MemoryError('30GiB available RAM reserve')
    import shutil
    if shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise MemoryError('30GiB disk reserve')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise MemoryError('64MiB local writes')

def prepare(mechanics_sha):
    """Metadata-only semantic admission once, before any prefix/model calls."""
    admission=mechanical_screen.completed_screen_gate(mechanics_sha)
    if admission.get('all_numeric_checks_passed')is not True or admission.get('development_only')is not True:raise ValueError('Closed mechanical development screen required')
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No implicit pilot refreeze')
    p=dict(schema='layout-mixture-cost-pilot-v01',source_SHA256=sources(),mechanics_admission=admission,resources=RESOURCES,parameter_orders=list(ORDERS),outcome_orders=list(OUTCOME_ORDERS),worklist=[dict(order=n,action=a)for n in ORDERS for a in range(8)],mesh_n=2,development_case=dict(layout_id=mechanics.LAYOUTS[0]['id'],E_Pa=4000.,K_N_m=float(np.sqrt(150)),unit=[.5,.5],prefix_noise='Exactly zero, deterministic development only'),label_roles=labels(),all_categories_retained=True,no_RNG=True,no_nonlinear_PDE_solves=True,no_generating_target_evaluation=True,external_watchdog_required=True)
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')

def verify_mechanics_receipt(value):
    if isinstance(value,dict):
        if 'path'in value and 'sha256'in value and record(ROOT/value['path'])!=value:raise ValueError('Changed mechanical input commitment')
        for v in value.values():verify_mechanics_receipt(v)
    elif isinstance(value,list):
        for v in value:verify_mechanics_receipt(v)


def validate(digest):
    p=read(LOCAL/'protocol.json')
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=RESOURCES or p['worklist']!=[dict(order=n,action=a)for n in ORDERS for a in range(8)]or p['label_roles']!={k:list(v)for k,v in labels().items()}:raise ValueError('Frozen pilot sources/config/worklist')
    verify_mechanics_receipt(p['mechanics_admission'])
    e=read(ROOT/p['mechanics_admission']['execution']['path'])
    for pid in(e['controller_pid'],e['worker_pid']):mechanical_screen.shared.old.absent(pid)
    r=read(LOCAL/'release.json')
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('mechanics_admission')!=p['mechanics_admission']or r.get('external_watchdog_120s_4GiB')is not True:raise ValueError('Exact root release with external bounds')
    return p

def worker(digest):
    p=validate(digest);guard()
    if(LOCAL/'execution.json').exists():raise FileExistsError('No pilot retry/overwrite')
    state=dict(completed=False,qualified=False,pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),all_worklist=p['worklist'],completed_actions=[],levels=[],new_RNG_calls=0,nonlinear_physics_solves=0,finite_element_modal_builds=0,generating_future_truth_evaluations=0);start=time.monotonic()
    def checkpoint():state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'execution.json',state)
    def timeout(*_):raise TimeoutError('110s internal stop; retain complete fixed worklist and negative')
    def stop(*_):raise InterruptedError('Explicit external pilot stop; preserve pending fixed worklist')
    previous={signal.SIGALRM:signal.signal(signal.SIGALRM,timeout),signal.SIGTERM:signal.signal(signal.SIGTERM,stop),signal.SIGINT:signal.signal(signal.SIGINT,stop)};signal.alarm(RESOURCES['internal_wall_s'])
    try:
        loads,_=mechanics.load_catalog(2);models=[]
        for layout in mechanics.LAYOUTS:
            f=mechanics.make_forward(2,layout['id']);models.append(mechanics.make_measurement_modal(f,loads));state['finite_element_modal_builds']=len(models);guard();checkpoint()
        state['model_construction_s']=time.monotonic()-start
        observation=prefix_only(models[0].select(labels()['prefix']));np.savez_compressed(LOCAL/'prefix-observation-LOCAL.npz',observed_prefix=observation);state['prefix_observation']=record(LOCAL/'prefix-observation-LOCAL.npz');checkpoint()
        for order in ORDERS:
            begin=time.monotonic();bank,metadata=posterior_bank(models,order,observation);guard();actions=[]
            for action in range(8):
                r=action_record(bank,action);guard();actions.append(r);path=LOCAL/f'order-{order}-action-{action}-LOCAL.json';write(path,r);state['completed_actions'].append(dict(order=order,action=action,result=record(path)));checkpoint()
            summary=level_summary(bank,metadata,actions);summary['seconds']=time.monotonic()-begin;write(LOCAL/f'order-{order}-summary-LOCAL.json',summary);state['levels'].append(summary);checkpoint()
        witnesses=[parameter_witness(a,b)for a,b in zip(state['levels'],state['levels'][1:])];state.update(completed=True,parameter_refinements=witnesses,qualified=bool(state['levels'][-1]['outcome_qualified']and witnesses[-1]['passed']),no_continuous_inference_or_physical_qualification_claim=True)
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        signal.alarm(0)
        for sig,handler in previous.items():signal.signal(sig,handler)
        checkpoint()
    return state

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker','worker-guard']);ap.add_argument('--mechanics-sha');ap.add_argument('--sha');args=ap.parse_args()
    if args.action=='worker-guard':guard();print(json.dumps(dict(guard_passed=True,no_models_or_data=True)))
    else:print(json.dumps(prepare(args.mechanics_sha)if args.action=='prepare'else worker(args.sha)))
