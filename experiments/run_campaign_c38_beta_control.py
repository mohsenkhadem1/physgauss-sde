"""Fixed learned-density control on the validated homogeneous log-NH family.

No broad block/RGBD/dynamic inference claim. Each stage is explicit and hash
bound; test labels are not opened by fitting or DEV/CAL validation. Raw datasets
and model checkpoints remain local. No checkpoint or architecture selection.
"""
from __future__ import annotations
import argparse,hashlib,json,os,resource,signal,time,traceback,shutil
from pathlib import Path
import numpy as np
import torch
from experiments.campaign_c38_beta_data import simulate_objects,CONTEXTS
from experiments.campaign_c38_beta_posterior import BetaPosterior,joint_log_prob,numpy_parameters,marginal_cdf,field_moments,mixture_quadrature
from experiments.campaign_c38_beta_metrics import parameter_diagnostics,reference_joint_kl
from experiments.campaign_c38_bar_posterior_z import BarPosteriorBatch
from experiments.campaign_c38_bar_reference import LOWER,UPPER,material_quadrature,prior_physical,oracle_positions,endpoint_displacements,SIGMA_M,TARGET_FORCE
from experiments.campaign_c38_actions import select_reset_actions
from experiments.campaign_c38_study import paired_object_interval

ROOT=Path(__file__).resolve().parents[1]
TEACHER='runs/gaussian-campaign-v01/c38-bar-active-reference-v04/summary.json'
TEACHER_SHA='77091e3889f75aa908c36173c9e39380099cc7d6a2e42c42d576179e39903894'
ROLE_COUNTS={'TRAIN':4096,'DEV':256,'CAL':128,'TEST':128}
DATA_SEEDS={'TRAIN':2026122400,'DEV':2026122401,'CAL':2026122402,'TEST':2026122403}
FIT_SEEDS=(2026122410,2026122411,2026122412)
GATES={'reference_relative_Z':5e-5,'reference_CDF':5e-4,'reference_field_mean_m':1e-7,'reference_field_std_m':2e-7,'reference_cloud_Z':5e-5,'KL_order_difference_nat':.005,'KL_mean_nat':.10,'KL_max_nat':.50,'CDF_mean_sup_grid_error':.05,'CDF_max_sup_grid_error':.20,'field_mean_m':20e-6,'field_std_m':20e-6,'Beta_12_24_field_moment_m':.2e-6}
NEW_SOURCES=['experiments/run_campaign_c38_beta_control.py','experiments/campaign_c38_beta_posterior.py','experiments/campaign_c38_beta_data.py','experiments/campaign_c38_beta_metrics.py']
# Explicit full physical dependency closure, inherited from the qualified
# teacher. Adding a new import requires changing the prospective protocol.
PHYSICAL_SOURCES=['experiments/campaign_c38_bar_reference.py','experiments/campaign_c38_bar_posterior_z.py','experiments/campaign_nonlinear_bar_inverse.py','experiments/campaign_static_uniaxial_bar.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/campaign_sparse_nonlinear_gaussian.py','experiments/boundary_block.py','experiments/diagnose_nonlinear_physical_probes.py','experiments/campaign_c38_actions.py','experiments/campaign_c38_study.py','experiments/campaign_point_active_views.py','experiments/censored_range_observation.py','experiments/campaign_boundary_gaussians.py','experiments/campaign_compliant_gaussian_linear.py','experiments/campaign_linear_fem.py','experiments/campaign_material_gaussians.py','experiments/campaign_nonlinear_excitation.py','experiments/campaign_nonlinear_fem.py','experiments/campaign_nonlinear_gaussian_reference.py','experiments/validate_isaac_dataset.py']
SOURCES=NEW_SOURCES+PHYSICAL_SOURCES

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return x.item()
    return x
def write(path,x):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(safe(x),indent=2,allow_nan=False)+'\n');tmp.replace(path)
def rel(path):
    p=Path(path).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Repository-local artifacts required')
    return str(p.relative_to(ROOT))
def record(path):return {'path':rel(path),'sha256':sha(path)}
def checked(rec):
    if not isinstance(rec,dict) or set(rec)!={'path','sha256'}:raise ValueError('Exact artifact path/digest record required')
    p=(ROOT/rec['path']).resolve()
    if not p.is_relative_to(ROOT) or sha(p)!=rec['sha256']:raise ValueError('Changed artifact: '+rec['path'])
    return p
def load_record(rec):return json.loads(checked(rec).read_text())
def fresh(path):
    p=Path(path)
    if p.exists():raise FileExistsError('Fresh output only; failed attempts are retained')
    p.mkdir(parents=True);return p
def threads():
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='2':raise ValueError('Explicit CPU2 numerical thread limits required')
    torch.set_num_threads(2)

def guard_resources(p,*,gpu=False):
    limits=p['resources'];rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;disk=shutil.disk_usage(ROOT).free
    if rss>limits['process_RSS_cap_bytes']:raise MemoryError('Frozen12GiB process RSS cap')
    if disk<limits['disk_free_reserve_bytes']:raise OSError('Frozen30GiB free disk reserve')
    allocated=0
    if gpu:
        allocated=torch.cuda.max_memory_allocated()
        if allocated>limits['GPU_allocated_cap_bytes']:raise MemoryError('Frozen8GiB GPU allocator cap')
    return dict(peak_process_RSS_bytes=rss,disk_free_bytes=disk,peak_GPU_allocated_bytes=allocated)

def prepare(path,output,gpu_uuid):
    if Path(path).exists() or Path(output).exists():raise FileExistsError('Fresh protocol and output required')
    if not gpu_uuid.startswith('GPU-'):raise ValueError('Physical GPU2 UUID must be supplied and frozen')
    teacher=ROOT/TEACHER
    if sha(teacher)!=TEACHER_SHA:raise ValueError('Qualified teacher identity changed')
    t=json.loads(teacher.read_text())
    if t.get('completed') is not True or t.get('passed') is not True:raise ValueError('All teacher numerical gates required')
    p=dict(schema='c38-beta-density-control-v01',status='FROZEN_BEFORE_FRESH_ROLE_DRAWS',output=rel(output),source_sha256={n:sha(ROOT/n) for n in SOURCES},teacher=record(teacher),roles=ROLE_COUNTS,data_seeds=DATA_SEEDS,fit_seeds=FIT_SEEDS,contexts=CONTEXTS,gates=GATES,
        physical_scope='Known homogeneous logarithmic NH prism, known uniform spring location/law, ideal endpoint displacement sensor; not RGBD/general BC/dynamic forecasting',prior='Independent uniform unit(log E[4500,9000]Pa, log K[100,400]N/m)',sensor_sigma_m=.0002,prefix_force_N=1.,poke_force_N=2.,target_force_N=1.5,target_scope='Static force interpolation from common reset state',acquisition_budget='One frame/one measured scalar, nominal0.2s equal-cost ACCOUNTING ONLY; dynamic settling feasibility not demonstrated',
        architecture=dict(components=16,hidden=[64,64],activation='SiLU',output='Product-Beta mixture; sigmoid means and exp(log concentration[-6,12]), positive .001 floors',input='Only sorted acquired known forces/sides/displacements/masks,10 features; no theta/role/ID/targets'),
        fit=dict(steps_per_seed=8000,batch=256,learning_rate=.001,gradient_clip_norm=5.,optimizer='Adam',checkpoint='FINAL ONLY, no DEV selection',wall_cap_seconds_total=600,cpu_threads=2,physical_gpu_uuid=gpu_uuid,max_jobs_on_gpu=1,device='cuda:0 under exactly frozen physical UUID CVD'),resources=dict(GPU_allocated_cap_bytes=8*1024**3,process_RSS_cap_bytes=12*1024**3,disk_free_reserve_bytes=30*1024**3),
        validation=dict(roles=['DEV','CAL'],reference_object_indices=list(range(16)),reference_context_assignment='object index modulo8: exactly2 percontext perrole',reference_lower=dict(rtol=1e-6,atol=1e-12,conditional_order=64),reference_higher=dict(rtol=1e-8,atol=1e-13,conditional_order=128),CDF_grid=np.linspace(.05,.95,19),KL_cloud_orders=[[16,64],[32,128]],Beta_moment_orders=[12,24],PIT='Percontext independent-object KS with simultaneous DKW alpha.05/48; exact90%coverage CP interval alpha.05/48 must include.9; not a calibration proof',reference_wall_cap_seconds=1800,eval_wall_cap_seconds=1800),
        test=dict(objects=128,closed_until_all3_DEV_CAL_pass=True,prefix_only_file=True,choice_freeze_before_counterfactual_outcomes=True,posterior_freeze_before_labels_and_targets=True,reference_object_indices=list(range(16)),no_best_seed_selection=True,cluster_bootstrap_unit='Whole object across all8contexts',bootstrap_draws=4000,seed=2026122420),
        policy=dict(policies=['fixed_left','fixed_right','random','opposite_endpoint','information_gain','predictive_risk'],one_action=True,independent_probe_RESET=True,common_target_all_policies=True,common_random_numbers=True,fantasies=64,quadrature_order=12,choice_seed=2026122421,bootstrap_seed=2026122422,scope='Working neural-prefix posterior utility; finite fantasy ranking uncertainty; two-observation map coherence must be reported'),raw_local=True,no_architecture_or_hyperparameter_search=True,preserve_all_failures=True)
    Path(path).parent.mkdir(parents=True,exist_ok=True);write(path,p);return sha(path)

def validate(path,digest):
    if sha(path)!=digest:raise ValueError('Protocol changed')
    p=json.loads(Path(path).read_text())
    if p.get('schema')!='c38-beta-density-control-v01' or p.get('status')!='FROZEN_BEFORE_FRESH_ROLE_DRAWS' or p.get('roles')!=ROLE_COUNTS or p.get('data_seeds')!=DATA_SEEDS or p.get('fit_seeds')!=list(FIT_SEEDS) or p.get('contexts')!=list(CONTEXTS) or p.get('gates')!=GATES or set(p.get('source_sha256',{}))!=set(SOURCES):raise ValueError('Exact prospective design required')
    for n,d in p['source_sha256'].items():
        if sha(ROOT/n)!=d:raise ValueError('Frozen source changed: '+n)
    checked(p['teacher']);threads();guard_resources(p);return p

def save_npz(path,values):
    np.savez(path,**values);return record(path)

def generate(p,protocol_record):
    out=fresh(ROOT/p['output']);data=out/'data';data.mkdir();manifest=dict(completed=False,protocol=protocol_record,roles={},raw_local=True)
    write(out/'dataset.json',manifest)
    try:
        for role,count in ROLE_COUNTS.items():
            guard_resources(p)
            public,private,prefix=simulate_objects(count,DATA_SEEDS[role]);directory=data/role;directory.mkdir()
            manifest['roles'][role]=dict(objects=count,public=save_npz(directory/'observations.npz',public),private=save_npz(directory/'labels-and-targets.npz',private),prefix_only=save_npz(directory/'prefix-only.npz',prefix))
            write(out/'dataset.json',manifest)
        manifest['completed']=True;write(out/'dataset.json',manifest)
    except BaseException as e:
        manifest['failure']=dict(type=type(e).__name__,message=str(e));write(out/'dataset.json',manifest);raise
    return record(out/'dataset.json')

def dataset(p,protocol_record):
    m=json.loads((ROOT/p['output']/'dataset.json').read_text())
    if m.get('completed') is not True or m.get('protocol')!=protocol_record or set(m.get('roles',{}))!=set(ROLE_COUNTS):raise ValueError('Complete exact object-role dataset required')
    for role,n in ROLE_COUNTS.items():
        if m['roles'][role]['objects']!=n:raise ValueError('Role object count changed')
        for k in ('public','private','prefix_only'):checked(m['roles'][role][k])
    return m

def load_arrays(rec):
    with np.load(checked(rec),allow_pickle=False) as a:return {k:a[k] for k in a.files}

def train(p,protocol_record):
    m=dataset(p,protocol_record);out=fresh(ROOT/p['output']/'fits');f=p['fit']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=f['physical_gpu_uuid'] or not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Exactly frozen physical GPU2 UUID visibility required')
    torch.cuda.set_per_process_memory_fraction(min(1.,p['resources']['GPU_allocated_cap_bytes']/torch.cuda.get_device_properties(0).total_memory),device=0);guard_resources(p,gpu=True)
    # Only TRAIN labels/outcomes are materialized. DEV/CAL/TEST remain unopened.
    public=load_arrays(m['roles']['TRAIN']['public']);private=load_arrays(m['roles']['TRAIN']['private']);x=torch.as_tensor(public['features'].reshape(-1,10),dtype=torch.float32,device='cuda:0');y=torch.as_tensor(np.repeat(private['unit'],8,axis=0),dtype=torch.float64,device='cuda:0')
    ledger=dict(completed=False,protocol=protocol_record,dataset=record(ROOT/p['output']/'dataset.json'),seeds={},pid=os.getpid(),raw_local=True);write(out/'execution.json',ledger);started=time.monotonic()
    def expired(*_):raise TimeoutError('Frozen total600s fit budget; no automatic retry')
    old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,f['wall_cap_seconds_total'])
    try:
        for seed in FIT_SEEDS:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=BetaPosterior().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=f['learning_rate']);rng=torch.Generator(device='cuda:0').manual_seed(seed+1000);history=[];start=time.monotonic()
            for step in range(1,f['steps_per_seed']+1):
                ids=torch.randint(len(x),(f['batch'],),generator=rng,device='cuda:0');optimizer.zero_grad(set_to_none=True);loss=-joint_log_prob(y[ids],model(x[ids])).mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),f['gradient_clip_norm'],error_if_nonfinite=True);optimizer.step()
                if step==1 or step%1000==0:
                    guard_resources(p,gpu=True)
                    history.append(dict(step=step,training_batch_NLL_nat=float(loss.detach()),gradient_norm=float(norm)));write(out/f'seed-{seed}-history.json',history)
            torch.cuda.synchronize();checkpoint=out/f'seed-{seed}-final.pt';torch.save({'model':model.cpu().state_dict(),'seed':seed,'step':f['steps_per_seed'],'protocol':protocol_record,'dataset':ledger['dataset']},checkpoint)
            ledger['seeds'][str(seed)]=dict(completed=True,final_checkpoint=record(checkpoint),seconds=time.monotonic()-start,steps=f['steps_per_seed'],resources=guard_resources(p,gpu=True));write(out/'execution.json',ledger);del model,optimizer;torch.cuda.empty_cache()
        ledger.update(completed=True,seconds=time.monotonic()-started,resources=guard_resources(p,gpu=True));write(out/'execution.json',ledger)
    except BaseException as e:
        ledger.update(seconds=time.monotonic()-started,failure=dict(type=type(e).__name__,message=str(e),seed=locals().get('seed'),last_accepted_step=locals().get('step',0)-1));write(out/'execution.json',ledger);raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)

def fitted(p,protocol_record):
    path=ROOT/p['output']/'fits/execution.json';fit=json.loads(path.read_text())
    if fit.get('completed') is not True or fit.get('protocol')!=protocol_record or set(fit.get('seeds',{}))!=set(map(str,FIT_SEEDS)):raise ValueError('All3 final fits required; no dropped seed')
    if fit['dataset']!=record(ROOT/p['output']/'dataset.json'):raise ValueError('Dataset changed after fitting')
    for r in fit['seeds'].values():checked(r['final_checkpoint'])
    return fit

def infer(checkpoint,features):
    content=torch.load(checked(checkpoint),map_location='cpu',weights_only=True);model=BetaPosterior();model.load_state_dict(content['model']);model.eval()
    with torch.no_grad():
        batches=[model(torch.as_tensor(features[i:i+256],dtype=torch.float32)) for i in range(0,len(features),256)]
    return {k:torch.cat([b[k] for b in batches]).double().numpy() for k in ('logits','alpha','beta')}

def slice_parameters(p,ids):return {k:np.asarray(v)[ids] for k,v in p.items()}

def held_density_score(parameters,unit):
    truth=np.repeat(np.asarray(unit),8,axis=0)
    with torch.no_grad():value=joint_log_prob(torch.as_tensor(truth),{k:torch.as_tensor(v) for k,v in parameters.items()}).numpy()
    return dict(mean_negative_log_density_nat=float(-value.mean()),context_mean_negative_log_density_nat=-value.reshape(len(unit),8).mean(0),density_measure='Unit logE/logK; unconditioned uniform-prior NLL is exactly0',cluster_unit='Object across8 dependent contexts')

def numerical_reference(public,context,indices,p):
    mask=public['mask'][0,context];forces=public['forces'][0,context,mask];sides=public['sides'][0,context,mask].astype(int);obs=public['observed_m'][indices,context][:,mask]
    cfg=p['validation'];low=BarPosteriorBatch(obs,forces,sides,**cfg['reference_lower']);high=BarPosteriorBatch(obs,forces,sides,**cfg['reference_higher']);X=material_quadrature();lm,ls=low.field_moments(X);hm,hs=high.field_moments(X)
    relative=float(np.max(abs(np.expm1(np.log(low.Z)-low.shift-np.log(high.Z)+high.shift))));cdf=0.;table=[]
    for j,kind in enumerate(('E','K')):
        rows=[]
        for u in cfg['CDF_grid']:
            value=np.exp(LOWER[j]+u*(UPPER[j]-LOWER[j]));a=low.cdf(kind,value);b=high.cdf(kind,value);cdf=max(cdf,float(np.max(abs(a-b))));rows.append(b)
        table.append(np.asarray(rows).T)
    errors=dict(reference_relative_Z=relative,reference_CDF=cdf,reference_field_mean_m=float(np.max(abs(lm-hm))),reference_field_std_m=float(np.max(abs(ls-hs))))
    return high,dict(passed=all(v<=GATES[k] for k,v in errors.items()),errors=errors,field_mean=hm,field_std=hs,marginal_CDF_grid=np.stack(table,axis=1),adaptive_error_estimates=high.errors,cubature_errors_are_not_rigorous=True)

def validate_models(p,protocol_record):
    m=dataset(p,protocol_record);fit=fitted(p,protocol_record);out=fresh(ROOT/p['output']/'validation');ledger=dict(completed=False,protocol=protocol_record,fit=record(ROOT/p['output']/'fits/execution.json'),roles={},all_gates_passed=False,test_labels_opened=False);write(out/'summary.json',ledger)
    started=time.monotonic()
    def expired(*_):raise TimeoutError('Frozen DEV/CAL numerical evaluation budget')
    old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,p['validation']['eval_wall_cap_seconds'])
    try:
        for role in ('DEV','CAL'):
            public=load_arrays(m['roles'][role]['public']);private=load_arrays(m['roles'][role]['private']);B=len(private['unit']);ledger['roles'][role]={'seeds':{},'references':{}};params={}
            for seed in FIT_SEEDS:
                guard_resources(p)
                pars=infer(fit['seeds'][str(seed)]['final_checkpoint'],public['features'].reshape(-1,10));params[seed]=pars;save_npz(out/f'{role}-seed-{seed}-parameters.npz',pars)
                diagnostics,quant,pit=parameter_diagnostics(pars,private['unit']);qm,qs=field_moments(pars,order=12);hm,hs=field_moments(pars,order=24);error=max(float(np.max(abs(qm-hm))),float(np.max(abs(qs-hs))))
                np.savez(out/f'{role}-seed-{seed}-states.npz',mean_positions_m=hm,std_m=hs,quantiles_unit=quant,PIT=pit)
                ledger['roles'][role]['seeds'][str(seed)]=dict(parameter_diagnostics=diagnostics,proper_density_score=held_density_score(pars,private['unit']),Beta_quadrature_max_moment_difference_m=error,Beta_quadrature_passed=error<=GATES['Beta_12_24_field_moment_m'],reference_rows=[])
                write(out/'summary.json',ledger)
            for c in range(8):
                guard_resources(p)
                ids=np.array([i for i in p['validation']['reference_object_indices'] if i%8==c]);posterior,reference=numerical_reference(public,c,ids,p);write(out/f'{role}-reference-context-{c}.json',reference);ledger['roles'][role]['references'][str(c)]=dict(objects=ids,passed=reference['passed'],report=record(out/f'{role}-reference-context-{c}.json'))
                for seed in FIT_SEEDS:
                    pars=params[seed];subset=slice_parameters(pars,ids*8+c);nm,ns=field_moments(subset,order=24);field_error=dict(mean_m=float(np.max(abs(nm-reference['field_mean']))),std_m=float(np.max(abs(ns-reference['field_std']))));grid=np.asarray(p['validation']['CDF_grid']);table=np.empty((len(ids),2,len(grid)))
                    for j,u in enumerate(grid):table[:,:,j]=marginal_cdf(subset,np.full((len(ids),2),u))
                    cdferror=np.max(abs(table-reference['marginal_CDF_grid']),axis=2)
                    for k,i in enumerate(ids):
                        kl=[];cloud=[]
                        for o,z in p['validation']['KL_cloud_orders']:
                            value,quality=reference_joint_kl(posterior,k,subset,k,outer_order=o,conditional_order=z);kl.append(value);cloud.append(quality)
                        row=dict(object_index=int(i),context=c,joint_KL_nat=kl[-1],KL_two_order_difference_nat=abs(kl[-1]-kl[0]),cloud_normalizer_relative_error=max(x['normalizer_relative_error'] for x in cloud),CDF_sup_grid_error_by_latent=cdferror[k],field_mean_max_error_m=field_error['mean_m'],field_std_max_error_m=field_error['std_m']);ledger['roles'][role]['seeds'][str(seed)]['reference_rows'].append(row)
                write(out/'summary.json',ledger)
            for seed in FIT_SEEDS:
                r=ledger['roles'][role]['seeds'][str(seed)];rows=r['reference_rows'];K=np.array([x['joint_KL_nat'] for x in rows]);C=np.array([x['CDF_sup_grid_error_by_latent'] for x in rows]);quality=all(x['KL_two_order_difference_nat']<=GATES['KL_order_difference_nat'] and x['cloud_normalizer_relative_error']<=GATES['reference_cloud_Z'] and x['joint_KL_nat']>=-GATES['KL_order_difference_nat'] for x in rows)
                r['reference_summary']=dict(cases=len(rows),joint_KL_mean_nat=float(K.mean()),joint_KL_max_nat=float(K.max()),CDF_mean_sup_grid_error_by_latent=C.mean(0),CDF_max_sup_grid_error_by_latent=C.max(0),reference_numerical_quality_passed=quality)
                r['passed']=bool(r['Beta_quadrature_passed'] and quality and len(rows)==16 and all(x['passed'] for x in ledger['roles'][role]['references'].values()) and K.mean()<=GATES['KL_mean_nat'] and K.max()<=GATES['KL_max_nat'] and (C.mean(0)<=GATES['CDF_mean_sup_grid_error']).all() and C.max()<=GATES['CDF_max_sup_grid_error'] and max(x['field_mean_max_error_m'] for x in rows)<=GATES['field_mean_m'] and max(x['field_std_max_error_m'] for x in rows)<=GATES['field_std_m'] and (role!='CAL' or r['parameter_diagnostics']['calibration_gate_passed']))
            write(out/'summary.json',ledger)
        ledger.update(completed=True,seconds=time.monotonic()-started,all_gates_passed=all(r['passed'] for role in ledger['roles'].values() for r in role['seeds'].values()));write(out/'summary.json',ledger)
        if ledger['all_gates_passed']:write(out/'test-release.json',dict(status='DEV_CAL_ALL3_PASS_TEST_PREFIX_RELEASED',protocol=protocol_record,validation=record(out/'summary.json'),test_label_release_requires_choice_and_posterior_freezes=True))
    except BaseException as e:
        ledger.update(seconds=time.monotonic()-started,failure=dict(type=type(e).__name__,message=str(e)));write(out/'summary.json',ledger);raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)

def test_release(p,protocol_record):
    directory=ROOT/p['output']/'validation';r=json.loads((directory/'test-release.json').read_text());v=load_record(r['validation'])
    if r.get('status')!='DEV_CAL_ALL3_PASS_TEST_PREFIX_RELEASED' or r.get('protocol')!=protocol_record or v.get('completed') is not True or v.get('all_gates_passed') is not True or v.get('protocol')!=protocol_record or set(v.get('roles',{}))!={'DEV','CAL'}:raise ValueError('Complete exact DEV/CAL release required before any TEST episode')
    for role in ('DEV','CAL'):
        if set(v['roles'][role]['seeds'])!=set(map(str,FIT_SEEDS)) or not all(s.get('passed') is True for s in v['roles'][role]['seeds'].values()):raise ValueError('All fixed seeds/roles must pass')
    fitted(p,protocol_record);return record(directory/'test-release.json')

def acquisition_records(protocol_record):
    def h(x):return hashlib.sha256(x.encode()).hexdigest()
    return [dict(id=name,kind=kind,duration_s=.2,frame_count=1,sensor_dimension=1,reset_to_common_rest=True,common_state_sha256=h('common-homogeneous-rest'),target_history_sha256=h('same-static-force1.5N'),force_domain_sha256=protocol_record['sha256']) for name,kind in [('left_F1','view'),('right_F1','view'),('F2_at_prefix_camera','reset_probe')]]

def choose_test(p,protocol_record):
    release=test_release(p,protocol_record);m=dataset(p,protocol_record);fit=fitted(p,protocol_record);out=fresh(ROOT/p['output']/'test-choices')
    # Only THIS archive is opened. No action observations, labels or target.
    prefix=load_arrays(m['roles']['TEST']['prefix_only']);B=ROLE_COUNTS['TEST'];X=material_quadrature();actions=acquisition_records(protocol_record);ledger=dict(completed=False,protocol=protocol_record,release=release,prefix_only=m['roles']['TEST']['prefix_only'],seeds={},action_outcomes_opened=False,target_labels_opened=False);write(out/'barrier.json',ledger)
    try:
        for seed in FIT_SEEDS:
            pars=infer(fit['seeds'][str(seed)]['final_checkpoint'],prefix['features']);save_npz(out/f'seed-{seed}-prefix-parameters.npz',pars);rows=[]
            for i in range(B):
                unit,w=mixture_quadrature(pars,i,order=p['policy']['quadrature_order']);theta=prior_physical(unit);target=oracle_positions(theta,TARGET_FORCE,X)-X;end=endpoint_displacements(theta,[1.,2.]);side=int(prefix['camera'][i]);predictions=np.stack([end[:,0,0],end[:,0,1],end[:,1,side]])[:,:,None]
                r=select_reset_actions(target.reshape(len(w),-1),predictions,np.array([[SIGMA_M**2]]),w,actions,object_index=i,fixed_index=0,fantasies=p['policy']['fantasies'],fantasy_seed=p['policy']['choice_seed']+i,random_seed=p['policy']['choice_seed']);r['policies'].pop('fixed');r['policies'].update(fixed_left=0,fixed_right=1,opposite_endpoint=1-side);r['prefix_camera']=side;rows.append(r)
            path=out/f'seed-{seed}-choices.json';write(path,dict(completed=True,protocol=protocol_record,objects=B,choices=rows,frozen_before_action_outcomes=True,frozen_before_test_labels_and_targets=True));ledger['seeds'][str(seed)]=record(path);write(out/'barrier.json',ledger)
        ledger['completed']=True;write(out/'barrier.json',ledger)
    except BaseException as e:
        ledger['failure']=dict(type=type(e).__name__,message=str(e));write(out/'barrier.json',ledger);raise

def checked_choices(p,protocol_record):
    rec=record(ROOT/p['output']/'test-choices/barrier.json');r=load_record(rec)
    if r.get('completed') is not True or r.get('protocol')!=protocol_record or set(r.get('seeds',{}))!=set(map(str,FIT_SEEDS)) or r.get('action_outcomes_opened') is not False or r.get('target_labels_opened') is not False:raise ValueError('All fixed-seed choices must freeze before outcomes or target labels')
    test_release(p,protocol_record)
    for seed,path in r['seeds'].items():
        v=load_record(path)
        if v.get('completed') is not True or v.get('protocol')!=protocol_record or v.get('objects')!=ROLE_COUNTS['TEST'] or len(v.get('choices',[]))!=ROLE_COUNTS['TEST'] or v.get('frozen_before_action_outcomes') is not True or v.get('frozen_before_test_labels_and_targets') is not True:raise ValueError('Incomplete outcome-independent choices')
    return r,rec

def freeze_test_posteriors(p,protocol_record):
    choices,choice_record=checked_choices(p,protocol_record);m=dataset(p,protocol_record);fit=fitted(p,protocol_record);out=fresh(ROOT/p['output']/'test-states');ledger=dict(completed=False,protocol=protocol_record,choices=choice_record,seeds={},target_labels_opened=False);write(out/'barrier.json',ledger)
    # The acquired counterfactual action frames are now admitted. Predictions
    # for ALL seeds/contexts freeze before this function ever opens labels.
    public=load_arrays(m['roles']['TEST']['public'])
    try:
        for seed in FIT_SEEDS:
            pars=infer(fit['seeds'][str(seed)]['final_checkpoint'],public['features'].reshape(-1,10));mean,std=field_moments(pars,order=24);path=out/f'seed-{seed}-states.npz';ledger['seeds'][str(seed)]=save_npz(path,{**pars,'mean_positions_m':mean,'std_m':std});write(out/'barrier.json',ledger)
        ledger['completed']=True;write(out/'barrier.json',ledger)
    except BaseException as e:
        ledger['failure']=dict(type=type(e).__name__,message=str(e));write(out/'barrier.json',ledger);raise

def checked_test_states(p,protocol_record):
    rec=record(ROOT/p['output']/'test-states/barrier.json');states=load_record(rec);_,choice=checked_choices(p,protocol_record)
    if states.get('completed') is not True or states.get('protocol')!=protocol_record or states.get('choices')!=choice or states.get('target_labels_opened') is not False or set(states.get('seeds',{}))!=set(map(str,FIT_SEEDS)):raise ValueError('All test posterior states must freeze before labels/targets')
    for path in states['seeds'].values():checked(path)
    return states,rec

def evaluate_test(p,protocol_record):
    states,state_record=checked_test_states(p,protocol_record);choices,_=checked_choices(p,protocol_record);m=dataset(p,protocol_record);out=fresh(ROOT/p['output']/'test-results');B=ROLE_COUNTS['TEST'];X=material_quadrature();ledger=dict(completed=False,protocol=protocol_record,states=state_record,objects=B,seeds={},scope='Fresh analytical known-family control, not RGBD/general BC/dynamic forecasting',all_seeds_reported=True,posteriors_and_choices_frozen_before_labels=True);write(out/'summary.json',ledger)
    private=load_arrays(m['roles']['TEST']['private']);public=load_arrays(m['roles']['TEST']['public']);prefix=load_arrays(m['roles']['TEST']['prefix_only']);truth=private['canonical_positions_m'];camera=prefix['camera'].astype(int);ids=np.arange(B)
    prior_parameters=dict(logits=np.zeros((1,16)),alpha=np.ones((1,16,2)),beta=np.ones((1,16,2)));prior_mean,prior_std=field_moments(prior_parameters,order=24)
    # Modelfree static and rigid-shift baselines have no label inputs.
    measured=public['observed_m'][ids,camera,0];rigid=np.broadcast_to(X,(B,*X.shape)).copy();rigid[:,:,0]+=TARGET_FORCE*measured[:,None]
    for name,value in [('static_rest',np.broadcast_to(X,truth.shape)),('prior_predictive_mean',np.broadcast_to(prior_mean,truth.shape)),('endpoint_rigid_translation',rigid)]:
        mse=np.mean((value-truth)**2,axis=(1,2));ledger.setdefault('baselines',{})[name]=dict(pooled_coordinate_RMSE_m=float(np.sqrt(mse.mean())),mean_object_coordinate_RMSE_m=float(np.sqrt(mse).mean()))
    try:
        for seed in FIT_SEEDS:
            arrays=load_arrays(states['seeds'][str(seed)]);pars={k:arrays[k] for k in ('logits','alpha','beta')};diagnostics,quant,pit=parameter_diagnostics(pars,private['unit']);means=arrays['mean_positions_m'].reshape(B,8,*X.shape);std=arrays['std_m'].reshape(B,8,*X.shape);mse=np.mean((means-truth[:,None])**2,axis=(2,3));policychoices=load_record(choices['seeds'][str(seed)])['choices'];policyMSE={'no_action':mse[ids,camera]};policies={}
            for name in p['policy']['policies']:
                action=np.array([r['policies'][name] for r in policychoices]);context=np.where(camera==0,2,5)+action;policyMSE[name]=mse[ids,context];policies[name]=dict(pooled_coordinate_RMSE_m=float(np.sqrt(policyMSE[name].mean())),mean_object_coordinate_RMSE_m=float(np.sqrt(policyMSE[name]).mean()),mean_latent_field_std_m=float(std[ids,context].mean()),choices=np.bincount(action,minlength=3))
            pairs={}
            for first in ('information_gain','predictive_risk'):
                for second in ('fixed_left','fixed_right','random','opposite_endpoint','no_action'):
                    pairs[first+'-minus-'+second]=paired_object_interval(policyMSE[first],policyMSE[second],seed=p['policy']['bootstrap_seed'],replicates=p['test']['bootstrap_draws'])
            ledger['seeds'][str(seed)]=dict(parameter_diagnostics=diagnostics,proper_density_score=held_density_score(pars,private['unit']),policies=policies,no_action=dict(pooled_coordinate_RMSE_m=float(np.sqrt(policyMSE['no_action'].mean()))),paired_object_MSE_intervals=pairs,context_pooled_coordinate_RMSE_m=np.sqrt(mse.mean(0)),pooled_context_MSE=float(mse.mean()),cluster_unit='Object, not8 dependent contexts',reference_rows=[])
            np.savez(out/f'seed-{seed}-metrics-local.npz',context_MSE_m2=mse,quantiles_unit=quant,PIT=pit,**{'MSE_'+k:v for k,v in policyMSE.items()});write(out/'summary.json',ledger)
        # Exactly16 TEST reference cases after all predictions have frozen.
        for c in range(8):
            ri=np.array([i for i in p['test']['reference_object_indices'] if i%8==c]);posterior,reference=numerical_reference(public,c,ri,p);write(out/f'reference-context-{c}.json',reference)
            for seed in FIT_SEEDS:
                arrays=load_arrays(states['seeds'][str(seed)]);pars=slice_parameters({k:arrays[k] for k in ('logits','alpha','beta')},ri*8+c)
                for k,i in enumerate(ri):
                    kl=[];cloud=[]
                    for o,z in p['validation']['KL_cloud_orders']:
                        v,q=reference_joint_kl(posterior,k,pars,k,outer_order=o,conditional_order=z);kl.append(v);cloud.append(q)
                    ledger['seeds'][str(seed)]['reference_rows'].append(dict(object_index=int(i),context=c,reference_numerical_passed=reference['passed'],joint_KL_nat=kl[-1],KL_order_difference_nat=abs(kl[0]-kl[1]),cloud_normalizer_relative_error=max(q['normalizer_relative_error'] for q in cloud),NN_field_mean_error_m=float(np.max(abs(arrays['mean_positions_m'][i*8+c]-reference['field_mean'][k])))))
            write(out/'summary.json',ledger)
        ledger['completed']=True;write(out/'summary.json',ledger)
    except BaseException as e:
        ledger['failure']=dict(type=type(e).__name__,message=str(e));write(out/'summary.json',ledger);raise

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','generate','train','validate','choose-test','freeze-test','evaluate-test']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--gpu-uuid');args=a.parse_args()
    if args.mode=='prepare':print(prepare(args.protocol,args.output,args.gpu_uuid));return
    p=validate(args.protocol,args.protocol_sha256);rec=record(args.protocol)
    if args.mode=='generate':generate(p,rec)
    elif args.mode=='train':train(p,rec)
    elif args.mode=='validate':validate_models(p,rec)
    elif args.mode=='choose-test':choose_test(p,rec)
    elif args.mode=='freeze-test':freeze_test_posteriors(p,rec)
    else:evaluate_test(p,rec)
if __name__=='__main__':main()
