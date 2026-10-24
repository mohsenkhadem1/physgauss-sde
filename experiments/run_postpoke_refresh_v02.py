"""Frozen all12/noise0 posterior refresh diagnostic; no action-policy rerun.

One worker per original object. Same acquired observations, four-dimensional
prior/model and selected action. New numerical reference seeds are prospective.
Old failed results remain immutable; every diagnostic object is retained.
"""
from pathlib import Path
import argparse,json,os,resource,shutil,time
import numpy as np
import torch
from experiments import campaign_postpoke_refresh_v02 as model
from experiments import run_diverse_force_cohort_v01 as previous
from experiments import campaign_compliant_full_count_continuation_v01 as count
from experiments.campaign_tempered_reference import tempered_smc
io=previous.io;ROOT=previous.ROOT;owner=previous.owner;bundle_io=previous.bundle_io
SOURCE='experiments/run_postpoke_refresh_v02.py'
PREVIOUS_PROTOCOL='runs/operator-dev/diverse-force-cohort-v01/protocol.json'
PREVIOUS_DIGEST='026aa14f924f02eed441c3d6e27d76ee9d88d02d5c5910170dd60767576c552c'
PREVIOUS_OUTPUT='runs/gaussian-campaign-v01/diverse-force-cohort-v01'
RESOURCES=dict(controller_cores=[12],worker_slots=[[i]for i in(0,2,4,6,8,10)],threads=1,worker_RSS_bytes=2*1024**3,no_GPU=True,worker_wall_s=2400,global_wall_s=5400,disk_free_reserve_bytes=30*1024**3,maximum_raw_bytes=4*1024**3)
DESIGN=dict(schema='postpoke-refresh-v02',objects=list(range(12)),action='Previously sealed target_variance choice per object; no action reselection',noise_index=0,same_original_observations=True,reconstructed_observation='Original runner did not persist poke vectors. Deterministic replay from frozen truth/force/noise seed must reproduce cached old weighted posterior moments before new inference',dimensions=[0,1,2,3],same_original_prior_and_discrete_model=True,particles=8192,prefix_cloud_particles=32768,prefix_SMC_seeds=[2026121101,2026121102,2026121103],bridge_seeds=[[2026305000+100*i+j for j in range(3)]for i in range(12)],subsample_seeds=[[2026320000+100*i+j for j in range(3)]for i in range(12)],fresh_seeds=[[2026300000+100*i+j for j in range(3)]for i in range(12)],MH_seeds=[2026310000+i for i in range(12)],bridge=dict(target='logL(A)+beta*logL(poke)',ESS_fraction=.8,rejuvenation_steps=16,max_stages=64,proposal='One fixed Gaussian covariance from old prefixSMCseed3; exact independence ratio, correlatedRW box rejection and prior-uniform mixture .5/.45/.05',RW_scale=.5,initialization='Uniform8192sample without replacement from each cached32768 prefix cloud; approximate prefixposterior initialization, not exact continuous proposal'),fresh_joint=dict(initialization='Independent original uniform4D prior, no cached posterior initialization',ESS_fraction=.8,rejuvenation_steps=8,proposal_scale=.08,prior_independence_probability=.25,max_stages=64),MH=dict(chains=8,warmup=1000,retained=64000,initialization='Independent fresh original uniform prior',proposal='Frozen designated fresh-jointSMCseed3 Gaussian; exact unchanged ridge_MH kernel'),gates=dict(rank_folded_split_Rhat=1.01,bulk_tail_ESS=400,CDF_MCSE=.01,unit_quantile_MCSE=.005,function_mean_MCSE_over_SD=.05,cross_CDF=.05,cross_unit_quantile=.02,cross_function_mean_over_SD=.1,predictive_mean_SD_RMS='max(2um,.03*reference_predictiveRMS_SD)',fresh_logZ_range=.05,bridge_conditional_logZ_range=.05),MCSE_batches=[100,200,400,800,1600],comparisons='All15SMC pairs and6SMCvsMH; no uniform-weight ESS qualification',bridge_evidence='Conditional poke evidence relative to approximate prefix clouds; cannot compare its absolute value to joint evidence because original per-seed prefix evidence was not saved',scope='Diagnostic selected-action/noise0 numerical posterior only; not16noise policy rerun, population calibration or physical-model validation',no_retry=True,no_object_drop=True,raw_LOCAL=True)


def sources():
 return sorted(set(previous.sources())|{SOURCE,'experiments/campaign_postpoke_refresh_v02.py','tests/test_postpoke_refresh_v02.py','experiments/campaign_postpoke_refresh_v01.py','experiments/run_postpoke_refresh_v01.py','tests/test_postpoke_refresh_v01.py'})


def predecessor():
 failed=ROOT/'runs/operator-dev/postpoke-refresh-v01/actual-supervisor-receipt.json'
 old=json.loads(failed.read_text())
 if old['completed']or old['exit_code']!=1 or not old['controller_reaped']or old['observed_owned_worker_PIDs']:raise ValueError('Preserved pre-worker v01 technical stop required')
 for pid in [old['supervisor_pid'],old['controller_pid']]:previous.old.previous.absent_process_and_group(pid)
 if (ROOT/'runs/gaussian-campaign-v01/postpoke-refresh-v01').exists():raise ValueError('v01 had no inference output directory')
 p=previous.validate(PREVIOUS_PROTOCOL,PREVIOUS_DIGEST);out=ROOT/PREVIOUS_OUTPUT
 execution=json.loads((out/'execution.json').read_text());summary=json.loads((out/'summary.json').read_text());receipt=json.loads((ROOT/'runs/operator-dev/diverse-force-cohort-v01/actual-supervisor-receipt.json').read_text())
 if not(execution['completed']and execution['all_owned_workers_reaped']and len(execution['records'])==12 and summary['completed']and summary['objects']==12 and summary['all_objects_retained']and receipt['completed']and receipt['controller_reaped']):raise ValueError('All12 completed original diagnostic objects required')
 for pid in [receipt['supervisor_pid'],receipt['controller_pid']]+receipt['observed_owned_worker_PIDs']:previous.old.previous.absent_process_and_group(pid)
 return p,execution,receipt


def prepare(protocol,output):
 dest=io.local(ROOT,protocol)
 if dest.exists()or io.local(ROOT,output).exists():raise FileExistsError('Exclusive new refresh namespace')
 predecessor();names=dict(technical_v01_protocol='runs/operator-dev/postpoke-refresh-v01/protocol.json',technical_v01_receipt='runs/operator-dev/postpoke-refresh-v01/actual-supervisor-receipt.json',technical_v01_log='runs/operator-dev/postpoke-refresh-v01/actual-refresh.log',cohort_protocol=PREVIOUS_PROTOCOL,cohort_execution=PREVIOUS_OUTPUT+'/execution.json',cohort_summary=PREVIOUS_OUTPUT+'/summary.json',cohort_receipt='runs/operator-dev/diverse-force-cohort-v01/actual-supervisor-receipt.json')
 for i in range(12):
  root=PREVIOUS_OUTPUT+f'/object-{i:02d}/'
  for name,file in dict(prefix='acquired-prefix-LOCAL.npz',decisions='decisions-before-target-truth.json',old_summary='summary.json',old_target='target-LOCAL.npy',old_pokes='pokes-LOCAL.npy',old_updates='updates-LOCAL.npz').items():names[f'object-{i}/'+name]=root+file
  for seed in DESIGN['prefix_SMC_seeds']:names[f'object-{i}/SMC-{seed}']=root+f'SMC-{seed}-LOCAL.npz'
 p=dict(schema='postpoke-refresh-v02',status='FROZEN_BEFORE_ANY_NEW_POSTPOKE_CALLS',design=DESIGN,resources=RESOURCES,runtime=io.runtime(),source_sha256={n:io.sha(ROOT/n)for n in sources()},inputs={k:io.record(ROOT,ROOT/n)for k,n in names.items()},output=output,raw_LOCAL=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest,index=None):
 p=io.read_json(ROOT,dict(path=protocol,sha256=digest))
 if p['design']!=DESIGN or p['resources']!=RESOURCES or p['runtime']!=io.runtime()or p['source_sha256']!={n:io.sha(ROOT/n)for n in sources()}:raise ValueError('Frozen refresh source/design/runtime/resources')
 for name,r in p['inputs'].items():
  if index is None or not name.startswith('object-')or name.startswith(f'object-{index}/'):io.checked(ROOT,r)
 predecessor();return p


def load(p,index,name):return io.checked(ROOT,p['inputs'][f'object-{index}/'+name])


def observations_and_replay(p,index):
 old_protocol=io.read_json(ROOT,p['inputs']['cohort_protocol']);decisions=json.loads(load(p,index,'decisions').read_text());action=decisions['choices']['target_variance']
 if decisions['protocol_sha256']!=PREVIOUS_DIGEST or decisions['index']!=index:raise ValueError('Original same-object sealed action')
 with np.load(load(p,index,'prefix'),allow_pickle=False)as z:observed_A=z['observation'].copy()
 force=list(model.cohort.old.poke_programs().values())[action]
 replay=model.JointObservationForward(force,observed_A,np.zeros(12));truth_unit=np.array(old_protocol['cohort_unit_coordinates_LOCAL'][index])[None]
 noiseless=replay.predictions(truth_unit)[0,1];noise_seed=old_protocol['design']['poke_noise_seeds'][index][0]
 observed_poke=noiseless+np.random.default_rng(noise_seed).normal(0,model.cohort.old.original.SIGMA,12)
 # Saved-only equality check, bounded target streaming, before any new sampler.
 old_target=np.load(load(p,index,'old_target'),mmap_mode='r');old_pokes=np.load(load(p,index,'old_pokes'),mmap_mode='r')
 weights,info=model.cohort.old.exact_reweight(old_pokes[action],observed_poke)
 mean,var,*_=previous.old.streamed_updates(old_target,weights[:,None])
 with np.load(load(p,index,'old_updates'),allow_pickle=False)as z:
  reference_mean=z['means'][action*16];reference_var=z['variances'][action*16]
 dm=float(np.max(abs(mean[0]-reference_mean)));dv=float(np.max(abs(var[0]-reference_var)))
 if dm>1e-12 or dv>1e-18:raise ValueError('Reconstructed original observation did not reproduce cached old posterior update')
 del old_target,old_pokes
 return observed_A,observed_poke,force,dict(action=action,noise_index=0,noise_seed=noise_seed,replayed_original_observation=True,maximum_old_mean_reproduction_difference_m=dm,maximum_old_variance_reproduction_difference_m2=dv,old_importance_diagnostic=info,old_mean=reference_mean,old_variance=reference_var)


def summarize(unit,prediction):
 flat=np.asarray(unit).reshape(-1,4);knots=np.linspace(0,1,33)
 return dict(unit_CDF=np.mean(flat[:,:,None]<=knots,axis=0),unit_quantiles=np.quantile(flat,[.05,.5,.95],axis=0).T,prediction=prediction)


def compare(a,b):
 distribution=previous.distribution_difference(a,b);field=model.cohort.compare_moments(a['prediction'],b['prediction']);d=abs(a['prediction']['function_mean']-b['prediction']['function_mean']);sd=b['prediction']['function_std'];ratio=float(np.max(np.divide(d,sd,out=np.where(d==0,0.,np.inf),where=sd>0)))
 return dict(distribution=distribution,field=field,maximum_function_mean_difference_over_SD=ratio,passed=bool(distribution['passed']and field['passed']and ratio<=.1))


def worker(protocol,digest,index,slot):
 start=time.monotonic();p=validate(protocol,digest,index);model.guard(RESOURCES['worker_slots'][slot]);torch.set_num_threads(1);torch.set_num_interop_threads(1)
 out=io.local(ROOT,p['output'])/f'object-{index:02d}';out.mkdir(parents=True,exist_ok=False)
 observed_A,observed_poke,force,replay=observations_and_replay(p,index)
 io.codec.exclusive_npz(out/'same-observations-LOCAL.npz',dict(prefix=observed_A,poke=observed_poke));io.write_new(out/'observation-replay.json',io.jsonable({k:v for k,v in replay.items()if k not in('old_mean','old_variance')}))
 forward=model.JointObservationForward(force,observed_A,observed_poke);fullforward=model.cohort.old.ArbitraryForceForward();clouds=[]
 for seed in DESIGN['prefix_SMC_seeds']:
  with np.load(load(p,index,f'SMC-{seed}'),allow_pickle=False)as z:clouds.append((z['unit'].copy(),z['log_likelihood'].copy()))
 proposal_mean=clouds[-1][0].mean(0);proposal_cov=np.cov(clouds[-1][0],rowvar=False)
 io.codec.exclusive_npz(out/'bridge-proposal-LOCAL.npz',dict(mean=proposal_mean,covariance=proposal_cov));values=[];bridgeZ=[];freshZ=[];fresh_unit=None
 for j,(cloud,oldll)in enumerate(clouds):
  if cloud.shape!=(32768,4):raise ValueError('Exact original full4D prefix cloud')
  ids=np.random.default_rng(DESIGN['subsample_seeds'][index][j]).choice(32768,size=8192,replace=False);initial=cloud[ids];a,_=forward.log_components(initial)
  if np.max(abs(a-oldll[ids]))>1e-9:raise ValueError('Original normalized prefix likelihood parity')
  r=model.bridge_smc(initial,forward.log_components,proposal_mean,proposal_cov,seed=DESIGN['bridge_seeds'][index][j],rejuvenation_steps=16)
  prediction=model.prediction_summary(fullforward,r.unit);value=summarize(r.unit,prediction);values.append(value);bridgeZ.append(r.log_conditional_evidence_estimate)
  io.codec.exclusive_npz(out/f'bridge-{j}-LOCAL.npz',dict(unit=r.unit,prefix_log_likelihood=r.prefix_log_likelihood,poke_log_likelihood=r.poke_log_likelihood));io.write_new(out/f'bridge-{j}-diagnostics.json',io.jsonable(dict(stages=r.stages,seed=r.seed,log_conditional_evidence_estimate=r.log_conditional_evidence_estimate,approximate_prefix_initialization=True)))
 for j,seed in enumerate(DESIGN['fresh_seeds'][index]):
  r=tempered_smc(model.cohort.old.base.BoxPrior(4),lambda state:forward.joint_log_likelihood(state.coordinates),particles=8192,seed=seed,ess_fraction=.8,rejuvenation_steps=8,proposal_scale=.08,independence_probability=.25,max_stages=64,periodic_indices=())
  unit=r.state.coordinates;value=summarize(unit,model.prediction_summary(fullforward,unit));values.append(value);freshZ.append(float(r.log_evidence_estimate));fresh_unit=unit
  io.codec.exclusive_npz(out/f'fresh-{j}-LOCAL.npz',dict(unit=unit,log_likelihood=r.log_likelihood));io.write_new(out/f'fresh-{j}-diagnostics.json',io.jsonable(r.diagnostics()))
 def freeze(mean,cov):
  io.codec.exclusive_npz(out/'joint-MH-proposal-LOCAL.npz',dict(mean=mean,covariance=cov));return io.record(ROOT,out/'joint-MH-proposal-LOCAL.npz')
 chain,diagnostics=model.cohort.ridge.ridge_mh(forward.joint_log_likelihood,fresh_unit,seed=DESIGN['MH_seeds'][index],warmup=1000,draws=64000,freeze_proposal=freeze)
 io.codec.exclusive_npz(out/'joint-MH-LOCAL.npz',dict(unit=chain,log_likelihood=diagnostics['log_likelihood']))
 mc=model.cohort.ridge.cdf_quantile_mc(chain);extra=count.long_batch_mc(chain);prediction=model.prediction_summary(fullforward,chain,chain=True);q=[v for row in mc['quantile_MCSE']for v in row]
 MHpassed=bool(diagnostics['diagnostics']['all_monitored_quantities_pass']and diagnostics['ridge_diagnostics']['all_monitored_quantities_pass']and np.max(mc['CDF_batch_MCSE'])<=.01 and all(v is not None and v<=.005 for v in q)and extra['passed']and prediction['field_MCSE']['passed']and prediction['function_MCSE_passed'])
 mh=summarize(chain,prediction);cross=[dict(method='bridge'if i<3 else'fresh',replicate=i%3,**compare(v,mh))for i,v in enumerate(values)];pairs=[dict(first=i,second=j,**compare(a,values[j]))for i,a in enumerate(values)for j in range(i+1,len(values))]
 fresh_range=max(freshZ)-min(freshZ);bridge_range=max(bridgeZ)-min(bridgeZ);passed=bool(MHpassed and all(v['passed']for v in cross+pairs)and fresh_range<=.05 and bridge_range<=.05)
 old_comparison=model.cohort.compare_moments(dict(mean=replay['old_mean'],variance=replay['old_variance']),prediction)
 io.write_new(out/'numerical-diagnostics-LOCAL.json',io.jsonable(dict(rank_ESS=diagnostics['diagnostics'],ridge_rank_ESS=diagnostics['ridge_diagnostics'],CDF_quantile_MCSE=mc,supplementary_CDF_quantile_MCSE=extra,function_rank_ESS=prediction['function_diagnostics'],function_maximum_relative_MCSE=prediction['function_maximum_relative_MCSE'],field_MCSE=prediction['field_MCSE'])))
 io.codec.exclusive_npz(out/'joint-prediction-LOCAL.npz',dict(mean=prediction['mean'],variance=prediction['variance'],mean_MCSE=prediction['mean_MCSE'],SD_MCSE=prediction['SD_MCSE']))
 io.write_new(out/'summary.json',io.jsonable(dict(completed=True,index=index,passed=passed,action=replay['action'],noise_index=0,observation_replay_passed=True,MH_passed=MHpassed,SMC_vs_MH=cross,SMC_pairs=pairs,fresh_joint_logZ_range=fresh_range,bridge_conditional_logZ_range=bridge_range,old_importance_vs_refreshed_joint=old_comparison,field_MCSE=prediction['field_MCSE'],maximum_function_relative_MCSE=prediction['function_maximum_relative_MCSE'],maximum_original_CDF_MCSE=float(np.max(mc['CDF_batch_MCSE'])),maximum_original_unit_quantile_MCSE=max((v for v in q if v is not None),default=None),supplementary_CDF_quantile_passed=extra['passed'],worker_wall_s=time.monotonic()-start,worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope=DESIGN['scope'],raw_LOCAL=True)))


def aggregate(rows):
 if len(rows)!=12 or sorted(r['index']for r in rows)!=list(range(12))or not all(r['completed']for r in rows):raise ValueError('All12 frozen diagnostic objects required')
 return dict(completed=True,objects=12,noise_trials_per_object=1,numerically_passed_objects=sum(r['passed']for r in rows),all_numerical_gates_passed=all(r['passed']for r in rows),all12_retained=True,policy_rerun=False,policy_advantage_claim=False,per_object=sorted(rows,key=lambda r:r['index']),scope=DESIGN['scope'],raw_LOCAL=True)


def run(protocol,digest):
 p=validate(protocol,digest);model.guard(RESOURCES['controller_cores']);out=io.local(ROOT,p['output']);out.mkdir(parents=True,exist_ok=False)
 if shutil.disk_usage(out).free<RESOURCES['disk_free_reserve_bytes']+RESOURCES['maximum_raw_bytes']:raise RuntimeError('Refresh disk reserve')
 io.atomic(out/'journal.json',dict(pid=os.getpid(),completed=False,status='BOOTSTRAP_BEFORE_EXPOSURE',live=[],records=[],pending_indices=list(range(12))))
 def command(i,slot):return ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','taskset','-c',str(RESOURCES['worker_slots'][slot][0]),str(ROOT/'.venv/bin/python'),'-m','experiments.run_postpoke_refresh_v02','worker','--protocol',protocol,'--digest',digest,'--index',str(i),'--slot',str(slot)]
 try:
  records=owner.owned_jobs(command,list(range(12)),6,out,worker_wall_s=2400,global_wall_s=5400,RSS_bytes=RESOURCES['worker_RSS_bytes'],controller_check=lambda:model.guard(RESOURCES['controller_cores']))
  rows=[json.loads((out/f'object-{i:02d}'/'summary.json').read_text())for i in range(12)];io.write_new(out/'summary.json',aggregate(rows));io.write_new(out/'execution.json',dict(completed=True,pid=os.getpid(),all_owned_workers_reaped=True,records=records,protocol_sha256=digest))
 except BaseException as error:
  io.write_new(out/'failure.json',dict(completed=False,error=repr(error),partial_artifacts_preserved=True,no_retry=True));raise


def supervise(protocol,digest):
 from experiments.supervise_surrogate_sbc_continuation import supervise as owned_supervise
 p=validate(protocol,digest);op=(ROOT/protocol).parent
 command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','taskset','-c','12',str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_postpoke_refresh_v02','run','--protocol',protocol,'--digest',digest]
 return owned_supervise(command,op/'actual-refresh.log',op/'actual-supervisor-receipt.json',journal_path=ROOT/p['output']/'journal.json',wall_s=5400,RSS_bytes=RESOURCES['worker_RSS_bytes'])


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','worker','supervise']);a.add_argument('--protocol',required=True);a.add_argument('--digest');a.add_argument('--output');a.add_argument('--index',type=int);a.add_argument('--slot',type=int);v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output))
 elif v.mode=='worker':worker(v.protocol,v.digest,v.index,v.slot)
 elif v.mode=='run':run(v.protocol,v.digest)
 else:
  receipt=supervise(v.protocol,v.digest);print(json.dumps(receipt));raise SystemExit(0 if receipt.get('completed')else 1)
