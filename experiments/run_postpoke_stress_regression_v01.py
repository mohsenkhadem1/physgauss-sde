"""Prospective regression on every one of nine original importance failures.

No filtered cases, replacement observations, action reselection or policy claim.
Source-only until root release and all prior count workers/groups are absent.
"""
import argparse,json,os,resource,shutil,time
import numpy as np
import torch
from experiments import campaign_postpoke_stress_regression_v01 as helper
from experiments import run_postpoke_count_continuation_v01 as count
old=helper.old;model=old.model;io=old.io;ROOT=old.ROOT;owner=old.owner
SOURCE='experiments/run_postpoke_stress_regression_v01.py'
COUNT_PROTOCOL='runs/operator-dev/postpoke-count-continuation-v01/protocol.json'
COUNT_OUTPUT='runs/gaussian-campaign-v01/postpoke-count-continuation-v01'
RESOURCES=dict(controller_cores=[16],worker_slots=[[i]for i in range(9)],threads=1,worker_RSS_bytes=2*1024**3,no_GPU=True,worker_wall_s=2400,global_wall_s=3000,disk_free_reserve_bytes=30*1024**3,maximum_raw_bytes=4*1024**3)
DESIGN=dict(schema='postpoke-stress-regression-v01',membership='All and only importance_gate_passed=False in complete original12x6x16 gategrid',cases=9,sorted_by=['object','action','noise_seed'],particles=32768,prefix_particles=32768,bridge_seeds=[[2026410000+100*i+j for j in range(3)]for i in range(9)],fresh_seeds=[[2026400000+100*i+j for j in range(3)]for i in range(9)],MH_seeds=[2026420000+100*i for i in range(9)],permutation_seeds=[[2026430000+100*i+j for j in range(3)]for i in range(9)],MH=dict(chains=8,warmup=1000,draws=64000,prior_initialization=True,proposal='Fixed designatedfreshjointreplicate3 Gaussian, unchanged exactridgeMH'),bridge=dict(old.DESIGN['bridge'],initialization='Uniform permutation of all32768 cached prefix particles, approximate prefix posterior initialization'),fresh_joint=old.DESIGN['fresh_joint'],gates=old.DESIGN['gates'],comparisons='All15SMCpairs+6vsnewpriorstartMH+freshandconditionalbridgeevidenceranges',same_original_prior_and_likelihood=True,same_original_acquisition_noise_and_poke_observations=True,replay='Cached512kMHweightedmoments at action*16+noiseindex mustreproducebeforeanynewinference; reproduceoldimportanceFAIL',old_negatives_preserved=True,all9_retained=True,no_retry=True,no_seed_search=True,scope='Regression on known bad importance inputs; no population calibration or active-policy superiority claim',raw_LOCAL=True)


def sources():
 return sorted(set(count.sources())|{SOURCE,'experiments/campaign_postpoke_stress_regression_v01.py','tests/test_postpoke_stress_regression_v01.py'})


def count_terminal(protocol_record=None):
 record=protocol_record or io.record(ROOT,ROOT/COUNT_PROTOCOL)
 p=count.validate(record['path'],record['sha256']);out=ROOT/p['output']
 execution=json.loads((out/'execution.json').read_text());receipt=json.loads((ROOT/record['path']).parent.joinpath('actual-supervisor-receipt.json').read_text());summary=json.loads((out/'summary.json').read_text())
 helper.terminal_count_gate(execution,receipt,summary,record['sha256'],old.previous.old.previous.absent_process_and_group)
 return p


def prepare(protocol,output):
 dest=io.local(ROOT,protocol)
 if dest.exists()or io.local(ROOT,output).exists():raise FileExistsError('Exclusive stressnamespace')
 c=count_terminal();refresh=old.validate(count.PREVIOUS_PROTOCOL,count.PREVIOUS_DIGEST)
 names=dict(count_protocol=COUNT_PROTOCOL,count_execution=c['output']+'/execution.json',count_summary=c['output']+'/summary.json',count_receipt=str((ROOT/COUNT_PROTOCOL).parent.joinpath('actual-supervisor-receipt.json').relative_to(ROOT)),refresh_protocol=count.PREVIOUS_PROTOCOL)
 all_rows=[]
 for i in range(12):
  path=old.PREVIOUS_OUTPUT+f'/object-{i:02d}/importance-and-MCSE.json';names[f'gate-{i}']=path;all_rows.append(json.loads((ROOT/path).read_text()))
 cohort=io.read_json(ROOT,refresh['inputs']['cohort_protocol']);cases=helper.failed_cases(all_rows,cohort['design']['poke_noise_seeds'])
 p=dict(schema='postpoke-stress-regression-v01',status='FROZEN_BEFORE_ANY_NEW_BADCASE_CALLS',design=DESIGN,resources=RESOURCES,runtime=io.runtime(),source_sha256={n:io.sha(ROOT/n)for n in sources()},inputs={k:io.record(ROOT,ROOT/n)for k,n in names.items()},original_inputs=refresh['inputs'],cases=cases,output=output,raw_LOCAL=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest,index=None):
 p=io.read_json(ROOT,dict(path=protocol,sha256=digest))
 if p['design']!=DESIGN or p['resources']!=RESOURCES or p['runtime']!=io.runtime()or p['source_sha256']!={n:io.sha(ROOT/n)for n in sources()}:raise ValueError('Frozen stressrecipe/source/runtime/resources')
 for r in p['inputs'].values():io.checked(ROOT,r)
 for name,r in p['original_inputs'].items():
  if index is None or not name.startswith('object-')or name.startswith(f"object-{p['cases'][index]['object_index']}/"):io.checked(ROOT,r)
 cohort=io.read_json(ROOT,p['original_inputs']['cohort_protocol']);rows=[json.loads(io.checked(ROOT,p['inputs'][f'gate-{i}']).read_text())for i in range(12)]
 if helper.failed_cases(rows,cohort['design']['poke_noise_seeds'])!=p['cases']:raise ValueError('Exactall9membership changed')
 count_terminal(p['inputs']['count_protocol']);return p


def original(p,index,name):return io.checked(ROOT,p['original_inputs'][f'object-{index}/'+name])


def worker(protocol,digest,index,slot):
 start=time.monotonic();p=validate(protocol,digest,index);model.guard(RESOURCES['worker_slots'][slot]);torch.set_num_threads(1);torch.set_num_interop_threads(1)
 case=p['cases'][index];obj=case['object_index'];out=io.local(ROOT,p['output'])/f'case-{index:02d}';out.mkdir(parents=True,exist_ok=False)
 with np.load(original(p,obj,'prefix'),allow_pickle=False)as z:observed_A=z['observation'].copy()
 cohort=io.read_json(ROOT,p['original_inputs']['cohort_protocol']);truth=np.asarray(cohort['cohort_unit_coordinates_LOCAL'][obj])
 target=np.load(original(p,obj,'old_target'),mmap_mode='r');pokes=np.load(original(p,obj,'old_pokes'),mmap_mode='r')
 with np.load(original(p,obj,'old_updates'),allow_pickle=False)as z:observed,force,replay,oldmean,oldvar=helper.replay_observation(model,observed_A,truth,case,target,pokes,z['means'],z['variances'])
 del target,pokes,truth
 io.codec.exclusive_npz(out/'same-observations-LOCAL.npz',dict(prefix=observed_A,poke=observed));io.write_new(out/'observation-replay.json',io.jsonable(dict(case=case,**replay)))
 forward=model.JointObservationForward(force,observed_A,observed);fullforward=model.cohort.old.ArbitraryForceForward();clouds=[]
 for seed in old.DESIGN['prefix_SMC_seeds']:
  with np.load(original(p,obj,f'SMC-{seed}'),allow_pickle=False)as z:clouds.append((z['unit'].copy(),z['log_likelihood'].copy()))
 mean=clouds[-1][0].mean(0);cov=np.cov(clouds[-1][0],rowvar=False);io.codec.exclusive_npz(out/'bridge-proposal-LOCAL.npz',dict(mean=mean,covariance=cov))
 values=[];bridgeZ=[];freshZ=[];fresh_unit=None
 for j,(cloud,oldll)in enumerate(clouds):
  if cloud.shape!=(32768,4):raise ValueError('Originalprefix32768cloud')
  ids=np.random.default_rng(DESIGN['permutation_seeds'][index][j]).choice(32768,size=32768,replace=False);initial=cloud[ids];a,_=forward.log_components(initial)
  if np.max(abs(a-oldll[ids]))>1e-9:raise ValueError('Normalizedprefixlikelihood unchanged')
  r=model.bridge_smc(initial,forward.log_components,mean,cov,seed=DESIGN['bridge_seeds'][index][j],rejuvenation_steps=16);bridgeZ.append(r.log_conditional_evidence_estimate);values.append(old.summarize(r.unit,model.prediction_summary(fullforward,r.unit)))
  io.codec.exclusive_npz(out/f'bridge-{j}-LOCAL.npz',dict(unit=r.unit,prefix_log_likelihood=r.prefix_log_likelihood,poke_log_likelihood=r.poke_log_likelihood));io.write_new(out/f'bridge-{j}-diagnostics.json',io.jsonable(dict(stages=r.stages,seed=r.seed,log_conditional_evidence_estimate=r.log_conditional_evidence_estimate,approximate_prefix_initialization=True)))
 for j,seed in enumerate(DESIGN['fresh_seeds'][index]):
  r=old.tempered_smc(model.cohort.old.base.BoxPrior(4),lambda state:forward.joint_log_likelihood(state.coordinates),particles=32768,seed=seed,ess_fraction=.8,rejuvenation_steps=8,proposal_scale=.08,independence_probability=.25,max_stages=64,periodic_indices=());unit=r.state.coordinates;fresh_unit=unit
  values.append(old.summarize(unit,model.prediction_summary(fullforward,unit)));freshZ.append(float(r.log_evidence_estimate));io.codec.exclusive_npz(out/f'fresh-{j}-LOCAL.npz',dict(unit=unit,log_likelihood=r.log_likelihood));io.write_new(out/f'fresh-{j}-diagnostics.json',io.jsonable(r.diagnostics()))
 def freeze(mean,cov):
  io.codec.exclusive_npz(out/'joint-MH-proposal-LOCAL.npz',dict(mean=mean,covariance=cov));return io.record(ROOT,out/'joint-MH-proposal-LOCAL.npz')
 chain,d=model.cohort.ridge.ridge_mh(forward.joint_log_likelihood,fresh_unit,seed=DESIGN['MH_seeds'][index],warmup=1000,draws=64000,freeze_proposal=freeze)
 io.codec.exclusive_npz(out/'joint-MH-LOCAL.npz',dict(unit=chain,log_likelihood=d['log_likelihood']));mc=model.cohort.ridge.cdf_quantile_mc(chain);extra=old.count.long_batch_mc(chain);prediction=model.prediction_summary(fullforward,chain,chain=True);q=[v for row in mc['quantile_MCSE']for v in row]
 mhpass=bool(d['diagnostics']['all_monitored_quantities_pass']and d['ridge_diagnostics']['all_monitored_quantities_pass']and np.max(mc['CDF_batch_MCSE'])<=.01 and all(v is not None and v<=.005 for v in q)and extra['passed']and prediction['field_MCSE']['passed']and prediction['function_MCSE_passed'])
 mh=old.summarize(chain,prediction);cross=[dict(method='bridge'if i<3 else'fresh',replicate=i%3,**old.compare(v,mh))for i,v in enumerate(values)];pairs=[dict(first=i,second=j,**old.compare(a,values[j]))for i,a in enumerate(values)for j in range(i+1,len(values))];fresh_range=max(freshZ)-min(freshZ);bridge_range=max(bridgeZ)-min(bridgeZ)
 passed=bool(mhpass and all(v['passed']for v in cross+pairs)and fresh_range<=.05 and bridge_range<=.05);oldcomparison=model.cohort.compare_moments(dict(mean=oldmean,variance=oldvar),prediction)
 io.codec.exclusive_npz(out/'joint-prediction-LOCAL.npz',dict(mean=prediction['mean'],variance=prediction['variance'],mean_MCSE=prediction['mean_MCSE'],SD_MCSE=prediction['SD_MCSE'],function_mean=prediction['function_mean'],function_std=prediction['function_std']))
 io.write_new(out/'numerical-diagnostics-LOCAL.json',io.jsonable(dict(rank_ESS=d['diagnostics'],ridge_rank_ESS=d['ridge_diagnostics'],CDF_quantile_MCSE=mc,supplementary_CDF_quantile_MCSE=extra,function_rank_ESS=prediction['function_diagnostics'],function_maximum_relative_MCSE=prediction['function_maximum_relative_MCSE'],field_MCSE=prediction['field_MCSE'])))
 io.write_new(out/'summary.json',io.jsonable(dict(completed=True,index=index,case=case,passed=passed,MH_passed=mhpass,observation_replay_passed=True,old_importance_failed=True,SMC_vs_MH=cross,SMC_pairs=pairs,fresh_joint_logZ_range=fresh_range,bridge_conditional_logZ_range=bridge_range,old_importance_vs_refreshed_joint=oldcomparison,field_MCSE=prediction['field_MCSE'],maximum_function_relative_MCSE=prediction['function_maximum_relative_MCSE'],maximum_CDF_MCSE=float(np.max(mc['CDF_batch_MCSE'])),maximum_unit_quantile_MCSE=max((v for v in q if v is not None),default=None),supplementary_CDF_quantile_passed=extra['passed'],worker_wall_s=time.monotonic()-start,worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope=DESIGN['scope'],raw_LOCAL=True)))


def aggregate(rows):
 if len(rows)!=9 or sorted(r['index']for r in rows)!=list(range(9))or not all(r['completed']and r['old_importance_failed']and r['observation_replay_passed']for r in rows):raise ValueError('All9 originalbadinputs retained')
 return dict(completed=True,cases=9,objects=4,numerically_passed_cases=sum(r['passed']for r in rows),all_numerical_gates_passed=all(r['passed']for r in rows),all9_retained=True,old9_failures_preserved=True,policy_rerun=False,policy_advantage_claim=False,per_case=sorted(rows,key=lambda r:r['index']),scope=DESIGN['scope'],raw_LOCAL=True)


def worker_command(protocol,digest,index,slot,*,guard_only=False):
 command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','taskset','-c',str(RESOURCES['worker_slots'][slot][0]),str(ROOT/'.venv/bin/python'),'-m','experiments.run_postpoke_stress_regression_v01']
 return command+(['guardworker','--slot',str(slot)]if guard_only else['worker','--protocol',protocol,'--digest',digest,'--index',str(index),'--slot',str(slot)])


def controller_command(protocol,digest,*,guard_only=False):
 command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','taskset','-c','16',str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_postpoke_stress_regression_v01']
 return command+(['guardcontroller']if guard_only else['run','--protocol',protocol,'--digest',digest])


def run(protocol,digest):
 p=validate(protocol,digest);model.guard(RESOURCES['controller_cores']);out=io.local(ROOT,p['output']);out.mkdir(parents=True,exist_ok=False)
 if shutil.disk_usage(out).free<RESOURCES['disk_free_reserve_bytes']+RESOURCES['maximum_raw_bytes']:raise RuntimeError('Stressregressiondiskreserve')
 io.atomic(out/'journal.json',dict(pid=os.getpid(),completed=False,status='BOOTSTRAP_BEFORE_EXPOSURE',live=[],records=[],pending_indices=list(range(9))))
 def command(i,slot):return worker_command(protocol,digest,i,slot)
 try:
  records=owner.owned_jobs(command,list(range(9)),9,out,worker_wall_s=2400,global_wall_s=3000,RSS_bytes=RESOURCES['worker_RSS_bytes'],controller_check=lambda:model.guard(RESOURCES['controller_cores']));rows=[json.loads((out/f'case-{i:02d}'/'summary.json').read_text())for i in range(9)];io.write_new(out/'summary.json',aggregate(rows));io.write_new(out/'execution.json',dict(completed=True,pid=os.getpid(),all_owned_workers_reaped=True,records=records,protocol_sha256=digest))
 except BaseException as error:
  io.write_new(out/'failure.json',dict(completed=False,error=repr(error),partial_artifacts_preserved=True,no_retry=True));raise


def supervise(protocol,digest):
 from experiments.supervise_surrogate_sbc_continuation import supervise as owned
 p=validate(protocol,digest);op=(ROOT/protocol).parent
 command=controller_command(protocol,digest)
 return owned(command,op/'actual-stress.log',op/'actual-supervisor-receipt.json',journal_path=ROOT/p['output']/'journal.json',wall_s=3000,RSS_bytes=RESOURCES['worker_RSS_bytes'])


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','worker','supervise','guardworker','guardcontroller']);a.add_argument('--protocol');a.add_argument('--digest');a.add_argument('--output');a.add_argument('--index',type=int);a.add_argument('--slot',type=int);v=a.parse_args()
 if v.mode=='guardworker':model.guard(RESOURCES['worker_slots'][v.slot])
 elif v.mode=='guardcontroller':model.guard(RESOURCES['controller_cores'])
 elif v.mode=='prepare':print(prepare(v.protocol,v.output))
 elif v.mode=='worker':worker(v.protocol,v.digest,v.index,v.slot)
 elif v.mode=='run':run(v.protocol,v.digest)
 else:
  r=supervise(v.protocol,v.digest);print(json.dumps(r));raise SystemExit(0 if r.get('completed')else 1)
