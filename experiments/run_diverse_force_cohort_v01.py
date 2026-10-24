"""Prospectively frozen CPU-only twelve-object force/poking challenge.

Source-only until explicit release. No raw fields, parameter arrays or samples
are publication artifacts. All objects, including numerical failures, retained.
"""
from pathlib import Path
import argparse,json,os,resource,time
import numpy as np
import torch
from experiments import campaign_diverse_force_cohort_v01 as helper
from experiments import campaign_compliant_observation_forward as observation
from experiments import campaign_compliant_full_count_continuation_v01 as count
from experiments import run_compliant_force_challenge_v01 as old
io=old.io;ROOT=old.ROOT;owner=old.previous.owner;bundle_io=old.bundle_io
SOURCE='experiments/run_diverse_force_cohort_v01.py'
RESOURCES=dict(controller_cores=[6,7],worker_slots=[[i,i+1]for i in range(8,32,2)],threads=2,worker_RSS_bytes=4*1024**3,no_GPU=True,worker_wall_s=1200,global_wall_s=1800,disk_free_reserve_bytes=30*1024**3,maximum_raw_bytes=20*1024**3)
DESIGN=dict(schema='diverse-force-cohort-v01',objects=12,cohort_seed=2026102404,cohort='Fixed Latin-hypercube in full original4D intrinsic prior, no object selection',support_geometry='Fixture Gaussian center varies only along original x-line; not general3D boundary conditions',observed_A_frames=[12,24],observed_A_nodes=[1,4],acquisition_noise_seeds=list(range(2026102500,2026102512)),SMC_particles=32768,SMC_seeds=[2026121101,2026121102,2026121103],MH_chains=8,MH_warmup=1000,MH_retained=64000,MH_seeds=list(range(2026102600,2026102612)),proposal='Fixed designatedSMCseed3 covariance; original exact independence ratio and correlated box rejection',target_frames=list(helper.TARGET_FRAMES),target_programs=list(helper.old.force_programs()),target_dimension=135,metric='Selected-frame135coordinate aggregate RMS, not full-trajectory metric',pokes=list(helper.old.poke_programs()),poke_noise_seeds=[[2026200000+100*i+j for j in range(16)]for i in range(12)],policies=['target_variance','fixed_first','local_Fisher','seeded_random','uniform_random_expected'],random_policy_seeds=list(range(2026102700,2026102712)),Fisher_steps=[1e-4,5e-5],MCSE_batches=list(helper.BATCHES),anchor='Saved old1620metric reference only, excluded from12object135metric denominators; no new anchor inference or forwards',same_parameter_gates=True,predictive_tolerance='max(2um,.03*pooledpredictiveRMS_SD)',importance_gates='ESS>=400 AND fraction>=.05 AND16half_mass_ESS>=8, equal-length normalized ratio-influence MCMC MCSE',acquisition_gates='All8chain tops and3SMC tops agree; scoremaxrelative difference<=.1, Fisherderivative topstable',comparison='All12objects retained; clusterbootstrap12objectswith16pairednoise,descriptiveconditionalLHScohortCI; empirical16half+8chain+3SMCcontrast envelope added, not rigorous bound',all_gates_required_for_superiority=True,no_retry=True,no_object_discard=True,raw_LOCAL=True)


def sources():
 return sorted(set(old.sources())|{SOURCE,'experiments/campaign_diverse_force_cohort_v01.py','experiments/campaign_force_prediction_precision_v01.py','tests/test_diverse_force_cohort_v01.py','experiments/campaign_compliant_observation_forward.py','experiments/campaign_compliant_full_count_continuation_v01.py','tests/test_force_prediction_precision_v01.py'})


def prepare(protocol,output):
 dest=io.local(ROOT,protocol)
 if dest.exists()or io.local(ROOT,output).exists():raise FileExistsError('Exclusive cohort namespace')
 units=helper.cohort_units()
 inputs={k:io.record(ROOT,ROOT/n)for k,n in dict(old_anchor_summary='runs/gaussian-campaign-v01/compliant-force-challenge-v01/summary.json',new_precision_protocol='runs/operator-dev/force-prediction-precision-v01/protocol.json',new_precision_summary='runs/operator-dev/force-prediction-precision-v01/summary.json').items()}
 check_predecessor(inputs)
 value=dict(schema='diverse-force-cohort-v01',status='FROZEN_BEFORE_TRUTH_FORWARD_OR_INFERENCE',design=DESIGN,resources=RESOURCES,runtime=io.runtime(),source_sha256={n:io.sha(ROOT/n)for n in sources()},cohort_unit_coordinates_LOCAL=units.tolist(),inputs=inputs,output=output,raw_LOCAL=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,value);return io.sha(dest)


def check_predecessor(inputs):
 for r in inputs.values():io.checked(ROOT,r)
 new=io.read_json(ROOT,inputs['new_precision_summary']);anchor=io.read_json(ROOT,inputs['old_anchor_summary'])
 if not(new['completed']and new['all_new_precision_diagnostics_passed']and new['saved_independent_SMC_all_three_passed']and new['saved_pre_acquisition_stability_passed']and new['new_diagnostic_with_independent_reference_and_acquisition_passed']and new['older_failed_qualification_remains_failed']):raise ValueError('Completed precise saved-anchor diagnostic and independent-reference gates required')
 if anchor['importance_and_split_half_passed']or anchor['pre_split_half_passed']or not anchor['pre_acquisition_stable']:raise ValueError('Preserve original failed half-subset diagnostic and passed acquisition')
 if not all(r['passed']for r in anchor['SMC_comparisons']):raise ValueError('Three original independent references required')


def validate(protocol,digest):
 p=io.read_json(ROOT,dict(path=protocol,sha256=digest))
 if p['design']!=DESIGN or p['resources']!=RESOURCES or p['runtime']!=io.runtime()or p['source_sha256']!={n:io.sha(ROOT/n)for n in sources()}or not np.array_equal(p['cohort_unit_coordinates_LOCAL'],helper.cohort_units()):raise ValueError('Frozen full source/design/cohort/runtime')
 check_predecessor(p['inputs'])
 return p


def scalar_moments(target,weights=None):
 if weights is None:weights=np.full(len(target),1/len(target))
 mean,var=helper.old.weighted_prediction(target,weights)
 return dict(mean=mean,variance=var)


def evaluate_sample(target,pokes,unit,truth_target,truth_pokes,choices,*,noise_seeds,chain=False):
 n=len(target);before=helper.batch_moment_mcse(target.reshape(8,64000,135))if chain else scalar_moments(target)
 rows=[];means=[];variances=[];errors=[];cluster_errors=[];passed=helper.moment_gate(before)['passed']if chain else True
 for k in range(6):
  for seed in noise_seeds:
   y=truth_pokes[k]+np.random.default_rng(seed).normal(0,helper.old.original.SIGMA,12)
   w,info=helper.old.exact_reweight(pokes[k],y)
   try:
    r=helper.weighted_ratio_mcse(target.reshape(8,64000,135),w.reshape(8,64000))if chain else scalar_moments(target,w)
    gate=helper.moment_gate(r)if chain else dict(passed=True)
   except (ValueError,FloatingPointError)as error:
    r=scalar_moments(target,w);gate=dict(passed=False,unresolved=str(error))
   if chain:
    cluster_errors.append(np.sqrt(np.mean((r['cluster_means']-truth_target)**2,axis=1))if 'cluster_means'in r else np.full(24,np.nan))
   passed=bool(passed and info['importance_gate_passed']and gate['passed']);means.append(r['mean']);variances.append(r['variance']);errors.append(float(np.sqrt(np.mean((r['mean']-truth_target)**2))));rows.append(dict(action=k,noise_seed=seed,importance=info,MCSE=gate))
 return dict(before=before,means=np.array(means),variances=np.array(variances),errors=np.array(errors).reshape(6,16),rows=rows,passed=passed,cluster_errors=np.array(cluster_errors).T.reshape(24,6,16)if chain else None)


def distribution_summary(unit):
 flat=unit.reshape(-1,4);knots=np.linspace(0,1,33)
 return dict(unit_CDF=np.mean(flat[:,:,None]<=knots,axis=0),unit_quantiles=np.quantile(flat,[.05,.5,.95],axis=0).T)


def distribution_difference(a,b):
 dc=float(np.max(abs(a['unit_CDF']-b['unit_CDF'])));dq=float(np.max(abs(a['unit_quantiles']-b['unit_quantiles'])))
 return dict(max_CDF_difference=dc,max_quantile_difference=dq,passed=bool(dc<=.05 and dq<=.02))


def worker(protocol,digest,index,slot):
 start=time.monotonic();p=validate(protocol,digest);bundle_io.guard(RESOURCES['worker_slots'][slot]);torch.set_num_threads(2);torch.set_num_interop_threads(2)
 out=io.local(ROOT,p['output'])/f'object-{index:02d}';out.mkdir(parents=True,exist_ok=False)
 unit_truth=np.array(p['cohort_unit_coordinates_LOCAL'][index]);theta_truth=helper.physical(unit_truth[None])
 obsforward=observation.ObservationOnlyForward();forward=helper.old.ArbitraryForceForward()
 # Only acquired prefixA is exposed to inference; no target or poke truth yet.
 obs=obsforward.observations(theta_truth)[0]+np.random.default_rng(DESIGN['acquisition_noise_seeds'][index]).normal(0,helper.old.original.SIGMA,12)
 io.codec.exclusive_npz(out/'acquired-prefix-LOCAL.npz',dict(observation=obs))
 smcs=[];logZ=[]
 for seed in DESIGN['SMC_seeds']:
  result=helper.old.base.smc(obsforward,helper.DIMS,obs,DESIGN['SMC_particles'],seed)
  u=np.array(result.state.coordinates);smcs.append(u);logZ.append(float(result.diagnostics()['log_evidence_estimate']))
  io.codec.exclusive_npz(out/f'SMC-{seed}-LOCAL.npz',dict(unit=u,log_likelihood=result.log_likelihood))
 def freeze(mean,cov):
  io.codec.exclusive_npz(out/'proposal-LOCAL.npz',dict(mean=mean,covariance=cov));return io.record(ROOT,out/'proposal-LOCAL.npz')
 chain,diag=helper.ridge.ridge_mh(lambda x:obsforward.log_likelihood(x,helper.DIMS,obs),smcs[-1],seed=DESIGN['MH_seeds'][index],warmup=1000,draws=64000,freeze_proposal=freeze)
 io.codec.exclusive_npz(out/'MH-LOCAL.npz',dict(unit=chain,log_likelihood=diag['log_likelihood']))
 mc=helper.ridge.cdf_quantile_mc(chain);extra=count.long_batch_mc(chain);q=[x for row in mc['quantile_MCSE']for x in row]
 param_passed=bool(diag['diagnostics']['all_monitored_quantities_pass']and diag['ridge_diagnostics']['all_monitored_quantities_pass']and np.max(mc['CDF_batch_MCSE'])<=.01 and all(x is not None and x<=.005 for x in q)and extra['passed'])
 unit=chain.reshape(-1,4);target,pokes,functions=helper.joint_features(forward,helper.physical(unit),with_functions=True)
 from physgauss.chain_diagnostics import diagnose_quantities
 function_diagnostics=diagnose_quantities(torch.tensor(functions.reshape(8,64000,18)),['function_'+str(i)for i in range(18)])
 function_mean=functions.mean(0);function_std=functions.std(0);function_mcse=[]
 for k in helper.BATCHES:
  block=functions.reshape(8,64000//k,k,18).mean(2).reshape(-1,18);function_mcse.append(block.std(0,ddof=1)/np.sqrt(len(block)))
 function_error=np.max(function_mcse,axis=0);relative=np.divide(function_error,function_std,out=np.where(function_error==0,0.,np.inf),where=function_std>0);function_passed=bool(function_diagnostics['all_monitored_quantities_pass']and np.max(relative)<=.05);param_passed=bool(param_passed and function_passed)
 io.write_new(out/'diagnostics-LOCAL.json',io.jsonable(dict(parameter_rank_ESS=diag['diagnostics'],ridge_rank_ESS=diag['ridge_diagnostics'],function_rank_ESS=function_diagnostics,original_CDF_quantile_MCSE=mc,supplementary_CDF_quantile_MCSE=extra,function_maximum_relative_MCSE=float(relative.max()),function_precision_passed=function_passed,parameter_precision_passed=param_passed)))
 del functions
 np.save(out/'target-LOCAL.npy',target);np.save(out/'pokes-LOCAL.npy',pokes)
 scores=helper.old.gaussian_target_reduction(target,pokes);cov=np.cov(unit,rowvar=False)
 fisher=[old.fisher_scores(forward,unit.mean(0),cov,helper.old.poke_programs(),h)for h in DESIGN['Fisher_steps']]
 choices=dict(target_variance=int(np.argmax(scores)),fixed_first=0,local_Fisher=int(np.argmax(fisher[-1])),seeded_random=int(np.random.default_rng(DESIGN['random_policy_seeds'][index]).integers(6)))
 chain_scores=[helper.old.gaussian_target_reduction(target[i*64000:(i+1)*64000],pokes[:,i*64000:(i+1)*64000])for i in range(8)]
 acquisition_pass=bool(np.argmax(fisher[0])==np.argmax(fisher[1])and all(np.argmax(s)==np.argmax(scores)and np.max(abs(s-scores))<=.1*max(abs(scores).max(),np.finfo(float).tiny)for s in chain_scores))
 io.write_new(out/'decisions-before-target-truth.json',dict(protocol_sha256=digest,index=index,choices=choices,scores=scores.tolist(),Fisher_scores=[s.tolist()for s in fisher],target_and_poke_truth_not_accessed=True))
 truth_target,truth_pokes=helper.joint_features(forward,theta_truth);truth_target=truth_target[0];truth_pokes=truth_pokes[:,0]
 mh=evaluate_sample(target,pokes,unit,truth_target,truth_pokes,choices,noise_seeds=DESIGN['poke_noise_seeds'][index],chain=True)
 # Baselines use acquired posterior only; truth is used solely for scoring.
 mean_theta=helper.physical(unit).mean(0)[None];nominal=helper.physical(np.full((1,4),.5))
 plugin=helper.joint_features(forward,mean_theta)[0][0];prior=helper.joint_features(forward,nominal)[0][0]
 baselines={name:float(np.sqrt(np.mean((v-truth_target)**2)))for name,v in [('posterior_mean',mh['before']['mean']),('physical_mean_plugin',plugin),('nominal_prior_center',prior)]}
 comparisons=[];smc_errors=[];smc_moments=[];mh_dist=distribution_summary(chain);smc_dists=[]
 for seed,smcunit in zip(DESIGN['SMC_seeds'],smcs):
  st,sy,sf=helper.joint_features(forward,helper.physical(smcunit),with_functions=True);sf_mean=sf.mean(0);sf_std=sf.std(0);function_difference=abs(sf_mean-function_mean);function_ratio=float(np.max(np.divide(function_difference,function_std,out=np.where(function_difference==0,0.,np.inf),where=function_std>0)));del sf;su=evaluate_sample(st,sy,smcunit,truth_target,truth_pokes,choices,noise_seeds=DESIGN['poke_noise_seeds'][index])
  ss=helper.old.gaussian_target_reduction(st,sy);same=int(np.argmax(ss))==choices['target_variance'];score_error=float(np.max(abs(ss-scores))/max(abs(scores).max(),np.finfo(float).tiny));before=helper.compare_moments(su['before'],mh['before']);posts=[helper.compare_moments(dict(mean=a,variance=b),dict(mean=c,variance=d))for a,b,c,d in zip(su['means'],su['variances'],mh['means'],mh['variances'])]
  dist=distribution_summary(smcunit);smc_dists.append(dist);dc=distribution_difference(dist,mh_dist)
  comparisons.append(dict(seed=seed,distribution=dc,before=before,maximum_post_mean_difference_m=max(r['mean_RMS_difference_m']for r in posts),maximum_post_SD_difference_m=max(r['SD_RMS_difference_m']for r in posts),post_moments_passed=all(r['passed']for r in posts),same_top_action=same,maximum_relative_score_difference=score_error,importance_passed=su['passed'],maximum_function_difference_over_posterior_SD=function_ratio,passed=bool(function_ratio<=.1 and dc['passed']and before['passed']and all(r['passed']for r in posts)and same and score_error<=.1 and su['passed'])))
  smc_errors.append(su['errors']);smc_moments.append(dict(before=su['before'],means=su['means'],variances=su['variances'],function_mean=sf_mean,function_std=sf_std));del st,sy,su
 pairwise=[]
 for i,a in enumerate(smc_dists):
  for j in range(i+1,len(smc_dists)):
   row=distribution_difference(a,smc_dists[j]);pa=smc_moments[i];pb=smc_moments[j];pre=helper.compare_moments(pa['before'],pb['before']);post=[helper.compare_moments(dict(mean=a,variance=b),dict(mean=c,variance=d))for a,b,c,d in zip(pa['means'],pa['variances'],pb['means'],pb['variances'])];fd=abs(pa['function_mean']-pb['function_mean']);fr=float(np.max(np.divide(fd,pb['function_std'],out=np.where(fd==0,0.,np.inf),where=pb['function_std']>0)));row.update(pre_moments_passed=pre['passed'],post_moments_passed=all(v['passed']for v in post),maximum_function_mean_difference_over_SD=fr,passed=bool(row['passed']and pre['passed']and all(v['passed']for v in post)and fr<=.1));pairwise.append(row)
 logz_range=max(logZ)-min(logZ);passed=bool(param_passed and mh['passed']and acquisition_pass and all(r['passed']for r in comparisons+pairwise)and logz_range<=.05)
 np.savez(out/'updates-LOCAL.npz',errors=mh['errors'],means=mh['means'],variances=mh['variances'],SMC_errors=np.array(smc_errors))
 io.write_new(out/'importance-and-MCSE.json',mh['rows'])
 policy_errors={name:mh['errors'][k].tolist()for name,k in choices.items()};policy_errors['uniform_random_expected']=mh['errors'].mean(0).tolist()
 envelopes={}
 for name in DESIGN['policies'][1:]:
  main=np.mean(mh['errors'][choices['target_variance']]-(mh['errors'].mean(0)if name=='uniform_random_expected'else mh['errors'][choices[name]]))
  ref=[np.mean(e[choices['target_variance']]-(e.mean(0)if name=='uniform_random_expected'else e[choices[name]]))for e in [*smc_errors,*mh['cluster_errors']]]
  envelopes[name]=float(max(abs(v-main)for v in ref))if np.isfinite(ref).all()else None
 io.write_new(out/'summary.json',dict(completed=True,index=index,passed=passed,parameter_precision_passed=param_passed,function_precision_passed=function_passed,maximum_function_relative_MCSE=float(relative.max()),predictive_precision_passed=mh['passed'],acquisition_stable=acquisition_pass,logZ_range=logz_range,SMC_comparisons=comparisons,SMC_pairwise=pairwise,choices=choices,baseline_RMSE_m=baselines,policy_RMSE_m=policy_errors,empirical_numerical_envelope_m=envelopes,worker_wall_s=time.monotonic()-start,worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,raw_LOCAL=True))


def aggregate(rows):
 if len(rows)!=12 or sorted(r['index']for r in rows)!=list(range(12))or not all(r['completed']for r in rows):raise ValueError('Every frozen object required')
 rows=sorted(rows,key=lambda x:x['index']);passed=all(r['passed']for r in rows);active=np.array([r['policy_RMSE_m']['target_variance']for r in rows]);comparisons={}
 for name in DESIGN['policies'][1:]:
  baseline=np.array([r['policy_RMSE_m'][name]for r in rows]);value=helper.cluster_comparison(active,baseline,passed=passed);envelopes=[r['empirical_numerical_envelope_m'][name]for r in rows];envelope=float(np.mean(envelopes))if all(v is not None for v in envelopes)else None;value.update(empirical_numerical_envelope_m=envelope,rigorous_bound=False,superiority_supported=bool(passed and envelope is not None and value['designed_cohort_cluster_bootstrap_95pct_CI_m'][1]+envelope<0));comparisons[name]=value
 return dict(completed=True,objects=12,numerically_passed_objects=sum(r['passed']for r in rows),all_numerical_gates_passed=passed,all_objects_retained=True,comparisons=comparisons,policy_mean_RMSE_m={name:float(np.mean([r['policy_RMSE_m'][name]for r in rows]))for name in DESIGN['policies']},baseline_mean_RMSE_m={name:float(np.mean([r['baseline_RMSE_m'][name]for r in rows]))for name in rows[0]['baseline_RMSE_m']},choices=[r['choices']for r in rows],scope=DESIGN['metric']+'; '+DESIGN['support_geometry'],population_calibration_claim=False,raw_LOCAL=True)


def run(protocol,digest):
 p=validate(protocol,digest);bundle_io.guard(RESOURCES['controller_cores']);out=io.local(ROOT,p['output']);out.mkdir(parents=True,exist_ok=False)
 import shutil
 if shutil.disk_usage(out).free<RESOURCES['disk_free_reserve_bytes']+RESOURCES['maximum_raw_bytes']:raise RuntimeError('Diskreserve+cohortwritebudget')
 io.atomic(out/'journal.json',dict(pid=os.getpid(),completed=False,status='BOOTSTRAP_BEFORE_EXPOSURE',live=[],records=[],pending_indices=list(range(12))))
 def command(i,slot):return ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','taskset','-c',','.join(map(str,RESOURCES['worker_slots'][slot])),str(ROOT/'.venv/bin/python'),'-m','experiments.run_diverse_force_cohort_v01','worker','--protocol',protocol,'--digest',digest,'--index',str(i),'--slot',str(slot)]
 try:
  records=owner.owned_jobs(command,list(range(12)),12,out,worker_wall_s=RESOURCES['worker_wall_s'],global_wall_s=RESOURCES['global_wall_s'],RSS_bytes=RESOURCES['worker_RSS_bytes'],controller_check=lambda:bundle_io.guard(RESOURCES['controller_cores']))
  rows=[json.loads((out/f'object-{i:02d}'/'summary.json').read_text())for i in range(12)];io.write_new(out/'summary.json',aggregate(rows));io.write_new(out/'execution.json',dict(completed=True,pid=os.getpid(),all_owned_workers_reaped=True,records=records,protocol_sha256=digest))
 except BaseException as error:
  io.write_new(out/'failure.json',dict(completed=False,error=repr(error),all_partial_artifacts_preserved=True,no_retry=True));raise


def supervise(protocol,digest):
 from experiments.supervise_surrogate_sbc_continuation import supervise as owned_supervise
 p=validate(protocol,digest);op=(ROOT/protocol).parent
 command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','taskset','-c','6,7',str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_diverse_force_cohort_v01','run','--protocol',protocol,'--digest',digest]
 return owned_supervise(command,op/'actual-cohort.log',op/'actual-supervisor-receipt.json',journal_path=ROOT/p['output']/'journal.json',wall_s=RESOURCES['global_wall_s'],RSS_bytes=RESOURCES['worker_RSS_bytes'])


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','worker','supervise']);a.add_argument('--protocol',required=True);a.add_argument('--digest');a.add_argument('--output');a.add_argument('--index',type=int);a.add_argument('--slot',type=int);v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output))
 elif v.mode=='worker':worker(v.protocol,v.digest,v.index,v.slot)
 elif v.mode=='run':run(v.protocol,v.digest)
 else:
  receipt=supervise(v.protocol,v.digest);print(json.dumps(receipt));raise SystemExit(0 if receipt.get('completed')else 1)
