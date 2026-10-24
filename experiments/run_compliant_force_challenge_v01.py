"""Frozen CPU-only held-force challenge; raw fields stay local.

Every posterior draw is used. Disk-backed target fields are streamed, with
resident mapped pages explicitly released after each block. No sampler runs.
"""
from pathlib import Path
import argparse, hashlib, json, mmap, os, resource, shutil, time
import numpy as np
import torch
from experiments import campaign_compliant_force_challenge_v01 as helper
from experiments import run_compliant_full_count_continuation_v02 as previous
from experiments.campaign_owned_proc import proc_identity

ROOT=previous.ROOT; io=previous.io; bundle_io=previous.bundle_io
SOURCE='experiments/run_compliant_force_challenge_v01.py'
REFERENCE_PROTOCOL='runs/operator-dev/compliant-full4D-count-only-continuation-v02/protocol.json'
REFERENCE_DIGEST='54396f69b1d19cbf967a06cdd2792051ef54b714708233934f8c526396927aab'
REFERENCE_OUTPUT='runs/gaussian-campaign-v01/compliant-full4D-count-only-continuation-v02'
RESOURCES=dict(worker_cores=[4,5],controller_cores=[6,7],threads=2,RSS_bytes=4*1024**3,worker_wall_s=1200,global_wall_s=1500,no_GPU=True,disk_free_reserve_bytes=30*1024**3)
DESIGN=dict(helper=helper.design(),all_MH_draws=True,chains=8,retained_per_chain=64000,SMC_particles=32768,SMC_seeds=[2026121101,2026121102,2026121103],noise_seeds=list(range(2026102410,2026102426)),chunk=1024,bootstrap_seed=2026102403,bootstrap_resamples=2000,Fisher_steps=[1e-4,5e-5],Fisher='Local exact finite-difference observation Jacobian at posterior intrinsic mean; trace((posterior_cov^-1+JtJ/sigma2)^-1) reduction',split_blocks='Each of eight original chains split into contiguous halves',field_agreement='Every chainhalf prediction RMS difference <=max(2e-6,.03*posterior_predictive_RMS_SD)',importance_gate='rawESS>=400 and fraction>=.05; autocorrelation-adjustedESS not established',scope='One known manufactured object and repeated observation noise only; no population calibration or nonlinear continuum claim',no_new_inference=True,no_retry=True)
DESIGN.update(moment_stability='Pre/post mean AND standard deviation; 16MH halves and three independent SMC references',acquisition_stability='All16MH half and3SMC target-variance top choices equal pooledMH; maximum score difference<=.1*max(abs(pooledscore))',superiority='All numerical gates and upper pairednoiseCI + empirical numerical envelope <0. Envelope=max advantage difference from16MHhalf and3SMC estimates; diagnostic, not rigorous bound',actuation_budget='Same net force peak, duration and scalar impulse; mechanical work and per-node force norm need not be equal')


def chunks(n,width=1024):
 for first in range(0,n,width):yield slice(first,min(n,first+width))


def release(array):
 """Flush dirty raw pages before discarding; never close a live mapping."""
 if isinstance(array,np.memmap):
  array.flush();array._mmap.madvise(mmap.MADV_DONTNEED)


def streamed_moments(target,pokes,width=1024):
 """Population posterior moments and Cov(target,observations), bounded memory."""
 n,d=target.shape;k,_,p=pokes.shape
 anchor=np.array(target[0]);mean=np.zeros(d);second=np.zeros(d);cross=np.zeros((k,d,p));ym=pokes.mean(1)
 for s in chunks(n,width):
  x=np.asarray(target[s])-anchor;mean+=x.sum(0);second+=np.square(x).sum(0)
  for action in range(k):cross[action]+=x.T@(pokes[action,s]-ym[action])
  del x;release(target)
 mean/=n;variance=positive_variance(second/n,mean);mean+=anchor;cross/=n
 scores=[]
 for action in range(k):
  y=pokes[action]-ym[action];cov=y.T@y/n+helper.original.SIGMA**2*np.eye(p)
  scores.append(float(np.sum(cross[action]*np.linalg.solve(cov,cross[action].T).T)/d))
 return mean,variance,np.array(scores)


def positive_variance(second,mean):
 value=second-mean**2;bound=64*np.finfo(float).eps*np.maximum(second,mean**2)
 if np.any(value < -bound):raise FloatingPointError('Unresolved negative anchored variance')
 return np.where(value<0,0,value)  # only explicitly bounded roundoff


def streamed_updates(target,weights,*,blocks=16,width=1024):
 """All paired updates in one streamed pass, retaining contiguous block means."""
 n,d=target.shape
 if n%blocks or weights.shape[0]!=n:raise ValueError('Complete contiguous blocks required')
 totals=np.zeros((weights.shape[1],d));seconds=np.zeros_like(totals)
 block_totals=np.zeros((blocks,weights.shape[1],d));block_seconds=np.zeros_like(block_totals);masses=np.zeros((blocks,weights.shape[1]))
 length=n//blocks;anchor=np.array(target[0])
 for block in range(blocks):
  for local in chunks(length,width):
   s=slice(block*length+local.start,block*length+local.stop);x=np.asarray(target[s])-anchor;w=weights[s]
   value=w.T@x;squared=w.T@np.square(x);totals+=value;seconds+=squared;block_totals[block]+=value;block_seconds[block]+=squared;masses[block]+=w.sum(0)
   del x;release(target)
 half=np.divide(block_totals,masses[:,:,None],out=np.zeros_like(block_totals),where=masses[:,:,None]>0)+anchor
 halfsecond=np.divide(block_seconds,masses[:,:,None],out=np.zeros_like(block_seconds),where=masses[:,:,None]>0)
 return totals+anchor,positive_variance(seconds,totals),half,masses,positive_variance(halfsecond,half-anchor)


def fisher_scores(forward,unit_mean,cov,pokes,step):
 if np.any(unit_mean-step<=0)or np.any(unit_mean+step>=1):raise ValueError('Central derivative must remain inside original prior')
 points=np.concatenate([unit_mean[None]+step*np.eye(4),unit_mean[None]-step*np.eye(4)])
 theta=helper.base.physical(points,(0,1,2,3));inverse=np.linalg.inv(cov);result=[]
 for force in pokes.values():
  y=helper.original.selected_observations(forward(theta,force));J=((y[:4]-y[4:])/(2*step)).T
  updated=np.linalg.inv(inverse+J.T@J/helper.original.SIGMA**2)
  result.append(float(np.trace(cov)-np.trace(updated)))
 return np.array(result)


def bootstrap_paired(a,b):
 difference=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(DESIGN['bootstrap_seed'])
 samples=difference[rng.integers(len(difference),size=(DESIGN['bootstrap_resamples'],len(difference)))].mean(1)
 return dict(mean_difference_m=float(difference.mean()),noise_only_95pct_CI_m=np.quantile(samples,[.025,.975]).tolist(),objects=1,noise_trials=len(difference))


def sources():
 return sorted(set(previous.sources())|{SOURCE,'experiments/campaign_compliant_force_challenge_v01.py','tests/test_run_compliant_force_challenge_v01.py','tests/test_compliant_force_challenge_v01.py','experiments/supervise_surrogate_sbc_continuation.py'})


def reference():
 p=previous.validate(REFERENCE_PROTOCOL,REFERENCE_DIGEST);out=ROOT/REFERENCE_OUTPUT
 e=json.loads((out/'execution.json').read_text());summary=json.loads((out/'summary.json').read_text())
 if not(summary['completed']and summary['passed']and e['all_owned_workers_reaped']):raise ValueError('Converged terminal reference required')
 for row in e['records']:
  if row['exit_code']!=0 or not row['reaped']or proc_identity(row['pid'])is not None:raise ValueError('Completed absent reference workers required')
 receipt=json.loads((ROOT/'runs/operator-dev/compliant-full4D-count-only-continuation-v02/actual-supervisor-receipt.json').read_text())
 if not(receipt['completed']and receipt['controller_reaped']and receipt['controller_pid']==e['pid']and receipt['controller_PID_absent']and receipt['all_observed_owned_worker_PIDs_absent']):raise ValueError('Complete exact reference ownership receipt')
 if set(receipt['observed_owned_worker_PIDs'])!={r['pid']for r in e['records']}:raise ValueError('Exact worker ownership cohort')
 for pid in [receipt['supervisor_pid'],receipt['controller_pid']]+receipt['observed_owned_worker_PIDs']:previous.absent_process_and_group(pid)
 return p,e


def prepare(protocol,output):
 dest=io.local(ROOT,protocol)
 if dest.exists()or io.local(ROOT,output).exists():raise FileExistsError('Exclusive new challenge namespace')
 p,e=reference();names=dict(reference_protocol=REFERENCE_PROTOCOL,reference_execution=REFERENCE_OUTPUT+'/execution.json',reference_summary=REFERENCE_OUTPUT+'/summary.json',reference_receipt='runs/operator-dev/compliant-full4D-count-only-continuation-v02/actual-supervisor-receipt.json',original_observation=p['inputs']['observation']['path'])
 for name,r in e['records'][0]['files'].items():names['MH/'+name]=r['path']
 for seed in DESIGN['SMC_seeds']:
  for name in previous.FILES:names[f'SMC-{seed}/'+name]=p['inputs'][f'smc-32768-{seed}/'+name]['path']
 value=dict(schema='compliant-force-challenge-v01',status='FROZEN_BEFORE_HELDOUT_CALLS',design=DESIGN,resources=RESOURCES,runtime=io.runtime(),source_sha256={n:io.sha(ROOT/n)for n in sources()},inputs={k:io.record(ROOT,ROOT/n)for k,n in names.items()},output=output,raw_LOCAL=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,value);return io.sha(dest)


def validate(protocol,digest):
 p=io.read_json(ROOT,dict(path=protocol,sha256=digest))
 if p['design']!=DESIGN or p['resources']!=RESOURCES or p['runtime']!=io.runtime()or p['source_sha256']!={n:io.sha(ROOT/n)for n in sources()}:raise ValueError('Frozen challenge identity')
 for record in p['inputs'].values():io.checked(ROOT,record)
 reference();return p


def read_saved(p,label):return bundle_io.read_bundle(ROOT,{name:p['inputs'][label+'/'+name]for name in previous.FILES})


def target_prediction(forward,theta,programs):
 return np.concatenate([forward(theta,force)[:,25:61].reshape(len(theta),-1)for force in programs.values()],axis=1)


def create_fields(forward,unit,programs,pokes,dest):
 n=len(unit);target=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float64,shape=(n,1620));observations=np.empty((len(pokes),n,12))
 for s in chunks(n):
  theta=helper.base.physical(unit[s],(0,1,2,3));target[s]=target_prediction(forward,theta,programs)
  for k,force in enumerate(pokes.values()):observations[k,s]=helper.original.selected_observations(forward(theta,force))
  release(target)
 return target,observations


def evaluate_updates(target,pokes,truth_targets,truth_pokes,choices):
 n=target.shape[0];K=len(pokes);R=len(DESIGN['noise_seeds']);weights=np.empty((n,K*R));diagnostics=[]
 for k in range(K):
  for j,seed in enumerate(DESIGN['noise_seeds']):
   # The same noise vector is shared across candidates for each paired trial.
   observation=truth_pokes[k]+np.random.default_rng(seed).normal(0,helper.original.SIGMA,12)
   w,info=helper.exact_reweight(pokes[k],observation);weights[:,k*R+j]=w;diagnostics.append(dict(action=k,noise_seed=seed,**info))
 means,variance,halves,masses,halfvariance=streamed_updates(target,weights)
 errors=np.sqrt(np.mean((means-truth_targets)**2,axis=1)).reshape(K,R)
 tolerance=np.maximum(2e-6,.03*np.sqrt(variance.mean(1)))
 split_error=np.sqrt(np.mean((halves-means[None])**2,axis=2))
 split_sd=np.sqrt(np.mean((np.sqrt(halfvariance)-np.sqrt(variance)[None])**2,axis=2))
 passed=bool(all(x['importance_gate_passed']for x in diagnostics)and np.all(split_error<=tolerance[None])and np.all(split_sd<=tolerance[None])and np.all(masses>0))
 return dict(errors=errors,means=means,variance=variance,halves=halves,halfvariance=halfvariance,masses=masses,diagnostics=diagnostics,passed=passed,zero_block_mass=bool(np.any(masses<=0)),maximum_split_half_difference_m=float(split_error.max()),maximum_split_half_SD_difference_m=float(split_sd.max()),minimum_ESS=float(min(x['raw_importance_ESS']for x in diagnostics)),minimum_ESS_fraction=float(min(x['ESS_fraction']for x in diagnostics)),policy_RMSE_m={name:errors[action].tolist()for name,action in choices.items()},uniform_random_expected_RMSE_m=errors.mean(0).tolist())


def worker(protocol,digest):
 started=time.monotonic();p=validate(protocol,digest)
 if set(os.sched_getaffinity(0))!={4,5}or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only affinity contract')
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 out=io.local(ROOT,p['output']);out.mkdir(parents=True,exist_ok=True)
 if shutil.disk_usage(out).free<RESOURCES['disk_free_reserve_bytes']+10*1024**3:raise RuntimeError('Raw field disk reserve')
 saved=read_saved(p,'MH');chain=np.asarray(saved['unit']);unit=chain.reshape(-1,4)
 if chain.shape!=(8,64000,4)or not saved['passed']:raise ValueError('Exact full saved reference')
 forward=helper.ArbitraryForceForward();programs=helper.force_programs();pokes=helper.poke_programs()
 target,y=create_fields(forward,unit,programs,pokes,out/'MH-target.npy')
 mean,var,scores=streamed_moments(target,y);cov=np.cov(unit,rowvar=False)
 _,_,pre_halves,pre_mass,pre_halfvar=streamed_updates(target,np.full((len(unit),1),1/len(unit)))
 pre_split_error=float(np.sqrt(np.mean((pre_halves[:,0]-mean)**2,axis=1)).max());pre_tolerance=max(2e-6,.03*float(np.sqrt(var.mean())))
 pre_sd_error=float(np.sqrt(np.mean((np.sqrt(pre_halfvar[:,0])-np.sqrt(var))**2,axis=1)).max())
 pre_scores=[]
 for block in range(16):
  s=slice(block*32000,(block+1)*32000);_,_,score=streamed_moments(target[s],y[:,s]);pre_scores.append(score)
 pre_score_passed=bool(all(np.argmax(score)==np.argmax(scores)and np.max(abs(score-scores))<=.1*max(abs(scores).max(),np.finfo(float).tiny)for score in pre_scores))
 pre_split_passed=bool(pre_split_error<=pre_tolerance and pre_sd_error<=pre_tolerance and pre_score_passed and np.all(pre_mass>0))
 fisher=[fisher_scores(forward,unit.mean(0),cov,pokes,h)for h in DESIGN['Fisher_steps']]
 if np.argmax(fisher[0])!=np.argmax(fisher[1]):raise ValueError('Fisher choice derivative refinement failed')
 choices=dict(target_variance=int(np.argmax(scores)),fixed_first=0,seeded_random=int(np.random.default_rng(2026102401).integers(len(pokes))),local_Fisher=int(np.argmax(fisher[-1])),parameter_information=int(np.argmax(helper.parameter_information(unit,y))))
 # An exclusive decision artifact precedes all truth, heldout labels and noise.
 io.write_new(out/'decisions-before-truth.json',dict(protocol_sha256=digest,choices=choices,target_scores=scores.tolist(),Fisher_scores=[x.tolist()for x in fisher],no_truth_or_realized_poke_observations_accessed=True))
 truth=helper.original.TRUTH;truth_target=target_prediction(forward,truth[None],programs)[0]
 oracle=np.concatenate([helper.independent_numpy_forward(truth,force)[25:61].reshape(-1)for force in programs.values()])
 if np.max(abs(oracle-truth_target))>1e-12:raise ValueError('Independent oracle parity')
 truth_pokes=np.stack([helper.original.selected_observations(forward(truth[None],force))[0]for force in pokes.values()])
 physical=helper.base.physical(unit,(0,1,2,3));loglik=np.asarray(saved['MH']['log_likelihood']).reshape(-1)
 baseline_theta=dict(nominal_prior_center=helper.base.physical(np.full((1,4),.5),(0,1,2,3)),posterior_MAP=physical[np.argmax(loglik)][None],posterior_mean_physical_plugin=physical.mean(0)[None])
 baselines={name:target_prediction(forward,theta,programs)[0]for name,theta in baseline_theta.items()}
 hold=np.zeros((61,5,3))
 for s in chunks(len(unit)):hold+=forward(physical[s],'A').sum(0)/len(unit)
 baselines.update(posterior_predictive_mean=mean,hold_posterior_mean_A_frame24=np.tile(hold[24].reshape(-1),108),zero_displacement=np.zeros(1620),exact_oracle=oracle)
 updates=evaluate_updates(target,y,truth_target,truth_pokes,choices)
 np.savez(out/'MH-updates.npz',**{k:updates[k]for k in ('errors','means','variance','halves','halfvariance','masses')})
 io.write_new(out/'MH-importance-diagnostics.json',updates['diagnostics'])
 smc_comparisons=[];smc_errors=[]
 for seed in DESIGN['SMC_seeds']:
  reference=read_saved(p,f'SMC-{seed}');smc_unit=np.asarray(reference['unit'])
  st,sy=create_fields(forward,smc_unit,programs,pokes,out/f'SMC-{seed}-target.npy');sm,sv,ss=streamed_moments(st,sy)
  su=evaluate_updates(st,sy,truth_target,truth_pokes,choices)
  tolerance=max(2e-6,.03*float(np.sqrt(var.mean())));post_tolerance=np.maximum(2e-6,.03*np.sqrt(updates['variance'].mean(1)))
  before=float(np.sqrt(np.mean((sm-mean)**2)));after=np.sqrt(np.mean((su['means']-updates['means'])**2,axis=1))
  pre_sd=float(np.sqrt(np.mean((np.sqrt(sv)-np.sqrt(var))**2)));post_sd=np.sqrt(np.mean((np.sqrt(su['variance'])-np.sqrt(updates['variance']))**2,axis=1));ranking=bool(np.argmax(ss)==np.argmax(scores))
  score_difference=float(np.max(abs(ss-scores))/max(abs(scores).max(),np.finfo(float).tiny))
  row=dict(seed=seed,pre_mean_RMS_difference_m=before,pre_SD_RMS_difference_m=pre_sd,post_max_mean_RMS_difference_m=float(after.max()),post_max_SD_RMS_difference_m=float(post_sd.max()),tolerance_m=tolerance,post_tolerance_min_m=float(post_tolerance.min()),acquisition_scores=ss.tolist(),acquisition_same_top_action=ranking,acquisition_max_relative_score_difference=score_difference,raw_importance_gates_passed=all(v['importance_gate_passed']for v in su['diagnostics']),passed=bool(before<=tolerance and pre_sd<=tolerance and ranking and score_difference<=.1 and np.all(after<=post_tolerance)and np.all(post_sd<=post_tolerance)and all(v['importance_gate_passed']for v in su['diagnostics'])))
  smc_comparisons.append(row);smc_errors.append(su['errors'].copy());np.savez(out/f'SMC-{seed}-updates.npz',means=su['means'],errors=su['errors']);del st,sy,su
 target_errors={name:float(np.sqrt(np.mean((value-truth_target)**2)))for name,value in baselines.items()}
 per_force={force:{name:float(np.sqrt(np.mean((value[i*540:(i+1)*540]-truth_target[i*540:(i+1)*540])**2)))for name,value in baselines.items()}for i,force in enumerate(programs)}
 active=updates['errors'][choices['target_variance']]
 comparisons={name:bootstrap_paired(active,np.asarray(values))for name,values in updates['policy_RMSE_m'].items()if name!='target_variance'}
 comparisons['uniform_random_expected']=bootstrap_paired(active,updates['errors'].mean(0))
 half_errors=np.sqrt(np.mean((updates['halves']-truth_target)**2,axis=2)).reshape(16,6,16)
 numerical_passed=bool(pre_split_passed and updates['passed']and all(x['passed']for x in smc_comparisons))
 for name,comparison in comparisons.items():
  active_index=choices['target_variance'];reference_differences=[]
  for errors in [*smc_errors,*half_errors]:
   baseline=errors.mean(0)if name=='uniform_random_expected'else errors[choices[name]]
   reference_differences.append(float(np.mean(errors[active_index]-baseline)))
  envelope=max(abs(value-comparison['mean_difference_m'])for value in reference_differences)
  comparison.update(empirical_numerical_envelope_m=envelope,rigorous_numerical_bound=False,exact_same_action=bool(name!='uniform_random_expected'and choices[name]==active_index),conditional_advantage_supported=bool(numerical_passed and comparison['noise_only_95pct_CI_m'][1]+envelope<0))
 results=dict(completed=True,prediction_RMSE_m=target_errors,per_force_RMSE_m=per_force,choices=choices,policy_mean_RMSE_m={name:float(np.mean(value))for name,value in updates['policy_RMSE_m'].items()},uniform_random_expected_mean_RMSE_m=float(updates['errors'].mean()),paired_noise_only_comparisons=comparisons,pre_split_half_passed=pre_split_passed,pre_split_half_max_difference_m=pre_split_error,importance_and_split_half_passed=updates['passed'],minimum_ESS=updates['minimum_ESS'],minimum_ESS_fraction=updates['minimum_ESS_fraction'],maximum_split_half_difference_m=updates['maximum_split_half_difference_m'],SMC_comparisons=smc_comparisons,reference_agreement_passed=bool(pre_split_passed and updates['passed']and all(x['passed']for x in smc_comparisons)),worker_wall_s=time.monotonic()-started,worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope=DESIGN['scope'],raw_LOCAL=True)
 results.update(pre_split_half_SD_max_difference_m=pre_sd_error,pre_acquisition_stable=pre_score_passed,pre_acquisition_max_relative_score_difference=float(max(np.max(abs(v-scores))/max(abs(scores).max(),np.finfo(float).tiny)for v in pre_scores)),post_split_half_SD_max_difference_m=updates['maximum_split_half_SD_difference_m'],zero_chainhalf_weight_mass=updates['zero_block_mass'],SMC_arbitrary_particle_halves_not_chain_diagnostics=True,noise_CI_excludes_posterior_MC_uncertainty=True,empirical_envelope_not_rigorous_bound=True)
 predictive_sd=np.sqrt(updates['variance'].mean(1)).reshape(6,16)
 results.update(before_predictive_RMS_SD_m=float(np.sqrt(var.mean())),policy_mean_predictive_RMS_SD_m={name:float(predictive_sd[action].mean())for name,action in choices.items()},uncertainty_narrowing_alone_not_benefit=True)
 io.write_new(out/'summary.json',results)


def run(protocol,digest):
 p=validate(protocol,digest)
 if set(os.sched_getaffinity(0))!={6,7}:raise RuntimeError('Controller CPU affinity')
 out=io.local(ROOT,p['output']);out.mkdir(parents=True,exist_ok=False)
 io.atomic(out/'journal.json',dict(pid=os.getpid(),completed=False,status='BOOTSTRAP_BEFORE_EXPOSURE',live=[],records=[],pending_indices=[0]))
 command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','taskset','-c','4,5',str(ROOT/'.venv/bin/python'),'-m','experiments.run_compliant_force_challenge_v01','worker','--protocol',protocol,'--digest',digest]
 try:
  records=previous.owner.owned_jobs(lambda i,slot:command,[0],1,out,worker_wall_s=1200,global_wall_s=1500,RSS_bytes=RESOURCES['RSS_bytes'])
  io.write_new(out/'execution.json',dict(completed=True,pid=os.getpid(),all_owned_workers_reaped=True,records=records,protocol_sha256=digest))
 except BaseException as error:
  io.write_new(out/'failure.json',dict(completed=False,error=repr(error),journal=io.record(ROOT,out/'journal.json'),no_retry=True));raise


def supervise(protocol,digest):
 from experiments.supervise_surrogate_sbc_continuation import supervise as owned_supervise
 p=validate(protocol,digest);op=(ROOT/protocol).parent
 command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','taskset','-c','6,7',str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_compliant_force_challenge_v01','run','--protocol',protocol,'--digest',digest]
 receipt=owned_supervise(command,op/'actual-challenge.log',op/'actual-supervisor-receipt.json',journal_path=ROOT/p['output']/'journal.json',wall_s=1500,RSS_bytes=RESOURCES['RSS_bytes'])
 print(json.dumps(receipt));return receipt


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker','supervise']);parser.add_argument('--protocol',required=True);parser.add_argument('--digest');parser.add_argument('--output');args=parser.parse_args()
 if args.mode=='prepare':print(prepare(args.protocol,args.output))
 elif args.mode=='run':run(args.protocol,args.digest)
 elif args.mode=='worker':worker(args.protocol,args.digest)
 else:
  receipt=supervise(args.protocol,args.digest);raise SystemExit(0 if receipt.get('completed')else 1)
