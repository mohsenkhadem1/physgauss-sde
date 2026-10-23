"""One frozen full4D count-only continuation; all earlier stages reused by hash.

No new prior, likelihood, observation, warmup, seeds, proposal or independent
chains. Previous numerical negatives remain immutable. No retry or count tuning.
"""
from pathlib import Path
import argparse,hashlib,json,os,resource,signal,time
signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGALRM})
import numpy as np
import torch
from experiments import campaign_compliant_full_count_continuation_v01 as model
from experiments import run_compliant_count_continuation_v01 as prior
from experiments import campaign_compliant_observation_forward as observation
from experiments.campaign_owned_proc import proc_identity
io=prior.io;ROOT=prior.ROOT;owner=prior.owner;bundle_io=prior.bundle_io;FILES=prior.FILES
SOURCE='experiments/run_compliant_full_count_continuation_v02.py'
PREVIOUS_PROTOCOL='runs/operator-dev/compliant-count-resolution-continuation-v01/protocol.json'
PREVIOUS_DIGEST='b61d0213a88b82c02f6dbd1849e958912136abf778e9837d43f85178c4022f1a'
PREVIOUS_OUTPUT='runs/gaussian-campaign-v01/compliant-count-resolution-continuation-v01'
RESOURCES=dict(controller_cores=[10,11],worker_slots=[[12,13]],threads=2,worker_RSS_bytes=2*1024**3,no_GPU=True,worker_wall_s=1200,global_wall_s=1500,disk_free_reserve_bytes=30*1024**3)
DESIGN=dict(previous_protocol_sha256=PREVIOUS_DIGEST,dimensions=[0,1,2,3],MH=dict(chains=8,seed=2026121120,old_warmup=1000,new_warmup=0,previous_retained=4000,additional_retained=60000,total_retained=64000,new_seed=False,new_proposal=False,new_independent_chains=False,exact_RNG_only_replay_steps=5000),gates=prior.DESIGN['gates'],MCSE=prior.DESIGN['MCSE'],supplementary_time_batches=[800,1600],reuse_2D_qualification=True,reuse_joint_diagnostics=True,reuse_all_six_SMC=True,reuse_all_fifteen_SMC_pairwise=True,no_optional_stopping=True,no_retry=True,new_observations=False,same_original_unit_uniform_intrinsic_prior=True,physical_calibration=False)


def sources():
 return sorted(set(prior.sources())|{SOURCE,'experiments/campaign_compliant_full_count_continuation_v01.py','experiments/supervise_compliant_full_count_continuation_v02.py','tests/test_compliant_full_count_continuation_v02.py'})


def absent_process_and_group(pid):
 if proc_identity(pid)is not None:raise ValueError('Predecessor owned PID must be absent')
 try:os.killpg(pid,0)
 except ProcessLookupError:return True
 raise ValueError('Predecessor owned process group must be absent')


def predecessor():
 p=prior.validate(PREVIOUS_PROTOCOL,PREVIOUS_DIGEST);out=ROOT/PREVIOUS_OUTPUT
 e=json.loads((out/'execution.json').read_text());s=json.loads((out/'summary.json').read_text())
 expected={'qualify2D','jointDiagnostics','fullMH'}|{'smc-'+str(n)+'-'+str(seed)for n in prior.COUNTS for seed in prior.SEEDS}
 if e.get('completed')is not True or e.get('all_owned_workers_reaped')is not True or len(e.get('records',[]))!=9 or {r['job']for r in e['records']}!=expected:raise ValueError('Exact completed predecessor stages')
 for row in e['records']:
  if row['exit_code']!=0 or row['reaped']is not True:raise ValueError('All predecessor workers must be reaped successfully')
  for record in row['files'].values():io.checked(ROOT,record)
 if s.get('completed')is not True or s.get('passed')is not False or s.get('qualification_passed')is not True or s.get('full4D_started')is not True or s.get('profile_convergence_complete')is not True or s['SMC_logZ_range']>.05 or not all(r['passed']for r in s['pairwise_comparisons']+s['MH_vs_SMC']):raise ValueError('Reuse exact numerical precision negative with all prior agreement gates passed')
 receipt_path=ROOT/'runs/operator-dev/compliant-count-resolution-continuation-v01/actual-supervisor-receipt.json'
 receipt=json.loads(receipt_path.read_text())
 if receipt.get('completed')is not True or receipt.get('controller_reaped')is not True or receipt.get('controller_PID_absent')is not True or receipt.get('all_observed_owned_worker_PIDs_absent')is not True or receipt.get('controller_pid')!=e['pid']or set(receipt.get('observed_owned_worker_PIDs',[]))!={r['pid']for r in e['records']}:raise ValueError('Exact completed predecessor supervisor/controller/worker identity')
 for pid in [receipt['supervisor_pid'],receipt['controller_pid']]+receipt['observed_owned_worker_PIDs']:absent_process_and_group(pid)
 return p,e,s


def old_files(e,job):return next(row['files']for row in e['records']if row['job']==job)


def state_identity(saved):
 r=saved['MH'];rng=model.recover_rng(r['seed'],r['initial_states'],r['warmup']+r['draws'])
 return dict(RNG_state=rng.bit_generator.state,previous_final_unit_sha256=hashlib.sha256(saved['unit'][:,-1].copy().tobytes()).hexdigest(),previous_final_loglik_sha256=hashlib.sha256(r['log_likelihood'][:,-1].copy().tobytes()).hexdigest(),RNG_only_replay_steps=5000,no_posterior_calls=True,new_initialization=False)


def prepare(protocol,output):
 dest=io.local(ROOT,protocol)
 if dest.exists()or io.local(ROOT,output).exists():raise FileExistsError('Exclusive new count-only continuation namespace')
 p,e,s=predecessor();saved=bundle_io.read_bundle(ROOT,old_files(e,'fullMH'))
 if saved.get('passed')is not False or saved['MC_diagnostics_passed']is not False:raise ValueError('Retain old fullMH precision negative')
 dest.parent.mkdir(parents=True,exist_ok=True);identity_path=dest.parent/'continuation-identity.json';io.write_new(identity_path,state_identity(saved))
 names=dict(previous_supervisor_receipt='runs/operator-dev/compliant-count-resolution-continuation-v01/actual-supervisor-receipt.json',previous_protocol=PREVIOUS_PROTOCOL,previous_execution=PREVIOUS_OUTPUT+'/execution.json',previous_summary=PREVIOUS_OUTPUT+'/summary.json',previous_qualification_seal=PREVIOUS_OUTPUT+'/qualification-seal.json',previous_SMC_seal=PREVIOUS_OUTPUT+'/SMC-seal.json',previous_proposal=PREVIOUS_OUTPUT+'/proposal-freezes/fullMH.npz',previous_freeze=PREVIOUS_OUTPUT+'/proposal-freezes/fullMH.json',observation=p['inputs']['observation']['path'],identity=str(identity_path.relative_to(ROOT)))
 for row in e['records']:
  for name,record in row['files'].items():names[row['job']+'/'+name]=record['path']
 value=dict(schema='COMPLIANT-full4D-count-only-continuation-v02',status='FROZEN_BEFORE_ANY_NEW_POSTERIOR_RUN',design=DESIGN,resources=RESOURCES,runtime=io.runtime(),source_sha256={n:io.sha(ROOT/n)for n in sources()},inputs={k:io.record(ROOT,ROOT/n)for k,n in names.items()},output=output,all_old_failures_preserved=True,raw_LOCAL=True)
 io.write_new(dest,value);return io.sha(dest)


def validate(protocol,digest):
 p=io.read_json(ROOT,dict(path=protocol,sha256=digest))
 if p.get('schema')!='COMPLIANT-full4D-count-only-continuation-v02'or p.get('status')!='FROZEN_BEFORE_ANY_NEW_POSTERIOR_RUN'or p.get('design')!=DESIGN or p.get('resources')!=RESOURCES or p.get('runtime')!=io.runtime()or p.get('source_sha256')!={n:io.sha(ROOT/n)for n in sources()}:raise ValueError('Exact frozen count-only continuation')
 for r in p['inputs'].values():io.checked(ROOT,r)
 predecessor();io.local(ROOT,p['output']);return p


def files(p,job):return {name:p['inputs'][job+'/'+name]for name in FILES}


def worker(protocol,digest):
 start=time.monotonic();p=validate(protocol,digest);bundle_io.guard(RESOURCES['worker_slots'][0]);torch.set_num_threads(2)
 previous=bundle_io.read_bundle(ROOT,files(p,'fullMH'));qualification=bundle_io.read_bundle(ROOT,files(p,'qualify2D'))
 if qualification.get('passed')is not True or qualification['long_batch_MC']['passed']is not True:raise ValueError('Original and supplemental qualified2D barrier')
 pilot=bundle_io.read_bundle(ROOT,files(p,'smc-32768-2026121103'))['unit']
 with np.load(io.checked(ROOT,p['inputs']['previous_proposal']),allow_pickle=False)as z:
  np.testing.assert_array_equal(z['mean'],previous['MH']['proposal_mean']);np.testing.assert_array_equal(z['covariance'],previous['MH']['proposal_covariance'])
 np.testing.assert_array_equal(pilot.mean(0),previous['MH']['proposal_mean']);np.testing.assert_array_equal(np.cov(pilot,rowvar=False),previous['MH']['proposal_covariance'])
 identity=io.read_json(ROOT,p['inputs']['identity'])
 if state_identity(previous)!=identity:raise ValueError('Exact prefrozen laststates and RNG identity')
 with np.load(io.checked(ROOT,p['inputs']['observation']),allow_pickle=False)as z:obs=z['observation'].copy()
 forward=observation.ObservationOnlyForward();out=io.local(ROOT,p['output'])
 def checkpoint(state):
  if state['RNG_state']!=identity['RNG_state']or hashlib.sha256(state['unit'].tobytes()).hexdigest()!=identity['previous_final_unit_sha256']or hashlib.sha256(state['log_likelihood'].tobytes()).hexdigest()!=identity['previous_final_loglik_sha256']:raise ValueError('Exact frozen initial continuation state')
  dest=out/'continuation-checkpoints';dest.mkdir();io.codec.exclusive_npz(dest/'start.npz',{k:state[k]for k in ('unit','log_likelihood','proposal_mean','proposal_covariance')});io.write_new(dest/'start.json',dict(protocol_sha256=digest,arrays=io.record(ROOT,dest/'start.npz'),RNG_state=state['RNG_state'],before_any_new_likelihood=True,previous_files=files(p,'fullMH')))
 chain,report=model.continue_mh(lambda x:forward.log_likelihood(x,(0,1,2,3),obs),previous,additional_draws=60000,checkpoint=checkpoint)
 np.testing.assert_array_equal(chain[:,:4000],previous['unit']);np.testing.assert_array_equal(report['log_likelihood'][:,:4000],previous['MH']['log_likelihood'])
 result=prior.summarize_MH(forward,chain,report,(0,1,2,3));result['long_batch_MC']=model.long_batch_mc(chain);result['passed']=bool(result['MC_diagnostics_passed']and result['long_batch_MC']['passed']);result.update(completed=True,job='fullMH',protocol=dict(path=protocol,sha256=digest),old_prefix_bit_identical=True,worker_wall_s=time.monotonic()-start,worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 return bundle_io.bundle(ROOT,out/'jobs/fullMH',result)


def run(protocol,digest):
 p=validate(protocol,digest);bundle_io.guard(RESOURCES['controller_cores']);out=io.local(ROOT,p['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();records=[]
 io.atomic(out/'journal.json',dict(pid=os.getpid(),completed=False,status='BOOTSTRAP_BEFORE_EXPOSURE',live=[],records=[],pending_indices=[0]))
 def command(i,slot):return ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','taskset','-c','12,13',str(ROOT/'.venv/bin/python'),'-m','experiments.run_compliant_full_count_continuation_v02','worker','--protocol',protocol,'--digest',digest]
 try:
  records=owner.owned_jobs(command,[0],1,out,worker_wall_s=1200,global_wall_s=1500,RSS_bytes=RESOURCES['worker_RSS_bytes'],controller_check=lambda:bundle_io.guard(RESOURCES['controller_cores']))
  for r in records:r.update(job='fullMH',files={name:io.record(ROOT,out/'jobs/fullMH'/name)for name in FILES})
  mh=bundle_io.read_bundle(ROOT,records[0]['files']);old_summary=io.read_json(ROOT,p['inputs']['previous_summary']);cross=[]
  for n in prior.COUNTS:
   for seed in prior.SEEDS:
    cached=bundle_io.read_bundle(ROOT,files(p,'smc-'+str(n)+'-'+str(seed)));cross.append(dict(particles=n,seed=seed,MH_seed=2026121120,**model.compare(cached,mh)))
  passed=bool(mh['passed']and all(row['passed']for row in cross)and old_summary['SMC_logZ_range']<=.05 and all(row['passed']for row in old_summary['pairwise_comparisons']))
  result=dict(completed=True,passed=passed,full4D_started_previously=True,qualification_reused_passed=True,jointDiagnostics_reused=True,six_SMC_reused=True,fifteen_SMC_pairwise_reused=True,SMC_logZ_range_reused=old_summary['SMC_logZ_range'],MH_vs_SMC=cross,MH_original_MC_passed=mh['MC_diagnostics_passed'],MH_long_batch_MC_passed=mh['long_batch_MC']['passed'],MH_max_CDF_MCSE=float(np.max(mh['CDF_batch_MCSE'])),MH_max_quantile_MCSE=max((v for row in mh['quantile_MCSE']for v in row if v is not None),default=None),MH_long_max_CDF_MCSE=float(np.max(mh['long_batch_MC']['CDF_batch_MCSE'])),MH_long_max_quantile_MCSE=max((v for row in mh['long_batch_MC']['quantile_MCSE']for v in row if v is not None),default=None),MH_field_MCSE_passed=mh['field_MCSE_passed'],MH_function_relative_MCSE=mh['maximum_function_relative_MCSE'],old_prefix_bit_identical=mh['old_prefix_bit_identical'],worker_wall_s=mh['worker_wall_s'],worker_peak_RSS_bytes=mh['worker_peak_RSS_bytes'],all_old_failures_preserved=True,raw_LOCAL=True,physical_calibration=False,structural_nonidentifiability_claim=False)
  io.write_new(out/'summary.json',result);io.atomic(out/'execution.json',dict(completed=True,pid=os.getpid(),all_owned_workers_reaped=True,records=records,protocol=dict(path=protocol,sha256=digest),wall_s=time.monotonic()-start))
 except BaseException as error:
  io.write_new(out/'failure.json',dict(completed=False,error=repr(error),records=records,active_stage_journal=io.record(ROOT,out/'journal.json'),all_completed_and_partial_jobs_preserved=True,no_retry=True));raise


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run','worker'));a.add_argument('--protocol',required=True);a.add_argument('--digest');a.add_argument('--output');v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output))
 elif v.mode=='worker':worker(v.protocol,v.digest)
 else:run(v.protocol,v.digest)
