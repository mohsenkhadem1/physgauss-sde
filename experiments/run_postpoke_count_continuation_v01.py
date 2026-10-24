"""One predeclared32768-particle diagnostic; archived8192 negatives preserved.

All12 selected actions/noise0 are fixed by the original cohort. Reuse saved
qualified joint MH and135prediction diagnostics; only6SMC cohorts rerun at4xN.
"""
from pathlib import Path
import argparse,json,os,resource,shutil,time
import numpy as np
import torch
from experiments import campaign_postpoke_count_continuation_v01 as helper
from experiments import run_postpoke_refresh_v02 as previous
io=previous.io;ROOT=previous.ROOT;owner=previous.owner;model=helper.model
SOURCE='experiments/run_postpoke_count_continuation_v01.py'
PREVIOUS_PROTOCOL='runs/operator-dev/postpoke-refresh-v02/protocol.json'
PREVIOUS_DIGEST='151ed2fa82595ea9e08c245019139c9c1f1a266844311bd89285bdd26eddf4cf'
PREVIOUS_OUTPUT='runs/gaussian-campaign-v01/postpoke-refresh-v02'
RESOURCES=dict(previous.RESOURCES,controller_cores=[16],worker_slots=[[i]for i in range(12)],global_wall_s=3000)
DESIGN=dict(schema='postpoke-count-continuation-v01',previous_protocol_sha256=PREVIOUS_DIGEST,objects=list(range(12)),particles=32768,old_particles=8192,counts_only=True,resource_allocation_changed_prospectively=True,same_observations=True,same_actions_and_noise_index0=True,same_bridge_fresh_and_subsample_seeds=True,same_prior_and_likelihood=True,same_fixed_prefix_proposal=True,same16_bridge_and8_fresh_rejuvenation_steps=True,reuse_MH_8x64000=True,new_MH_draws=0,reuse135_prediction_moments_and_MCSE_bitexact=True,reuse_MH_mixing_quantile_function_MCSE_diagnostics_bitexact=True,MH_per_object_pass_required_before_any_new_SMC=True,archival_gap='Original18function mean/std were not persisted. Replay onlyfixture12 andoriginalAendnode1/Bendnode4 sixfrom same savedMHstates, without new inference',gates=previous.DESIGN['gates'],all15SMC_pairs_and6vs_savedMH=True,bridge_replicates_share_fixed_prefix_seed3_proposal=True,fresh_joint3_genuinely_independent=True,no_seed_search=True,no_retry=True,no_object_drop=True,old8192_numerical_negatives_preserved=True,scope=previous.DESIGN['scope'],raw_LOCAL=True)


def sources():
 return sorted(set(previous.sources())|{SOURCE,'experiments/campaign_postpoke_count_continuation_v01.py','tests/test_postpoke_count_continuation_v01.py'})


def terminal_gate(execution,receipt,summary,rows,absent):
 if not(execution.get('completed')and execution.get('all_owned_workers_reaped')and execution.get('protocol_sha256')==PREVIOUS_DIGEST and len(execution.get('records',[]))==12 and {r['object_index']for r in execution['records']}==set(range(12))):raise ValueError('Exact all12 completed original execution required')
 if not all(r.get('reaped')and r.get('exit_code')==0 for r in execution['records']):raise ValueError('All12 original workers successfully reaped')
 if not(receipt.get('completed')and receipt.get('controller_reaped')and receipt.get('controller_PID_absent')and receipt.get('all_observed_owned_worker_PIDs_absent')and receipt.get('controller_pid')==execution.get('pid')and len(receipt.get('observed_owned_worker_PIDs',[]))==12 and set(receipt['observed_owned_worker_PIDs'])=={r['pid']for r in execution['records']}):raise ValueError('Exact completed original ownership receipt required')
 if not(summary.get('completed')and summary.get('objects')==12 and summary.get('all12_retained')):raise ValueError('All12 original numerical outcomes retained')
 if len(rows)!=12 or {r['index']for r in rows}!=set(range(12))or not all(r.get('completed')and r.get('MH_passed')and r.get('observation_replay_passed')for r in rows):raise ValueError('All12 saved64k MH references must qualify before any new SMC')
 for pid in[receipt['supervisor_pid'],receipt['controller_pid']]+receipt['observed_owned_worker_PIDs']:absent(pid)
 return True


def predecessor():
 p=previous.validate(PREVIOUS_PROTOCOL,PREVIOUS_DIGEST);out=ROOT/PREVIOUS_OUTPUT
 execution=json.loads((out/'execution.json').read_text());receipt=json.loads((ROOT/'runs/operator-dev/postpoke-refresh-v02/actual-supervisor-receipt.json').read_text());summary=json.loads((out/'summary.json').read_text());rows=[json.loads((out/f'object-{i:02d}'/'summary.json').read_text())for i in range(12)]
 terminal_gate(execution,receipt,summary,rows,previous.previous.old.previous.absent_process_and_group)
 return p,execution,receipt


def prepare(protocol,output):
 dest=io.local(ROOT,protocol)
 if dest.exists()or io.local(ROOT,output).exists():raise FileExistsError('Exclusive count-only namespace')
 old,execution,receipt=predecessor();names=dict(old_protocol=PREVIOUS_PROTOCOL,old_execution=PREVIOUS_OUTPUT+'/execution.json',old_summary=PREVIOUS_OUTPUT+'/summary.json',old_receipt='runs/operator-dev/postpoke-refresh-v02/actual-supervisor-receipt.json')
 for i in range(12):
  root=PREVIOUS_OUTPUT+f'/object-{i:02d}/'
  for key,file in dict(summary='summary.json',observations='same-observations-LOCAL.npz',MH='joint-MH-LOCAL.npz',MH_prediction='joint-prediction-LOCAL.npz',MH_diagnostics='numerical-diagnostics-LOCAL.json',MH_proposal='joint-MH-proposal-LOCAL.npz',bridge_proposal='bridge-proposal-LOCAL.npz',observation_replay='observation-replay.json').items():names[f'object-{i}/'+key]=root+file
 p=dict(schema='postpoke-count-continuation-v01',status='FROZEN_BEFORE_ANY_32768_POSTERIOR_CALLS',design=DESIGN,resources=RESOURCES,runtime=io.runtime(),source_sha256={n:io.sha(ROOT/n)for n in sources()},inputs={k:io.record(ROOT,ROOT/n)for k,n in names.items()},original_frozen_inputs=old['inputs'],output=output,raw_LOCAL=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest,index=None):
 p=io.read_json(ROOT,dict(path=protocol,sha256=digest))
 if p['design']!=DESIGN or p['resources']!=RESOURCES or p['runtime']!=io.runtime()or p['source_sha256']!={n:io.sha(ROOT/n)for n in sources()}:raise ValueError('Frozen count-only design/resources/source/runtime')
 for name,r in p['inputs'].items():
  if index is None or not name.startswith('object-')or name.startswith(f'object-{index}/'):io.checked(ROOT,r)
 for name,r in p['original_frozen_inputs'].items():
  if index is None or not name.startswith('object-')or name.startswith(f'object-{index}/'):io.checked(ROOT,r)
 predecessor();return p


def load(p,index,name):return io.checked(ROOT,p['inputs'][f'object-{index}/'+name])


def worker(protocol,digest,index,slot):
 start=time.monotonic();p=validate(protocol,digest,index);model.guard(RESOURCES['worker_slots'][slot]);torch.set_num_threads(1);torch.set_num_interop_threads(1)
 out=io.local(ROOT,p['output'])/f'object-{index:02d}';out.mkdir(parents=True,exist_ok=False);old=json.loads(load(p,index,'summary').read_text())
 with np.load(load(p,index,'observations'),allow_pickle=False)as z:observed_A=z['prefix'].copy();observed_poke=z['poke'].copy()
 force=list(model.cohort.old.poke_programs().values())[old['action']];forward=model.JointObservationForward(force,observed_A,observed_poke);fullforward=model.cohort.old.ArbitraryForceForward()
 with np.load(load(p,index,'MH'),allow_pickle=False)as z:
  chain=z['unit'].copy();saved_log_likelihood=z['log_likelihood'].copy()
 if saved_log_likelihood.shape!=(8,64000)or not np.isfinite(saved_log_likelihood).all():raise ValueError('Finite exact saved joint likelihood chains required')
 del saved_log_likelihood
 if chain.shape!=(8,64000,4)or not old['MH_passed']:raise ValueError('Qualified exact old8x64k jointMH required')
 with np.load(load(p,index,'MH_prediction'),allow_pickle=False)as z:prediction={name:z[name].copy()for name in['mean','variance','mean_MCSE','SD_MCSE']}
 reconstruction=helper.replay_function_moments(fullforward,chain)
 io.codec.exclusive_npz(out/'saved-MH-function-replay-LOCAL.npz',dict(function_mean=reconstruction['function_mean'],function_std=reconstruction['function_std']))
 io.write_new(out/'reuse-and-replay.json',dict(saved_MH=p['inputs'][f'object-{index}/MH'],saved_prediction=p['inputs'][f'object-{index}/MH_prediction'],saved_diagnostics=p['inputs'][f'object-{index}/MH_diagnostics'],function_replay=io.record(ROOT,out/'saved-MH-function-replay-LOCAL.npz'),new_MH_draws=0,original_prediction_arrays_unchanged=True))
 prediction.update(function_mean=reconstruction['function_mean'],function_std=reconstruction['function_std']);mh=previous.summarize(chain,prediction);del chain
 oldp=io.read_json(ROOT,p['inputs']['old_protocol']);clouds=[]
 for seed in previous.DESIGN['prefix_SMC_seeds']:
  with np.load(io.checked(ROOT,oldp['inputs'][f'object-{index}/SMC-{seed}']),allow_pickle=False)as z:clouds.append((z['unit'].copy(),z['log_likelihood'].copy()))
 with np.load(load(p,index,'bridge_proposal'),allow_pickle=False)as z:mean=z['mean'].copy();cov=z['covariance'].copy()
 values=[];bridgeZ=[];freshZ=[]
 for j,(cloud,oldll)in enumerate(clouds):
  if cloud.shape!=(32768,4):raise ValueError('Original prefix fullcloud')
  ids=np.random.default_rng(previous.DESIGN['subsample_seeds'][index][j]).choice(32768,size=32768,replace=False);initial=cloud[ids];a,_=forward.log_components(initial)
  if np.max(abs(a-oldll[ids]))>1e-9:raise ValueError('Unchanged prefix normalized likelihood')
  r=model.bridge_smc(initial,forward.log_components,mean,cov,seed=previous.DESIGN['bridge_seeds'][index][j],rejuvenation_steps=16);bridgeZ.append(r.log_conditional_evidence_estimate);values.append(previous.summarize(r.unit,model.prediction_summary(fullforward,r.unit)))
  io.codec.exclusive_npz(out/f'bridge-{j}-LOCAL.npz',dict(unit=r.unit,prefix_log_likelihood=r.prefix_log_likelihood,poke_log_likelihood=r.poke_log_likelihood));io.write_new(out/f'bridge-{j}-diagnostics.json',io.jsonable(dict(stages=r.stages,seed=r.seed,log_conditional_evidence_estimate=r.log_conditional_evidence_estimate,approximate_prefix_initialization=True)))
 for j,seed in enumerate(previous.DESIGN['fresh_seeds'][index]):
  r=previous.tempered_smc(model.cohort.old.base.BoxPrior(4),lambda state:forward.joint_log_likelihood(state.coordinates),particles=32768,seed=seed,ess_fraction=.8,rejuvenation_steps=8,proposal_scale=.08,independence_probability=.25,max_stages=64,periodic_indices=())
  unit=r.state.coordinates;freshZ.append(float(r.log_evidence_estimate));values.append(previous.summarize(unit,model.prediction_summary(fullforward,unit)))
  io.codec.exclusive_npz(out/f'fresh-{j}-LOCAL.npz',dict(unit=unit,log_likelihood=r.log_likelihood));io.write_new(out/f'fresh-{j}-diagnostics.json',io.jsonable(r.diagnostics()))
 cross=[dict(method='bridge'if i<3 else'fresh',replicate=i%3,**previous.compare(v,mh))for i,v in enumerate(values)];pairs=[dict(first=i,second=j,**previous.compare(a,values[j]))for i,a in enumerate(values)for j in range(i+1,len(values))];fresh_range=max(freshZ)-min(freshZ);bridge_range=max(bridgeZ)-min(bridgeZ)
 passed=bool(old['MH_passed']and all(v['passed']for v in cross+pairs)and fresh_range<=.05 and bridge_range<=.05)
 io.write_new(out/'summary.json',io.jsonable(dict(completed=True,index=index,passed=passed,action=old['action'],noise_index=0,MH_passed_reused=True,new_MH_draws=0,SMC_vs_saved_MH=cross,SMC_pairs=pairs,fresh_joint_logZ_range=fresh_range,bridge_conditional_logZ_range=bridge_range,old8192_passed=old['passed'],old8192_negative_preserved=True,function_replay_only=True,worker_wall_s=time.monotonic()-start,worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,raw_LOCAL=True)))


def aggregate(rows):
 if len(rows)!=12 or sorted(r['index']for r in rows)!=list(range(12))or not all(r['completed']for r in rows):raise ValueError('Exactlyall12 same diagnostic objects required')
 return dict(completed=True,objects=12,particles=32768,old_particles=8192,numerically_passed_objects=sum(r['passed']for r in rows),all_numerical_gates_passed=all(r['passed']for r in rows),all12_retained=True,new_MH_draws=0,policy_rerun=False,policy_advantage_claim=False,per_object=sorted(rows,key=lambda r:r['index']),old8192_negatives_preserved=True,raw_LOCAL=True)


def worker_command(protocol,digest,index,slot):
 return ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','taskset','-c',str(RESOURCES['worker_slots'][slot][0]),str(ROOT/'.venv/bin/python'),'-m','experiments.run_postpoke_count_continuation_v01','worker','--protocol',protocol,'--digest',digest,'--index',str(index),'--slot',str(slot)]


def controller_command(protocol,digest):
 return ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','taskset','-c',str(RESOURCES['controller_cores'][0]),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_postpoke_count_continuation_v01','run','--protocol',protocol,'--digest',digest]


def run(protocol,digest):
 p=validate(protocol,digest);model.guard(RESOURCES['controller_cores']);out=io.local(ROOT,p['output']);out.mkdir(parents=True,exist_ok=False)
 if shutil.disk_usage(out).free<RESOURCES['disk_free_reserve_bytes']+RESOURCES['maximum_raw_bytes']:raise RuntimeError('Count-only disk reserve')
 io.atomic(out/'journal.json',dict(pid=os.getpid(),completed=False,status='BOOTSTRAP_BEFORE_EXPOSURE',live=[],records=[],pending_indices=list(range(12))))
 def command(i,slot):return worker_command(protocol,digest,i,slot)
 try:
  records=owner.owned_jobs(command,list(range(12)),12,out,worker_wall_s=RESOURCES['worker_wall_s'],global_wall_s=RESOURCES['global_wall_s'],RSS_bytes=RESOURCES['worker_RSS_bytes'],controller_check=lambda:model.guard(RESOURCES['controller_cores']))
  rows=[json.loads((out/f'object-{i:02d}'/'summary.json').read_text())for i in range(12)];io.write_new(out/'summary.json',aggregate(rows));io.write_new(out/'execution.json',dict(completed=True,pid=os.getpid(),all_owned_workers_reaped=True,records=records,protocol_sha256=digest))
 except BaseException as error:
  io.write_new(out/'failure.json',dict(completed=False,error=repr(error),all_partial_artifacts_preserved=True,no_retry=True));raise


def supervise(protocol,digest):
 from experiments.supervise_surrogate_sbc_continuation import supervise as owned_supervise
 p=validate(protocol,digest);op=(ROOT/protocol).parent
 command=controller_command(protocol,digest)
 return owned_supervise(command,op/'actual-count.log',op/'actual-supervisor-receipt.json',journal_path=ROOT/p['output']/'journal.json',wall_s=RESOURCES['global_wall_s'],RSS_bytes=RESOURCES['worker_RSS_bytes'])


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','worker','supervise','guard-worker','guard-controller']);a.add_argument('--protocol',required=True);a.add_argument('--digest');a.add_argument('--output');a.add_argument('--index',type=int);a.add_argument('--slot',type=int);v=a.parse_args()
 if v.mode=='guard-worker':
  model.guard(RESOURCES['worker_slots'][v.slot]);print(json.dumps(dict(CPU=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),GPU_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))))
 elif v.mode=='guard-controller':
  model.guard(RESOURCES['controller_cores']);print(json.dumps(dict(CPU=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),GPU_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))))
 elif v.mode=='prepare':print(prepare(v.protocol,v.output))
 elif v.mode=='worker':worker(v.protocol,v.digest,v.index,v.slot)
 elif v.mode=='run':run(v.protocol,v.digest)
 else:
  receipt=supervise(v.protocol,v.digest);print(json.dumps(receipt));raise SystemExit(0 if receipt.get('completed')else 1)
