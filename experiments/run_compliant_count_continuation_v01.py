"""NEW count-resolution continuation; same fixed chain and unchanged finite likelihood.

Changed MH kernel must first match frozen2D reference. No model/data tuning,
new observations, quadrature-grid posterior or physical calibration claim.
"""
from pathlib import Path
import argparse,hashlib,json,os,signal,time
signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGALRM})
import numpy as np
import torch
from scipy.optimize import minimize
from experiments import campaign_compliant_count_continuation_v01 as model
from experiments import run_compliant_ridge_reference as predecessor
from experiments import campaign_compliant_observation_forward as observation
from experiments import run_campaign_surrogate_sbc_v03 as owner
from experiments import run_campaign_surrogate_sbc_v02 as bundle_io
io=owner.io;ROOT=owner.ROOT;SOURCE='experiments/run_compliant_count_continuation_v01.py'
SEEDS=(2026121101,2026121102,2026121103);COUNTS=(8192,32768);FILES=owner.FILES
# Only qualification count changes; original full4D seeds/model/design survive.
RESOURCES=dict(predecessor.RESOURCES,qualification_wall_s=300)
DESIGN=dict(predecessor.DESIGN)
DESIGN['MH']=dict(predecessor.DESIGN['MH'],qualification_draws=16000)
DESIGN['continuation']=dict(previous_retained=2000,additional_retained=14000,total_retained=16000,new_warmup=0,new_seed=False,exact_RNG_only_replay=True,supplementary_long_batches=[800,1600],no_optional_stopping=True)
PREVIOUS_PROTOCOL='runs/operator-dev/compliant-continuous-full4D-v01/protocol.json'
PREVIOUS_OUTPUT='runs/gaussian-campaign-v01/compliant-continuous-full4D-v01'



def sources():
 prior=json.loads((ROOT/'runs/operator-dev/compliant-smc-resolution-v04/protocol.json').read_text())
 return sorted(set(prior['source_sha256'])|set(owner.sources())|{SOURCE,'experiments/campaign_compliant_ridge_reference.py','experiments/supervise_compliant_ridge_reference.py','experiments/supervise_surrogate_sbc_continuation.py','tests/test_compliant_ridge_reference.py','experiments/run_compliant_ridge_reference.py','experiments/campaign_compliant_count_continuation_v01.py','experiments/supervise_compliant_count_continuation_v01.py','tests/test_compliant_count_continuation_v01.py'})


def prepare(protocol,output):
 dest=io.local(ROOT,protocol)
 if dest.exists()or io.local(ROOT,output).exists():raise FileExistsError('New exclusive full4D namespace')
 prior=json.loads((ROOT/'runs/operator-dev/compliant-smc-resolution-v04/protocol.json').read_text())
 for n,h in prior['source_sha256'].items():io.checked(ROOT,dict(path=n,sha256=h))
 names=dict(prior_reference_protocol='runs/operator-dev/compliant-smc-resolution-v04/protocol.json',original_design='runs/operator-dev/compliant-patch-inverse-v01/design.json',observation='runs/gaussian-campaign-v01/compliant-patch-inverse-v01/posterior-order5.npz',qualified2D='runs/operator-dev/compliant-smc-resolution-v04/result-summary.json',adaptive='runs/gaussian-campaign-v01/compliant-continuous-reference-v01/conditional2D/summary.json',adaptive_moments='runs/gaussian-campaign-v01/compliant-continuous-reference-v01/conditional2D/adaptive1-moments.npz',pilot2D='runs/gaussian-campaign-v01/compliant-smc-resolution-v04/smc-32768-2026121103.npz',old4D='runs/gaussian-campaign-v01/compliant-patch-resolution-v01/summary.json')
 old=predecessor.validate(PREVIOUS_PROTOCOL,io.sha(ROOT/PREVIOUS_PROTOCOL))
 execution=json.loads((ROOT/PREVIOUS_OUTPUT/'execution.json').read_text())
 row=execution['records'][0]
 if execution.get('completed')is not True or execution.get('all_owned_workers_reaped')is not True or len(execution['records'])!=1 or row['job']!='qualify2D' or row['exit_code']!=0 or row['reaped']is not True:raise ValueError('Completed predecessor qualification only')
 saved=bundle_io.read_bundle(ROOT,row['files'])
 if saved.get('passed')is not False or saved.get('vs_frozen_adaptive',{}).get('passed')is not True:raise ValueError('Preserve original2D agreement and precision negative')
 recovered=model.recover_rng(saved['MH']['seed'],saved['MH']['initial_states'],saved['MH']['warmup']+saved['MH']['draws'])
 dest.parent.mkdir(parents=True,exist_ok=True)
 identity_path=dest.parent/'continuation-identity.json'
 identity=dict(predecessor_protocol=io.record(ROOT,ROOT/PREVIOUS_PROTOCOL),predecessor_files=row['files'],RNG_state=recovered.bit_generator.state,RNG_only_replay_steps=2500,previous_final_unit_sha256=hashlib.sha256(saved['unit'][:,-1].copy().tobytes()).hexdigest(),previous_final_loglik_sha256=hashlib.sha256(saved['MH']['log_likelihood'][:,-1].copy().tobytes()).hexdigest(),no_posterior_calls=True,new_initialization=False)
 io.write_new(identity_path,identity)
 names['continuation_identity']=str(identity_path.relative_to(ROOT))
 names.update(predecessor_protocol=PREVIOUS_PROTOCOL,predecessor_execution=PREVIOUS_OUTPUT+'/execution.json',predecessor_summary=PREVIOUS_OUTPUT+'/summary.json',predecessor_proposal=PREVIOUS_OUTPUT+'/proposal-freezes/qualify2D.npz',predecessor_freeze=PREVIOUS_OUTPUT+'/proposal-freezes/qualify2D.json')
 for name,record in row['files'].items():names['previous_'+name]=record['path']
 p=dict(schema='COMPLIANT-count-resolution-continuation-v01',status='FROZEN_BEFORE_ANY_NEW_POSTERIOR_RUN',design=DESIGN,resources=RESOURCES,runtime=io.runtime(),inputs={k:io.record(ROOT,ROOT/n)for k,n in names.items()},source_sha256={n:io.sha(ROOT/n)for n in sources()},output=output,all_old_failures_preserved=True,raw_LOCAL=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest):
 p=io.read_json(ROOT,dict(path=protocol,sha256=digest))
 if p.get('schema')!='COMPLIANT-count-resolution-continuation-v01'or p.get('status')!='FROZEN_BEFORE_ANY_NEW_POSTERIOR_RUN'or p.get('design')!=DESIGN or p.get('resources')!=RESOURCES or p.get('runtime')!=io.runtime()or p.get('source_sha256')!={n:io.sha(ROOT/n)for n in sources()}:raise ValueError('Exact full4D source/design/prior/runtime/resources')
 for r in p['inputs'].values():io.checked(ROOT,r)
 if io.read_json(ROOT,p['inputs']['qualified2D']).get('declared_conditional2D_numerical_validation_passed')is not True:raise ValueError('Previously qualified conditional reference required')
 predecessor.validate(PREVIOUS_PROTOCOL,p['inputs']['predecessor_protocol']['sha256'])
 io.local(ROOT,p['output']);return p


def load_observation(p):
 with np.load(io.checked(ROOT,p['inputs']['observation']),allow_pickle=False)as z:return z['observation'].copy()


def summarize_smc(forward,result):
 unit=result.state.coordinates;pred=model.predictions(forward,unit,(0,1,2,3));knots=np.linspace(0,1,33)
 return dict(unit_CDF=np.mean(unit[:,:,None]<=knots,axis=0),unit_quantiles=np.quantile(unit,[.05,.5,.95],axis=0).T,predictions=pred,SMC=result.diagnostics(),unit=unit,log_likelihood=result.log_likelihood)


def summarize_MH(forward,unit,diagnostics,dims):
 q=model.cdf_quantile_mc(unit);q.update(predictions=model.predictions(forward,unit,dims,chain=True),MH=diagnostics,unit=unit)
 qerrors=[x for row in q['quantile_MCSE']for x in row];pred=q['predictions'];field_mc=[]
 for label,start in [('A',25),('B',0)]:
  r=pred[label];tol=max(2e-6,.03*float(np.sqrt(np.mean(r['std'][start:]**2))));field_mc.append(all(float(np.sqrt(np.mean(r[k][start:]**2)))<=tol for k in ('mean_batch_MCSE','std_delta_batch_MCSE')))
 ids=pred['function_unknown_indices'];sd=pred['function_std'][ids];err=pred['function_batch_mean_MCSE'][ids];ratio=np.divide(err,sd,out=np.where(err==0,0.,np.inf),where=sd>0);q['maximum_function_relative_MCSE']=float(ratio.max());q['field_MCSE_passed']=all(field_mc)
 q['MC_diagnostics_passed']=bool(all(field_mc)and q['maximum_function_relative_MCSE']<=.05 and diagnostics['ridge_diagnostics']['all_monitored_quantities_pass']and diagnostics['diagnostics']['all_monitored_quantities_pass']and q['predictions']['function_diagnostics']['all_monitored_quantities_pass']and np.max(q['CDF_batch_MCSE'])<=DESIGN['gates']['CDF_batch_MCSE']and all(x is not None and x<=DESIGN['gates']['unit_quantile_MCSE']for x in qerrors));return q


def reference_2D(p):
 raw=io.read_json(ROOT,p['inputs']['adaptive'])['adaptive_reference'][-1];pred={}
 with np.load(io.checked(ROOT,p['inputs']['adaptive_moments']),allow_pickle=False)as z:
  for k in ('A','B'):pred[k]=dict(mean=z[k+'_mean'].copy(),std=z[k+'_std'].copy())
 return dict(unit_CDF=np.asarray(raw['unit_CDF']),unit_quantiles=np.asarray(raw['unit_quantiles']),predictions=pred)


def diagnostics(forward,obs,p):
 original=model.original;truth=(np.array([np.log(original.TRUTH[0]),original.TRUTH[1],np.log(original.TRUTH[2]),np.log(original.TRUTH[3])])-original.LOWER)/(original.UPPER-original.LOWER);old=io.read_json(ROOT,p['inputs']['old4D'])['levels'][-1]['material_marginals'];physical=np.array([x['quantiles_05_50_95'][1]for x in old]);intrinsic=physical.copy();intrinsic[[0,2,3]]=np.log(intrinsic[[0,2,3]]);median=(intrinsic-original.LOWER)/(original.UPPER-original.LOWER);points=np.array([np.full(4,.5),truth,median]);sensitivity=[]
 for x in points:
  jac=[]
  for h in DESIGN['diagnostic']['jacobian_steps']:
   rows=[]
   for j in range(4):
    a=x.copy();b=x.copy();a[j]+=h;b[j]-=h;y=forward.observations(model.base.physical(np.array([a,b]),(0,1,2,3)));rows.append((y[0]-y[1])/(2*h*original.SIGMA))
   jac.append(np.array(rows).T)
  U,s,V=np.linalg.svd(jac[-1],full_matrices=False);sensitivity.append(dict(unit_point=x,singular_values=s,right_singular_vectors=V,Jacobian_step_difference_spectral_norm=float(np.linalg.norm(jac[0]-jac[1],2)),local_rank_is_not_global_identifiability=True))
 profiles=[]
 for j in range(4):
  nuisance=[k for k in range(4)if k!=j]
  for knot in np.linspace(0,1,DESIGN['diagnostic']['profile_knots']):
   attempts=[]
   def objective(z):
    x=np.empty(4);x[j]=knot;x[nuisance]=z;return -float(forward.log_likelihood(x[None],(0,1,2,3),obs)[0])
   for start in points:
    fit=minimize(objective,start[nuisance],method='L-BFGS-B',bounds=[(0.,1.)]*3,options=dict(maxiter=80,maxfun=256,ftol=1e-10,gtol=1e-6));attempts.append(dict(success=bool(fit.success),negative_loglik=float(fit.fun),nuisance=fit.x,nfev=int(fit.nfev),message=str(fit.message)))
   profiles.append(dict(parameter=j,unit_value=float(knot),attempts=attempts,best_negative_loglik=min(a['negative_loglik']for a in attempts)))
 return dict(sensitivity=sensitivity,profiles=profiles,profile_convergence_complete=all(any(a['success']for a in row['attempts'])for row in profiles),structural_nonidentifiability_claim=False,scope='Whitened joint local sensitivity and constrained profile diagnostics; no model retuning or global injectivity proof')


def stage_barrier(p,digest,kind):
 out=io.local(ROOT,p['output']);seal=io.read_json(ROOT,io.record(ROOT,out/(kind+'-seal.json')))
 expected=['qualify2D']if kind=='qualification'else['smc-'+str(n)+'-'+str(seed)for n in COUNTS for seed in SEEDS]
 if seal.get('protocol_sha256')!=digest or seal.get('completed')is not True or len(seal.get('records',[]))!=len(expected)or {r['job']for r in seal['records']}!=set(expected):raise ValueError('Exact completed stage seal before next stage')
 for row in seal['records']:
  if row.get('reaped')is not True or row.get('exit_code')!=0 or Path('/proc',str(row['pid'])).exists():raise ValueError('Previous stage workers must be absent/reaped')
  value=bundle_io.read_bundle(ROOT,row['files'])
  if value.get('completed')is not True or value.get('job')!=row['job']or value.get('protocol',{}).get('sha256')!=digest or value.get('passed')is not True:raise ValueError('Qualified same-protocol stage artifact')
 return seal


def worker(protocol,digest,job,slot):
 p=validate(protocol,digest);bundle_io.guard(RESOURCES['worker_slots'][slot]);torch.set_num_threads(2)
 if job!='qualify2D':stage_barrier(p,digest,'qualification')
 if job=='fullMH':smc_seal=stage_barrier(p,digest,'SMC')
 pilot_identity=p['inputs']['pilot2D']if job=='qualify2D'else None;obs=load_observation(p);forward=observation.ObservationOnlyForward();out=io.local(ROOT,p['output'])/'jobs'/job
 def freeze(mean,cov):
  directory=io.local(ROOT,p['output'])/'proposal-freezes';directory.mkdir(exist_ok=True);path=directory/(job+'.npz');io.codec.exclusive_npz(path,dict(mean=mean,covariance=cov));record=io.record(ROOT,path);io.write_new(directory/(job+'.json'),dict(protocol_sha256=digest,job=job,pilot_identity=pilot_identity,proposal=record,frozen_before_initialization_and_production=True));return record
 if job=='qualify2D':
  with np.load(io.checked(ROOT,p['inputs']['pilot2D']),allow_pickle=False)as z:pilot=z['unit'].copy()
  files={n:p['inputs']['previous_'+n]for n in FILES};previous=bundle_io.read_bundle(ROOT,files)
  with np.load(io.checked(ROOT,p['inputs']['predecessor_proposal']),allow_pickle=False)as z:
   np.testing.assert_array_equal(z['mean'],previous['MH']['proposal_mean']);np.testing.assert_array_equal(z['covariance'],previous['MH']['proposal_covariance'])
  np.testing.assert_array_equal(pilot.mean(0),previous['MH']['proposal_mean']);np.testing.assert_array_equal(np.cov(pilot,rowvar=False),previous['MH']['proposal_covariance'])
  def checkpoint(state):
   identity=io.read_json(ROOT,p['inputs']['continuation_identity'])
   if state['RNG_state']!=identity['RNG_state']or hashlib.sha256(state['unit'].tobytes()).hexdigest()!=identity['previous_final_unit_sha256']or hashlib.sha256(state['log_likelihood'].tobytes()).hexdigest()!=identity['previous_final_loglik_sha256']:raise ValueError('Exact prefrozen continuation identity')
   dest=io.local(ROOT,p['output'])/'continuation-checkpoints';dest.mkdir(exist_ok=True)
   arrays={k:state[k]for k in ('unit','log_likelihood','proposal_mean','proposal_covariance')};io.codec.exclusive_npz(dest/'start.npz',arrays)
   io.write_new(dest/'start.json',dict(protocol_sha256=digest,arrays=io.record(ROOT,dest/'start.npz'),RNG_state=state['RNG_state'],previous_draws=state['previous_draws'],before_any_new_likelihood=True,previous_files=files))
  chain,report=model.continue_mh(lambda x:forward.log_likelihood(x,(0,3),obs),previous,additional_draws=14000,checkpoint=checkpoint)
  result=summarize_MH(forward,chain,report,(0,3));result['long_batch_MC']=model.long_batch_mc(chain);result['vs_frozen_adaptive']=model.compare(result,reference_2D(p));result['passed']=result['MC_diagnostics_passed']and result['vs_frozen_adaptive']['passed']and result['long_batch_MC']['passed']
 elif job=='jointDiagnostics':result=diagnostics(forward,obs,p);result['passed']=True # Does not authorize an identifiability claim.
 elif job.startswith('smc-'):
  count,seed=map(int,job.split('-')[1:])
  if count not in COUNTS or seed not in SEEDS:raise ValueError('Only frozen six SMC slots')
  result=summarize_smc(forward,model.base.smc(forward,(0,1,2,3),obs,count,seed));result['passed']=True
 elif job=='fullMH':
  files=next(r['files']for r in smc_seal['records']if r['job']=='smc-32768-2026121103');pilot_identity=files;pilot=bundle_io.read_bundle(ROOT,files)['unit'];chain,report=model.ridge_mh(lambda x:forward.log_likelihood(x,(0,1,2,3),obs),pilot,seed=2026121120,warmup=1000,draws=4000,freeze_proposal=freeze);result=summarize_MH(forward,chain,report,(0,1,2,3));result['pilot_files']=files;result['passed']=result['MC_diagnostics_passed']
 else:raise ValueError('Unknown frozen job')
 result.update(completed=True,job=job,protocol=dict(path=protocol,sha256=digest));return bundle_io.bundle(ROOT,out,result)


def run(protocol,digest):
 p=validate(protocol,digest);bundle_io.guard(RESOURCES['controller_cores']);out=io.local(ROOT,p['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();all_records=[]
 def execute(jobs,label,cap):
  stage=out/label;stage.mkdir();ids=list(range(len(jobs)))
  io.atomic(stage/'journal.json',dict(pid=os.getpid(),completed=False,status='BOOTSTRAP_BEFORE_EXPOSURE',live=[],records=[],pending_indices=ids))
  link=out/'active-journal.next';link.symlink_to(stage/'journal.json');link.replace(out/'active-journal.json')
  def command(i,slot):return ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','taskset','-c',','.join(map(str,RESOURCES['worker_slots'][slot])),str(ROOT/'.venv/bin/python'),'-m','experiments.run_compliant_count_continuation_v01','worker','--protocol',protocol,'--digest',digest,'--job',jobs[i],'--slot',str(slot)]
  remaining=RESOURCES['global_wall_s']-(time.monotonic()-start)
  if remaining<=0:raise TimeoutError('Full campaign1800s cap')
  records=owner.owned_jobs(command,ids,min(len(jobs),9),stage,worker_wall_s=cap,global_wall_s=remaining,RSS_bytes=RESOURCES['worker_RSS_bytes'],controller_check=lambda:bundle_io.guard(RESOURCES['controller_cores']))
  values=[]
  for r in records:
   r['job']=jobs[r['object_index']];r['files']={n:io.record(ROOT,out/'jobs'/r['job']/n)for n in FILES};values.append(bundle_io.read_bundle(ROOT,r['files']));all_records.append(r)
  io.atomic(out/'execution.json',dict(completed=False,pid=os.getpid(),all_owned_workers_reaped=True,records=all_records,protocol=dict(path=protocol,sha256=digest)))
  return values
 try:
  qualification=execute(['qualify2D'],'qualification',RESOURCES['qualification_wall_s'])[0]
  if not qualification['passed']:
   io.write_new(out/'summary.json',dict(completed=True,passed=False,full4D_started=False,reason='New kernel failed unchanged conditional2D reference/MC gates',blocked_stages=dict(jointDiagnostics='BLOCKED_BY_2D_GATE',six_SMC='BLOCKED_BY_2D_GATE',fullMH='BLOCKED_BY_2D_GATE'),raw_LOCAL=True));io.atomic(out/'execution.json',dict(completed=True,status='STOPPED_ON_NUMERICAL_QUALIFICATION_NEGATIVE',pid=os.getpid(),all_owned_workers_reaped=True,records=all_records,protocol=dict(path=protocol,sha256=digest)));return
  io.write_new(out/'qualification-seal.json',dict(completed=True,protocol_sha256=digest,records=[r for r in all_records if r['job']=='qualify2D']))
  joint=execute(['jointDiagnostics'],'joint-diagnostics',180)[0]
  smc=execute(['smc-'+str(n)+'-'+str(seed)for n in COUNTS for seed in SEEDS],'SMC',180)
  io.write_new(out/'SMC-seal.json',dict(completed=True,protocol_sha256=digest,records=[r for r in all_records if r['job'].startswith('smc-')]))
  mh=execute(['fullMH'],'MH',600)[0];comparisons=[dict(first=dict(particles=a['SMC']['particles'],seed=a['SMC']['seed']),second=dict(particles=b['SMC']['particles'],seed=b['SMC']['seed']),**model.compare(a,b))for i,a in enumerate(smc)for b in smc[i+1:]];cross=[dict(particles=a['SMC']['particles'],seed=a['SMC']['seed'],MH_seed=2026121120,**model.compare(a,mh))for a in smc];logZ=[r['SMC']['log_evidence_estimate']for r in smc];logZ_range=max(logZ)-min(logZ)
  # Independent-seed Monte Carlo uncertainty, distinct from particle ESS.
  seed_errors={}
  for n in COUNTS:
   selected=[r for r in smc if r['SMC']['particles']==n];fields={}
   for label in ('A','B'):
    for moment in ('mean','std'):
     arrays=np.array([r['predictions'][label][moment]for r in selected]);se=arrays.std(0,ddof=1)/np.sqrt(3);fields[label+'_'+moment+'_seed_MCSE_max_m']=float(se.max())
   seed_errors[str(n)]=fields
  passed=mh['passed']and all(r['passed']for r in comparisons+cross)and logZ_range<=.05
  io.write_new(out/'summary.json',io.jsonable(dict(completed=True,passed=passed,full4D_started=True,qualification_passed=True,six_SMC_slots=6,pairwise_comparisons=comparisons,MH_vs_SMC=cross,SMC_logZ_range=logZ_range,independent_seed_prediction_MCSE=seed_errors,MH_rank_ESS=mh['MH']['diagnostics'],MH_function_rank_ESS=mh['predictions']['function_diagnostics'],MH_ridge_rank_ESS=mh['MH']['ridge_diagnostics'],independent_prior_chains=8,MH_function_relative_MCSE=mh['maximum_function_relative_MCSE'],MH_field_MCSE_passed=mh['field_MCSE_passed'],SMC_logZ_by_slot=[dict(particles=r['SMC']['particles'],seed=r['SMC']['seed'],log_evidence=r['SMC']['log_evidence_estimate'])for r in smc],MH_max_CDF_MCSE=float(np.max(mh['CDF_batch_MCSE'])),MH_max_unit_quantile_MCSE=max((x for row in mh['quantile_MCSE']for x in row if x is not None),default=None),joint_diagnostics_completed=True,profile_convergence_complete=joint['profile_convergence_complete'],structural_nonidentifiability_claim=False,physical_calibration=False,all_old_failures_preserved=True,raw_LOCAL=True)))
  io.atomic(out/'execution.json',dict(completed=True,pid=os.getpid(),all_owned_workers_reaped=True,records=all_records,protocol=dict(path=protocol,sha256=digest),wall_s=time.monotonic()-start))
 except BaseException as error:io.write_new(out/'failure.json',dict(completed=False,error=repr(error),records=all_records,active_stage_journal=io.record(ROOT,out/'active-journal.json')if(out/'active-journal.json').exists()else None,all_completed_and_partial_jobs_preserved=True,no_retry=True));raise


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run','worker'));a.add_argument('--protocol',required=True);a.add_argument('--digest');a.add_argument('--output');a.add_argument('--job');a.add_argument('--slot',type=int);v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output))
 elif v.mode=='worker':worker(v.protocol,v.digest,v.job,v.slot)
 else:run(v.protocol,v.digest)
