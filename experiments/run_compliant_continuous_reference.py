"""Frozen, bounded 2D -> 4D manufactured continuous posterior development."""
from __future__ import annotations
import argparse,hashlib,json,signal,time
from pathlib import Path
import numpy as np
import torch
import scipy
from scipy.integrate import cubature
from scipy.special import ndtr
from experiments import campaign_compliant_continuous_reference as model

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(2026121101,2026121102,2026121103)
COUNTS=(512,2048)
KNOTS=np.linspace(0,1,33)
TOLERANCES=((1e-5,1e-10),(1e-7,1e-12))
class BudgetReached(Exception):pass

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rec(p):return dict(path=Path(p).resolve().relative_to(ROOT).as_posix(),sha256=sha(p))
def check(r):
 p=ROOT/r['path']
 if sha(p)!=r['sha256']:raise ValueError('Frozen reference source/input changed: '+r['path'])
 return p

def prepare(path):
 p=Path(path)
 if p.exists():raise ValueError('Fresh protocol required')
 old=ROOT/'runs/operator-dev/compliant-patch-inverse-v01/design.json'
 sources=json.loads(old.read_text())['source_sha256']
 names=['experiments/campaign_compliant_continuous_reference.py','experiments/run_compliant_continuous_reference.py',
        'tests/test_campaign_compliant_continuous_reference.py','experiments/campaign_tempered_reference.py','src/physgauss/chain_diagnostics.py']
 sources={**sources,**{x:sha(ROOT/x) for x in names}}
 d=dict(schema='compliant-continuous-reference-v0.1',status='FROZEN_BEFORE_PHYSICAL_POSTERIOR_CALLS',
  original_design=rec(old),observation_archive=rec(ROOT/'runs/gaussian-campaign-v01/compliant-patch-inverse-v01/posterior-order5.npz'),
  source_sha256=sources,CPU_threads=2,phase_caps_s={'conditional2D':180,'full4D':360},seeds=list(SEEDS),particles=list(COUNTS),
  prior='UNCHANGED uniform intrinsic(logE,center,logwidth,logfixtureK); unit-box transform includes exact original law',
  conditional_slice=dict(dimensions=[0,3],fixed_center_width=model.original.TRUTH[[1,2]].tolist(),
   interpretation='Conditional diagnostic only, not the 4D marginal or changed 4D prior'),
  adaptive_reference=dict(algorithm='Independent adaptive2D Gauss-Kronrod21 product cubature',tolerances=TOLERANCES,max_subdivisions=256,
   marginal_CDF_unit_knots=KNOTS.tolist(),quantiles=[.05,.5,.95],quantile_bisections=18,
   CDF_tolerance_agreement=2e-4,unit_quantile_tolerance_agreement=1e-4,predictive_field_agreement_m=1e-7),
  SMC=dict(ess_fraction=.8,rejuvenation_steps=8,proposal_scale=.08,independence_probability=.25,max_stages=64,
   fixed_3seeds_2counts=True,ESS_alone_never_sufficient=True),
  MCMC=dict(chains=8,all_chains_independent_prior_initialization=True,conditional_warmup=500,conditional_draws=2000,
   full_warmup=1000,full_draws=4000,proposal='Frozen SMC pilot normal independence(.5) + symmetric diagonal reflected RW(.45) + prior(.05), exact MH correction',
   prior_chains_start_seed=2026121110,full_start_seed=2026121120,no_adaptation_extension_or_thinning=True),
  validation=dict(SMC_and_MCMC_CDF_vs_2D_reference_max=.05,SMC_and_MCMC_unit_quantile_max=.02,
   between_seed_count_CDF_max=.05,between_seed_count_unit_quantile_max=.02,
   forecast_RMS_absolute_m=2e-6,forecast_fraction_finer_latent_std=.03,
   rank_folded_split_Rhat_max=1.01,bulk_tail_ESS_min=400,no_frozen_chains=True,
   analytic_truncatedGaussian_SMC_2048_CDF_max=.04,analytic_unit_quantile_max=.02,
   gate='All2D adaptive/analytic/SMC/MCMC checks pass BEFORE full4D executes. Every independent seed and both counts included.'),
  parity_validation='4 synthetic tests passed before freeze: randomPRIOR/boundary/posterior-scale full-field and likelihood parity, prior slice, analytic cubature, MH prior recovery',
  same_saved_observation_prior_noise_physics_controls=True,old_GL_CC_failures_preserved=True,no_current_cohort_access=True,
  preserve_every_attempt_no_retry=True,scope='Numerical manufactured posterior validation, NOT physical calibration or Isaac/generalization',
  versions=dict(numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__))
 p.parent.mkdir(parents=True,exist_ok=True);write(p,d);return rec(p)


def summaries(forward,x,dims,out,label):
 theta=model.physical(x,dims);w=np.ones(len(x))/len(x);values={};means={};stds={}
 for program,start in [('A',25),('B',0)]:
  total=np.zeros((61,5,3));second=total.copy();anchor=None
  for first in range(0,len(x),512):
   u=forward(theta[first:first+512],program)
   if anchor is None:anchor=u[0].copy()
   delta=u-anchor;total+=delta.sum(0)/len(x);second+=(delta**2).sum(0)/len(x)
  variance=second-total**2
  if variance.min() < -1e-20:raise FloatingPointError('Centered field variance invalid')
  means[program]=anchor+total;stds[program]=np.sqrt(np.maximum(variance,0))
  values[program]=dict(latent_std_RMS_m=float(np.sqrt(np.mean(stds[program][start:]**2))))
 np.savez_compressed(out/(label+'-moments.npz'),**{p+'_'+k:v for k,data in [('mean',means),('std',stds)] for p,v in data.items()})
 return dict(unit_CDF=[model.empirical_cdf(x[:,j],KNOTS).tolist() for j in range(x.shape[1])],
  unit_quantiles=np.quantile(x,[.05,.5,.95],axis=0).T.tolist(),
  physical_parameters=model.original.parameter_summary(theta,w),targets=values,_mean=means,_std=stds)


def field_difference(a,b):
 result={}
 for p,start in [('A',25),('B',0)]:
  tolerance=max(2e-6,.03*np.sqrt(np.mean(b['_std'][p][start:]**2)))
  mean=float(np.sqrt(np.mean((a['_mean'][p][start:]-b['_mean'][p][start:])**2)))
  std=float(np.sqrt(np.mean((a['_std'][p][start:]-b['_std'][p][start:])**2)))
  result[p]=dict(mean_RMS_m=mean,std_RMS_m=std,tolerance_m=float(tolerance),passed=mean<=tolerance and std<=tolerance)
 return result


def comparison(a,b):
 cdf=float(np.max(np.abs(np.asarray(a['unit_CDF'])-b['unit_CDF'])))
 quantile=float(np.max(np.abs(np.asarray(a['unit_quantiles'])-b['unit_quantiles'])))
 fields=field_difference(a,b)
 return dict(max_CDF_difference=cdf,max_unit_quantile_difference=quantile,fields=fields,
  passed=cdf<=.05 and quantile<=.02 and all(v['passed'] for v in fields.values()))


def adaptive_reference(forward,obs,rtol,atol,out,label):
 dims=(0,3);anchor=float(forward.log_likelihood(np.full((1,2),.5),dims,obs)[0]);reports=[]
 def density(x):return np.exp(forward.log_likelihood(x,dims,obs)-anchor)
 def integral(upper,func=density):
  result=cubature(func,np.zeros(2),np.asarray(upper),rule='gk21',rtol=rtol,atol=atol,max_subdivisions=256)
  reports.append(dict(status=result.status,max_error=float(np.max(result.error)),subdivisions=result.subdivisions))
  if result.status!='converged':raise RuntimeError('Adaptive cubature did not meet declared tolerance')
  return result.estimate
 Z=float(integral(np.ones(2)));cdfs=[];quantiles=[]
 for j in range(2):
  cdf=[]
  for k in KNOTS:
   upper=np.ones(2);upper[j]=k
   cdf.append(0. if k==0 else 1. if k==1 else float(integral(upper)/Z))
  cdfs.append(cdf);q=[]
  for probability in (.05,.5,.95):
   lo,hi=0.,1.
   for _ in range(18):
    mid=(lo+hi)/2;upper=np.ones(2);upper[j]=mid
    if integral(upper)/Z<probability:lo=mid
    else:hi=mid
   q.append((lo+hi)/2)
  quantiles.append(q)
 def field_integrand(x):
  theta=model.physical(x,dims);a=forward(theta,'A');b=forward(theta,'B')
  prediction=model.original.selected_observations(a)
  ll=-.5*np.sum(((prediction-obs)/model.original.SIGMA)**2,axis=1)-len(obs)*np.log(model.original.SIGMA*np.sqrt(2*np.pi))
  w=np.exp(ll-anchor);u=np.concatenate([a.reshape(len(x),-1),b.reshape(len(x),-1)],1)/.001
  return np.column_stack([w[:,None]*u,w[:,None]*u**2])
 moment=integral(np.ones(2),field_integrand)/Z;n=61*5*3*2
 mean=moment[:n]*.001;variance=(moment[n:]-moment[:n]**2)*1e-6
 if variance.min()<-1e-18:raise FloatingPointError('Adaptive field variance invalid')
 means={p:v.reshape(61,5,3) for p,v in zip(('A','B'),mean.reshape(2,-1))}
 stds={p:v.reshape(61,5,3) for p,v in zip(('A','B'),np.sqrt(np.maximum(variance,0)).reshape(2,-1))}
 np.savez_compressed(out/(label+'-moments.npz'),**{p+'_'+k:v for k,data in [('mean',means),('std',stds)] for p,v in data.items()})
 qphys=model.physical(np.array(quantiles).T,dims)
 return dict(rtol=rtol,atol=atol,Z_scaled=Z,log_evidence=float(np.log(Z)+anchor),unit_CDF=cdfs,
  unit_quantiles=quantiles,physical_quantiles={model.original.PARAMETERS[d]:qphys[:,d].tolist() for d in dims},
  integrals=reports,_mean=means,_std=stds)


def analytic_check():
 mu=np.array([.42,.57]);sd=np.array([.08,.13]);prior=model.BoxPrior(2)
 result=model.tempered_smc(prior,lambda s:-.5*np.sum(((s.coordinates-mu)/sd)**2,1),particles=2048,seed=2026121100,
  rejuvenation_steps=8,periodic_indices=(),proposal_scale=.08)
 x=result.state.coordinates;cdfs=[]
 from scipy.special import ndtri
 exactq=mu+sd*ndtri(ndtr(-mu/sd)[None]+np.array([.05,.5,.95])[:,None]*(ndtr((1-mu)/sd)-ndtr(-mu/sd))[None])
 for j in range(2):
  exact=(ndtr((KNOTS-mu[j])/sd[j])-ndtr(-mu[j]/sd[j]))/(ndtr((1-mu[j])/sd[j])-ndtr(-mu[j]/sd[j]))
  cdfs.append(float(np.max(np.abs(model.empirical_cdf(x[:,j],KNOTS)-exact))))
 q=float(np.max(np.abs(np.quantile(x,[.05,.5,.95],axis=0)-exactq)))
 return dict(max_CDF_difference=max(cdfs),max_unit_quantile_difference=q,passed=max(cdfs)<=.04 and q<=.02)


def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.startswith('_')}
 if isinstance(x,list):return [clean(v) for v in x]
 return x


def phase(protocol,out,obs,forward,dimension):
 name='conditional2D' if dimension==2 else 'full4D';cap=protocol['phase_caps_s'][name]
 p=out/name;p.mkdir();start=time.monotonic();state=dict(phase=name,completed=False,cap_s=cap,events=[],passed=False)
 def save():write(p/'summary.json',clean(state))
 def alarm(_s,_f):raise BudgetReached('Declared phase wall cap reached')
 old=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,cap)
 try:
  dims=(0,3) if dimension==2 else (0,1,2,3)
  if dimension==2:
   state['analytic_limit']=analytic_check();refs=[]
   for i,(rtol,atol) in enumerate(TOLERANCES):
    ref=adaptive_reference(forward,obs,rtol,atol,p,'adaptive'+str(i));refs.append(ref)
    state['adaptive_reference']=refs;state['events'].append('adaptive'+str(i)+' completed');save()
   delta=comparison(refs[0],refs[1]);fields=delta['fields']
   state['adaptive_tolerance_agreement']=dict(max_CDF_difference=delta['max_CDF_difference'],max_unit_quantile_difference=delta['max_unit_quantile_difference'],fields=fields,
    passed=delta['max_CDF_difference']<=2e-4 and delta['max_unit_quantile_difference']<=1e-4 and all(max(x['mean_RMS_m'],x['std_RMS_m'])<=1e-7 for x in fields.values()))
  runs=[]
  for count in COUNTS:
   for seed in SEEDS:
    result=model.smc(forward,dims,obs,count,seed);label=f'smc-{count}-{seed}'
    np.savez_compressed(p/(label+'.npz'),unit=result.state.coordinates,log_likelihood=result.log_likelihood)
    summary=summaries(forward,result.state.coordinates,dims,p,label)
    runs.append(dict(particles=count,seed=seed,SMC=result.diagnostics(),summary=summary))
    state['SMC_runs']=runs;state['events'].append(label+' completed');save()
  pilot=runs[-1];raw=np.load(p/f"smc-2048-{SEEDS[-1]}.npz",allow_pickle=False)
  chain,diagnostics=model.continuous_mcmc(lambda x:forward.log_likelihood(x,dims,obs),raw['unit'],seed=2026121110 if dimension==2 else 2026121120,
   warmup=500 if dimension==2 else 1000,draws=2000 if dimension==2 else 4000,
   progress=lambda step,total:write(p/'progress.json',dict(phase=name,step=step,planned=total,elapsed_s=time.monotonic()-start)))
  raw.close();np.savez_compressed(p/'MCMC.npz',unit=chain)
  mcmc=summaries(forward,chain.reshape(-1,dimension),dims,p,'MCMC');state['MCMC']=dict(**diagnostics,summary=mcmc)
  state['SMC_pairwise_comparisons']=[dict(first=[a['particles'],a['seed']],second=[b['particles'],b['seed']],**comparison(a['summary'],b['summary'])) for i,a in enumerate(runs) for b in runs[i+1:]]
  state['MCMC_vs_SMC']=[dict(particles=r['particles'],seed=r['seed'],**comparison(r['summary'],mcmc)) for r in runs]
  if dimension==2:
   state['vs_adaptive_reference']=[dict(kind='SMC',particles=r['particles'],seed=r['seed'],**comparison(r['summary'],refs[-1])) for r in runs]+[dict(kind='MCMC',**comparison(mcmc,refs[-1]))]
  checks=[x['passed'] for x in state['SMC_pairwise_comparisons']+state['MCMC_vs_SMC']]
  checks+=[diagnostics['diagnostics']['all_monitored_quantities_pass']]
  if dimension==2:checks += [state['analytic_limit']['passed'],state['adaptive_tolerance_agreement']['passed']]+[x['passed'] for x in state['vs_adaptive_reference']]
  state.update(completed=True,passed=all(checks),status='COMPLETED_ALL_DECLARED_CHECKS')
 except BudgetReached as e:state.update(status='DECLARED_WALL_CAP',error=str(e))
 except Exception as e:state.update(status='PRESERVED_FAILURE',error_type=type(e).__name__,error=str(e))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);state['elapsed_s']=time.monotonic()-start;save()
 return state


def run(protocol_path,digest,out):
 protocol=json.loads(check(dict(path=protocol_path,sha256=digest)).read_text())
 for path,h in protocol['source_sha256'].items():check(dict(path=path,sha256=h))
 check(protocol['original_design'])
 if protocol['seeds']!=list(SEEDS) or protocol['particles']!=list(COUNTS) or protocol['CPU_threads']!=2 or protocol['phase_caps_s']!={'conditional2D':180,'full4D':360}:raise ValueError('Declared finite plan required')
 out=Path(out)
 if out.exists():raise ValueError('Fresh output; preserve all prior attempts')
 out.mkdir(parents=True);write(out/'protocol.json',protocol);torch.set_num_threads(2)
 with np.load(check(protocol['observation_archive']),allow_pickle=False) as data:obs=data['observation'].copy()
 forward=model.ExactVectorForward();two=phase(protocol,out,obs,forward,2)
 result=dict(schema='manufactured-continuous-reference-results-v0.1',protocol=dict(path=protocol_path,sha256=digest),conditional2D=clean(two),
  full4D_started=two['passed'],physical_calibration_claim=False,continuous_accuracy_proven=False,old_grid_failures_unchanged=True,raw_arrays_LOCAL=True)
 if two['passed']:result['full4D']=clean(phase(protocol,out,obs,forward,4))
 else:result['full4D_not_started_reason']='2D independent validation gate did not pass; no claim of4D validation'
 result['numerical_validation_passed']=two['passed'] and result.get('full4D',{}).get('passed',False)
 write(out/'summary.json',result)
 return dict(conditional2D_completed=two['completed'],conditional2D_passed=two['passed'],full4D_started=result['full4D_started'],numerical_validation_passed=result['numerical_validation_passed'])


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
 a=sub.add_parser('prepare');a.add_argument('--output',required=True)
 a=sub.add_parser('run')
 for name in ('protocol','protocol-sha256','output'):a.add_argument('--'+name,required=True)
 args=parser.parse_args();value=prepare(args.output) if args.mode=='prepare' else run(args.protocol,args.protocol_sha256,args.output)
 print(json.dumps(value))
