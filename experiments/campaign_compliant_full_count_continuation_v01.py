"""Full4D count-only exact MH continuation, including RNG-only state recovery.

No posterior is evaluated while recovering RNG. The predecessor's RNG stream
has state-independent draw shapes except the number of labelled prior jumps,
which is itself recoverable from that stream. Old accepted states and exact
log likelihoods are required; no acceptance replay or new initialization.
"""
import copy
import numpy as np
import torch
from experiments.campaign_compliant_ridge_reference import *
from physgauss.chain_diagnostics import diagnose_quantities


def recover_rng(seed,initial,steps):
 initial=np.asarray(initial);chains,dimension=initial.shape
 rng=np.random.default_rng(seed)
 if not np.array_equal(rng.uniform(size=initial.shape),initial):
  raise ValueError('Original independent-prior initial states must match RNG')
 for _ in range(steps):
  kind=rng.choice(3,size=chains,p=[.5,.45,.05])
  rng.normal(size=(chains,dimension));rng.normal(size=(chains,dimension))
  rng.uniform(size=(int((kind==2).sum()),dimension));rng.random(chains)
 return rng


def continue_mh(loglik,previous,*,additional_draws,checkpoint=None):
 """Append one prospectively fixed count, preserving original time order."""
 old=np.asarray(previous['unit'],float);r=previous['MH'];C,N,D=old.shape
 if C!=8 or N!=4000 or D!=4 or r['seed']!=2026121120 or r['warmup']!=1000 or r['draws']!=N:
  raise ValueError('Exact predecessor full4D chain geometry and seed')
 if additional_draws!=60000:raise ValueError('One fixed additional60000 draws/chain')
 if r['proposal_frozen_before_production']is not True or r['correlated_RW_outside_box_rejected']is not True or r['coordinate_reflection']is not False or r['rw_scale']!=.5:
  raise ValueError('Unchanged original fixed rejection proposal')
 likelihood=np.asarray(r['log_likelihood'],float)
 if likelihood.shape!=(C,N)or not np.isfinite(old).all()or not np.isfinite(likelihood).all()or np.any(old<0)or np.any(old>1):raise ValueError('Complete finite saved predecessor states')
 mean=np.asarray(r['proposal_mean']);cov=np.asarray(r['proposal_covariance']);L=np.linalg.cholesky(cov);inverse=np.linalg.inv(cov)
 rng=recover_rng(r['seed'],r['initial_states'],r['warmup']+N)
 recovered=copy.deepcopy(rng.bit_generator.state);x=old[:,-1].copy();ll=likelihood[:,-1].copy()
 if checkpoint is not None:checkpoint(dict(unit=x.copy(),log_likelihood=ll.copy(),proposal_mean=mean.copy(),proposal_covariance=cov.copy(),RNG_state=copy.deepcopy(recovered),previous_draws=N))
 chain=np.empty((C,N+additional_draws,D));chain[:,:N]=old;savedll=np.empty((C,N+additional_draws));savedll[:,:N]=likelihood
 proposed=np.zeros(3,int);accepted=proposed.copy();outside=proposed.copy()
 def logq(z):return -.5*np.einsum('bi,ij,bj->b',z-mean,inverse,z-mean)
 for step in range(additional_draws):
  kind=rng.choice(3,size=C,p=[.5,.45,.05]);candidate=x+.5*(rng.normal(size=x.shape)@L.T)
  independent=mean+rng.normal(size=x.shape)@L.T;candidate[kind==0]=independent[kind==0];candidate[kind==2]=rng.uniform(size=(int((kind==2).sum()),D))
  valid=((candidate>=0)&(candidate<=1)).all(1);newll=np.full(C,-np.inf)
  if valid.any():newll[valid]=loglik(candidate[valid])
  if np.isnan(newll).any()or np.isposinf(newll).any():raise FloatingPointError('Invalid proposed exact likelihood')
  correction=np.where(kind==0,logq(x)-logq(candidate),0.);accept=valid&(np.log(rng.random(C))<np.minimum(0.,newll-ll+correction));x[accept]=candidate[accept];ll[accept]=newll[accept]
  for k in range(3):proposed[k]+=int((kind==k).sum());accepted[k]+=int((accept&(kind==k)).sum());outside[k]+=int((~valid&(kind==k)).sum())
  chain[:,N+step]=x;savedll[:,N+step]=ll
 report=diagnose_quantities(torch.tensor(np.concatenate([chain,savedll[:,:,None]],2)),['unit_'+str(i)for i in range(D)]+['log_likelihood'])
 eig,V=np.linalg.eigh(cov);ridge=(chain-mean)@V/np.sqrt(eig);ridge_report=diagnose_quantities(torch.tensor(ridge),['pilot_ridge_'+str(i)for i in range(D)])
 result=dict(r);result.update(diagnostics=report,ridge_diagnostics=ridge_report,draws=N+additional_draws,log_likelihood=savedll,continuation=dict(previous_retained=N,additional_retained=additional_draws,new_warmup=0,new_seed=False,new_independent_chains=False,RNG_recovery='Exact RNG-only replay; no likelihood or acceptance replay',RNG_start=recovered,RNG_end=copy.deepcopy(rng.bit_generator.state),additional_proposed=proposed,additional_accepted=accepted,additional_outside_rejected=outside))
 # Legacy two-phase counts retain original warmup and aggregate production.
 for key,extra in [('proposed',proposed),('accepted',accepted),('outside_rejected',outside)]:
  result[key]=np.asarray(r[key]).copy();result[key][1]+=extra
 return chain,result


def long_batch_mc(chain):
 """Additional conservative same-threshold gates for800/1600 time batches."""
 x=np.asarray(chain,float);C,N,D=x.shape;flat=x.reshape(-1,D);q=np.quantile(flat,[.05,.5,.95],axis=0).T;knots=np.linspace(0,1,33);cdf=[];quant=[]
 for j in range(D):
  cse=[];qse=[]
  for k in (800,1600):
   if N%k:raise ValueError('Complete prospective long batches')
   b=(x[:,:,j,None]<=knots).reshape(C,N//k,k,33).mean(2).reshape(-1,33);cse.append(b.std(0,ddof=1)/np.sqrt(len(b)))
  for threshold in q[j]:
   errors=[]
   for k in (800,1600):
    b=(x[:,:,j]<=threshold).reshape(C,N//k,k).mean(2).reshape(-1);errors.append(b.std(ddof=1)/np.sqrt(len(b)))
   lo=max(0.,threshold-.01);hi=min(1.,threshold+.01);density=np.mean((flat[:,j]>lo)&(flat[:,j]<=hi))/(hi-lo);qse.append(float(max(errors)/density)if density>0 else None)
  cdf.append(np.max(cse,axis=0));quant.append(qse)
 return dict(CDF_batch_MCSE=np.asarray(cdf),quantile_MCSE=quant,time_batch_lengths=[800,1600],passed=bool(np.max(cdf)<=.01 and all(v is not None and v<=.005 for row in quant for v in row)))
