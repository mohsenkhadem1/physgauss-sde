"""Continuous exact-model fixture reference primitives; no I/O or actual data.

Correlated Gaussian random walks use out-of-box REJECTION, never reflection.
Proposal covariance is fixed from the designated pilot before production.
"""
import numpy as np
import torch
from scipy.special import logsumexp
from experiments import campaign_compliant_continuous_reference as base
from experiments import campaign_compliant_patch_inverse as original
from physgauss.chain_diagnostics import diagnose_quantities


def ridge_mh(loglik,pilot,*,seed,warmup,draws,chains=8,rw_scale=.5,freeze_proposal=None):
 pilot=np.asarray(pilot,float)
 if pilot.ndim!=2 or not len(pilot)or not np.isfinite(pilot).all()or np.any(pilot<0)or np.any(pilot>1):raise ValueError('Fixed in-prior pilot')
 mean=pilot.mean(0);cov=np.cov(pilot,rowvar=False);L=np.linalg.cholesky(cov);inverse=np.linalg.inv(cov);proposal_artifact=freeze_proposal(mean.copy(),cov.copy())if freeze_proposal is not None else None;dimension=pilot.shape[1];rng=np.random.default_rng(seed)
 x=rng.uniform(size=(chains,dimension));initial=x.copy();ll=np.asarray(loglik(x));saved=[];savedll=[];proposed=np.zeros((2,3),int);accepted=proposed.copy();outside=proposed.copy()
 if ll.shape!=(chains,)or not np.isfinite(ll).all():raise FloatingPointError('Finite exact initial likelihoods')
 def logq(z):return -.5*np.einsum('bi,ij,bj->b',z-mean,inverse,z-mean)
 for step in range(warmup+draws):
  kind=rng.choice(3,size=chains,p=[.5,.45,.05]);candidate=x+rw_scale*(rng.normal(size=x.shape)@L.T)
  independent=mean+rng.normal(size=x.shape)@L.T;candidate[kind==0]=independent[kind==0];candidate[kind==2]=rng.uniform(size=(int((kind==2).sum()),dimension))
  valid=((candidate>=0)&(candidate<=1)).all(1);newll=np.full(chains,-np.inf)
  if valid.any():newll[valid]=loglik(candidate[valid])
  if np.isnan(newll).any()or np.isposinf(newll).any():raise FloatingPointError('Invalid proposed exact likelihood')
  correction=np.where(kind==0,logq(x)-logq(candidate),0.);accept=valid&(np.log(rng.random(chains))<np.minimum(0.,newll-ll+correction));x[accept]=candidate[accept];ll[accept]=newll[accept];phase=int(step>=warmup)
  for k in range(3):proposed[phase,k]+=int((kind==k).sum());accepted[phase,k]+=int((accept&(kind==k)).sum());outside[phase,k]+=int((~valid&(kind==k)).sum())
  if phase:saved.append(x.copy());savedll.append(ll.copy())
 chain=np.stack(saved,1);likelihood=np.stack(savedll,1);report=diagnose_quantities(torch.tensor(np.concatenate([chain,likelihood[:,:,None]],axis=2)),['unit_'+str(i)for i in range(dimension)]+['log_likelihood'])
 eig,V=np.linalg.eigh(cov);ridge=(chain-mean)@V/np.sqrt(eig);ridge_report=diagnose_quantities(torch.tensor(ridge),['pilot_ridge_'+str(i)for i in range(dimension)])
 return chain,dict(diagnostics=report,ridge_diagnostics=ridge_report,initial_states=initial,proposal_mean=mean,proposal_covariance=cov,proposal_frozen_before_production=True,proposal_artifact=proposal_artifact,correlated_RW_outside_box_rejected=True,coordinate_reflection=False,rw_scale=rw_scale,accepted=accepted,proposed=proposed,outside_rejected=outside,chains=chains,warmup=warmup,draws=draws,seed=seed,log_likelihood=likelihood)


def cdf_quantile_mc(chain,*,batch=100):
 x=np.asarray(chain,float);C,N,D=x.shape;levels=(100,200,400)
 if any(N%k for k in levels):raise ValueError('Fixed complete100/200/400 time batches')
 knots=np.linspace(0,1,33);flat=x.reshape(-1,D);q=np.quantile(flat,[.05,.5,.95],axis=0).T;cdf=np.mean(flat[:,:,None]<=knots[None,None,:],axis=0);allse=[];qse=[]
 for j in range(D):
  se=[]
  for k in levels:
   means=(x[:,:,j,None]<=knots).reshape(C,N//k,k,33).mean(2).reshape(-1,33);se.append(means.std(0,ddof=1)/np.sqrt(len(means)))
  allse.append(np.max(se,axis=0));errors=[]
  for threshold in q[j]:
   estimates=[]
   for k in levels:
    ind=(x[:,:,j]<=threshold).reshape(C,N//k,k).mean(2).reshape(-1);estimates.append(ind.std(ddof=1)/np.sqrt(len(ind)))
   lo=max(0.,threshold-.01);hi=min(1.,threshold+.01);density=np.mean((flat[:,j]>lo)&(flat[:,j]<=hi))/(hi-lo);errors.append(float(max(estimates)/density)if density>0 else None)
  qse.append(errors)
 return dict(unit_CDF=cdf,unit_quantiles=q,CDF_batch_MCSE=np.asarray(allse),quantile_MCSE=qse,MCSE_scope='Maximum aligned100/200/400-draw batch means and fixed local density window; estimated sampling error, not rigorous posterior certificate',independent_objects=False)


def fixture_functions(theta,areas,X):
 theta=np.asarray(theta);center=np.column_stack([theta[:,1],np.full(len(theta),.02),np.zeros(len(theta))]);logw=np.log(areas)[None]-.5*np.sum((X[None]-center[:,None])**2,2)/theta[:,2,None]**2;w=np.exp(logw-logsumexp(logw,axis=1)[:,None]);centroid=w@X
 return np.concatenate([theta,w*theta[:,3,None],centroid],axis=1)


def predictions(forward,unit,dims,*,chain=False,batch=100):
 """Positive batch-centered variance; MCMC batch-MCSE, no IID SMC claim."""
 unit=np.asarray(unit);shape=unit.shape;flat=unit.reshape(-1,len(dims));means={k:[]for k in ('A','B')};vars={k:[]for k in ('A','B')};counts=[];functions=[]
 if chain and(shape[1]%batch or batch!=100):raise ValueError('Prospective complete100-draw time batches')
 width=batch if chain else 512
 for first in range(0,len(flat),width):
  theta=base.physical(flat[first:first+width],dims);a=forward(theta,'A');b=forward(theta,'B');n=len(theta);counts.append(n)
  if not np.isfinite(a).all()or not np.isfinite(b).all():raise FloatingPointError('Nonfinite prescribed predictive field')
  functions.append(np.concatenate([fixture_functions(theta,forward.areas,forward.X),a[:,-1,1],b[:,-1,4]],axis=1))
  for label,u in [('A',a),('B',b)]:
   mu=u.mean(0);means[label].append(mu);vars[label].append(np.mean((u-mu)**2,axis=0))
 result={};n=np.asarray(counts,float)
 for label in ('A','B'):
  m=np.asarray(means[label]);v=np.asarray(vars[label]);mean=np.einsum('b,bijk->ijk',n/n.sum(),m);second=v+(m-mean)**2;variance=np.einsum('b,bijk->ijk',n/n.sum(),second);std=np.sqrt(variance);row=dict(mean=mean,std=std)
  if chain:
   se_means=[];se_stds=[]
   for level in (100,200,400):
    group=level//100;bm=m.reshape(shape[0],shape[1]//level,group,*m.shape[1:]).mean(2).reshape(-1,*m.shape[1:]);bs=second.reshape(shape[0],shape[1]//level,group,*second.shape[1:]).mean(2).reshape(-1,*second.shape[1:]);se_mean=bm.std(0,ddof=1)/np.sqrt(len(bm));se_var=bs.std(0,ddof=1)/np.sqrt(len(bs));se_std=np.zeros_like(std);positive=std>0
    if np.any((~positive)&(se_var>0)):raise FloatingPointError('Unresolved zero-variance MC error')
    se_std[positive]=se_var[positive]/(2*std[positive]);se_means.append(se_mean);se_stds.append(se_std)
   row.update(mean_batch_MCSE=np.max(se_means,axis=0),std_delta_batch_MCSE=np.max(se_stds,axis=0))
  result[label]=row
 f=np.concatenate(functions)
 if chain:
  f=f.reshape(shape[0],shape[1],-1);names=['E','center','width','K']+['fixtureK_node'+str(i)for i in range(5)]+['fixture_centroid_'+a for a in 'xyz']+['A_end_node1_'+a for a in 'xyz']+['B_end_node4_'+a for a in 'xyz'];known=[1,2,9,10,11]if tuple(dims)==(0,3)else[];unknown=[i for i in range(len(names))if i not in known]
  result['function_diagnostics']=diagnose_quantities(torch.tensor(f[:,:,unknown]),[names[i]for i in unknown]);result['known_fixed_functions']=[names[i]for i in known];fb=f.reshape(shape[0],shape[1]//batch,batch,-1).mean(2).reshape(-1,f.shape[-1]);result['function_batch_mean_MCSE']=np.max([f.reshape(shape[0],shape[1]//k,k,-1).mean(2).reshape(-1,f.shape[-1]).std(0,ddof=1)/np.sqrt(shape[0]*(shape[1]//k))for k in (100,200,400)],axis=0)
 result['function_means']=np.concatenate(functions).mean(0);result['function_std']=np.concatenate(functions).std(0);result['function_unknown_indices']=[i for i in range(18)if not(tuple(dims)==(0,3)and i in(1,2,9,10,11))];result['scope']='Exact fixed discrete linear model; physical calibration/continuum error not assessed; SMC uncertainty assessed across independent seeds, not uniform post-resampling weights';return result


def compare(a,b):
 fields={}
 for label,start in [('A',25),('B',0)]:
  x=a['predictions'][label];y=b['predictions'][label];tol=max(2e-6,.03*float(np.sqrt(np.mean(y['std'][start:]**2))));dm=float(np.sqrt(np.mean((x['mean'][start:]-y['mean'][start:])**2)));ds=float(np.sqrt(np.mean((x['std'][start:]-y['std'][start:])**2)));fields[label]=dict(mean_RMS_m=dm,std_RMS_m=ds,tolerance_m=tol,passed=dm<=tol and ds<=tol)
 bc_pass=True;bc_error=None
 if 'function_std'in b['predictions']:
  ids=b['predictions']['function_unknown_indices'];error=abs(a['predictions']['function_means']-b['predictions']['function_means'])[ids];sd=b['predictions']['function_std'][ids];bc_error=float(np.max(np.divide(error,sd,out=np.where(error==0,0.,np.inf),where=sd>0)));bc_pass=bc_error<=.1
 dc=float(np.max(abs(a['unit_CDF']-b['unit_CDF'])));dq=float(np.max(abs(a['unit_quantiles']-b['unit_quantiles'])));return dict(max_CDF_difference=dc,max_unit_quantile_difference=dq,fields=fields,maximum_function_mean_difference_over_posterior_SD=bc_error,passed=bc_pass and dc<=.05 and dq<=.02 and all(v['passed']for v in fields.values()))
