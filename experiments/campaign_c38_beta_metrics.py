"""Numerical/held-calibration diagnostics for the fixed narrow density control."""
from __future__ import annotations
import numpy as np
from scipy.special import betaln,logsumexp
from scipy.stats import beta
from experiments.campaign_c38_beta_posterior import numpy_parameters,marginal_cdf,marginal_quantiles
from experiments.campaign_c38_bar_reference import LOWER,UPPER,SIGMA_M,endpoint_displacements


def log_density_numpy(unit,parameters,index=0):
    u=np.asarray(unit,dtype=float);p=numpy_parameters(parameters)
    if u.ndim!=2 or u.shape[1]!=2 or not np.isfinite(u).all() or ((u<=0)|(u>=1)).any():raise ValueError('Interior intrinsic coordinates required')
    a=p['alpha'][index];b=p['beta'][index];logpi=np.log(p['weights'][index]);terms=((a[None]-1)*np.log(u[:,None])+(b[None]-1)*np.log1p(-u[:,None])-betaln(a,b)[None]).sum(-1)
    return logsumexp(logpi[None]+terms,axis=1)


def reference_log_density(unit,posterior,index):
    """Posterior density wrt dunitE dunitK, not dc or dphysicalE dK."""
    u=np.asarray(unit,dtype=float);theta=np.exp(LOWER+u*(UPPER-LOWER));p=endpoint_displacements(theta,posterior.F)[:,np.arange(len(posterior.F)),posterior.sides]
    logL=-.5*np.sum(((p-posterior.y[index])/SIGMA_M)**2,axis=1)
    logZ=np.log(posterior.Z[index])+np.log(posterior.std*np.sqrt(2*np.pi)/(UPPER[1]-LOWER[1]))-posterior.shift[index]
    return logL-logZ


def reference_joint_kl(posterior,index,parameters,param_index,*,outer_order,conditional_order):
    theta,w,cloud=posterior.quadrature_cloud(index,outer_order=outer_order,conditional_order=conditional_order);unit=(np.log(theta)-LOWER)/(UPPER-LOWER)
    difference=reference_log_density(unit,posterior,index)-log_density_numpy(unit,parameters,param_index);value=float(w@difference)
    return value,cloud


def exact_coverage(covered,*,alpha=.05):
    flags=np.asarray(covered,dtype=bool);n=len(flags);s=int(flags.sum())
    if n<1 or not 0<alpha<1:raise ValueError('Nonempty independent-object sample and confidence alpha required')
    return dict(successes=s,objects=n,fraction=s/n,interval=[0. if s==0 else float(beta.ppf(alpha/2,s,n-s+1)),1. if s==n else float(beta.ppf(1-alpha/2,s+1,n-s))])


def parameter_diagnostics(parameters,unit,*,contexts=8,family_tests=48):
    """Every context has N independent objects; never use N*contexts as N."""
    truth=np.asarray(unit,dtype=float);B=len(truth)
    if truth.shape!=(B,2) or numpy_parameters(parameters)['weights'].shape[0]!=B*contexts:raise ValueError('Object-major complete context records required')
    repeated=np.repeat(truth,contexts,axis=0);pit=marginal_cdf(parameters,repeated).reshape(B,contexts,2);quant=marginal_quantiles(parameters,(.05,.25,.5,.75,.95)).reshape(B,contexts,2,5)
    threshold=np.sqrt(np.log(2*family_tests/.05)/(2*B));rows=[];allpass=True
    for c in range(contexts):
        record={}
        for j,key in enumerate(('E','K')):
            x=np.sort(pit[:,c,j]);ks=float(max(np.max(np.arange(1,B+1)/B-x),np.max(x-np.arange(B)/B)));covered=(truth[:,j]>=quant[:,c,j,0])&(truth[:,j]<=quant[:,c,j,4]);family=exact_coverage(covered,alpha=.05/family_tests)
            passed=ks<=threshold and family['interval'][0]<=.9<=family['interval'][1];allpass&=passed
            covered50=(truth[:,j]>=quant[:,c,j,1])&(truth[:,j]<=quant[:,c,j,3])
            physical_interval=np.exp(LOWER[j]+quant[:,c,j,[0,4]]*(UPPER[j]-LOWER[j]))
            record[key]=dict(coverage90=exact_coverage(covered),coverage50=exact_coverage(covered50),familywise_coverage90=family,PIT_KS_distance=ks,PIT_DKW_family_threshold=float(threshold),PIT_histogram_10bins=np.histogram(pit[:,c,j],np.linspace(0,1,11))[0].tolist(),mean_interval_width_unit=float(np.mean(quant[:,c,j,4]-quant[:,c,j,0])),mean_interval_width_physical=float(np.mean(physical_interval[:,1]-physical_interval[:,0])),physical_unit='Pa' if j==0 else 'N/m',gate_passed=bool(passed))
        rows.append(record)
    return dict(objects=B,contexts=contexts,independent_objects_per_context=B,pooled_frames_are_not_independent=True,context_results=rows,calibration_gate_passed=bool(allpass)),quant,pit
