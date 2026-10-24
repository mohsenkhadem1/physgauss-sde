"""Prospective finite-model cohort and memory-bounded joint force predictions.

No I/O, truth generation on import, sampling or automatic experiment execution.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import qmc
from experiments import campaign_compliant_force_challenge_v01 as old
from experiments import campaign_compliant_ridge_reference as ridge

DIMS=(0,1,2,3)
TARGET_FRAMES=(30,45,60)
BATCHES=(100,200,400,800,1600)


def cohort_units(count=12,seed=2026102404):
    """Fixed Latin hypercube in the original intrinsic uniform box; no selection."""
    if count!=12 or seed!=2026102404:raise ValueError('Only the prospective frozen cohort')
    return qmc.LatinHypercube(d=4,seed=seed).random(count)


def physical(unit):
    unit=np.asarray(unit,float)
    if unit.ndim!=2 or unit.shape[1]!=4 or not np.isfinite(unit).all()or np.any((unit<0)|(unit>1)):
        raise ValueError('All four original intrinsic coordinates, within prior')
    # Build all coordinates explicitly: no implicit original-TRUTH defaults.
    value=old.original.LOWER+unit*(old.original.UPPER-old.original.LOWER)
    result=value.copy();result[:,[0,2,3]]=np.exp(value[:,[0,2,3]])
    return result


def joint_features(forward,theta,*,chunk=1024,with_functions=False):
    """Exact nine-force multi-RHS implicit Euler; retain only135+72 features.

    No time/space reduction of the solver: all60 original steps are executed.
    Only output clocks are selected, prospectively. Cholesky is shared between
    nine force programs because these share the same physical parameter draw.
    """
    theta=np.asarray(theta,float)
    if theta.ndim!=2 or theta.shape[1]!=4 or not len(theta)or not np.isfinite(theta).all()or np.any(theta[:,[0,2,3]]<=0):raise ValueError('Finite physical theta[B,4]')
    programs=list(old.force_programs().values())+list(old.poke_programs().values())
    if with_functions:programs.extend([old.original.known_force('A'),old.original.known_force('B')])
    rhs_count=len(programs)
    held=torch.tensor(np.stack(programs).reshape(rhs_count,60,15).transpose(1,2,0))
    targets=[];pokes=[];functions=[];h=old.original.H;m=forward.mass
    for first in range(0,len(theta),chunk):
        values=theta[first:first+chunk];n=len(values)
        centers=np.column_stack([values[:,1],np.full(n,.02),np.zeros(n)])
        logw=np.log(forward.areas)[None]-.5*np.sum((forward.X[None]-centers[:,None])**2,axis=2)/values[:,2,None]**2
        w=np.exp(logw-logsumexp(logw,axis=1)[:,None]);d=torch.tensor(np.repeat(w,3,axis=1))
        K=torch.tensor(values[:,0,None,None])*forward.Kunit+torch.diag_embed(d*torch.tensor(values[:,3,None]));C=.05*d
        chol=torch.linalg.cholesky(K+torch.diag_embed(m[None]/h**2+C/h))
        u=torch.zeros((n,15,rhs_count),dtype=torch.float64);v=u.clone();target=[];poke=[]
        for step in range(1,61):
            rhs=held[step-1][None]+m[None,:,None]*(u/h**2+v/h)+C[:,:,None]*u/h
            new=torch.cholesky_solve(rhs,chol);v=(new-u)/h;u=new
            if step in TARGET_FRAMES:target.append(u[:,:,:3].numpy().transpose(0,2,1).copy())
            if step in(12,24):poke.append(u[:,:,3:9].reshape(n,5,3,6)[:,old.original.OBS_NODES].numpy().transpose(0,3,1,2).copy())
        # Program-major / frame / node / xyz; matches independent scalar calls.
        targets.append(np.stack(target,axis=2).reshape(n,135))
        pokes.append(np.stack(poke,axis=2).reshape(n,6,12))
        if with_functions:
            field=u.numpy().reshape(n,5,3,rhs_count)
            functions.append(np.concatenate([ridge.fixture_functions(values,forward.areas,forward.X),field[:,1,:,9],field[:,4,:,10]],axis=1))
    result=(np.concatenate(targets),np.concatenate(pokes).transpose(1,0,2))
    return (*result,np.concatenate(functions))if with_functions else result


def weighted_ratio_mcse(chain,weights):
    """Bounded-memory ratio MCSE using equal100-draw sufficient-statistic blocks.

    Preserve original chain boundaries. Max over100/200/400/800/1600 batches
    and independent-chain clusters, including denominator covariance.
    """
    from experiments.campaign_force_prediction_precision_v01 import combine,delta_mcse
    x=np.asarray(chain,float);w=np.asarray(weights,float)
    if x.ndim!=3 or not np.isfinite(x).all() or w.shape!=x.shape[:2] or any(x.shape[1]%k for k in BATCHES) or np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0:raise ValueError('Finite complete weighted chains')
    c,n,d=x.shape;masses=[];means=[];variances=[];total=w.sum()
    for chain_index in range(c):
        for first in range(0,n,1600):
            z=x[chain_index,first:first+1600].reshape(16,100,d)
            ww=w[chain_index,first:first+1600].reshape(16,100);mass=ww.sum(1)
            if np.any(mass<=0):raise ValueError('Zero100-draw block mass: unresolved')
            anchor=z[:,0];delta=z-anchor[:,None];mu=np.einsum('bn,bnd->bd',ww,delta)/mass[:,None]
            var=np.einsum('bn,bnd->bd',ww,(delta-mu[:,None])**2)/mass[:,None]
            masses.extend(mass/total);means.extend(anchor+mu);variances.extend(var)
    mass=np.asarray(masses)[:,None];mean=np.asarray(means)[:,None];var=np.asarray(variances)[:,None]
    estimates=[];blocks_per_chain=n//100
    for k in BATCHES:
        group=k//100;groups=[list(range(i,i+group))for i in range(0,len(mass),group)]
        estimates.append(delta_mcse(*combine(mass,mean,var,groups)))
    chain_groups=[list(range(i*blocks_per_chain,(i+1)*blocks_per_chain))for i in range(c)]
    half=blocks_per_chain//2
    half_groups=[list(range(i*half,(i+1)*half))for i in range(2*c)]
    half_mass,half_mean,_=combine(mass,mean,var,half_groups)
    cluster_means=np.concatenate([half_mean[:,0],combine(mass,mean,var,chain_groups)[1][:,0]])
    estimates.append(delta_mcse(*combine(mass,mean,var,chain_groups)))
    return dict(mean=estimates[0]['mean'][0],variance=estimates[0]['variance'][0],mean_MCSE=np.maximum.reduce([v['mean_MCSE'][0]for v in estimates]),SD_MCSE=np.maximum.reduce([v['SD_MCSE'][0]for v in estimates]),raw_ESS=float(total**2/np.square(w).sum()),minimum_block_weight_ESS=float(min(v['block_weight_ESS'][0]for v in estimates)),maximum_chain_mass=float(estimates[-1]['maximum_block_mass'][0]),half_block_weight_ESS=float(1/np.square(half_mass).sum()),cluster_means=cluster_means)


def batch_moment_mcse(chain):
    x=np.asarray(chain,float)
    return weighted_ratio_mcse(x,np.ones(x.shape[:2]))


def moment_gate(row):
    tol=max(2e-6,.03*float(np.sqrt(np.mean(row['variance']))))
    mean=float(np.sqrt(np.mean(row['mean_MCSE']**2)));sd=float(np.sqrt(np.mean(row['SD_MCSE']**2)))
    return dict(passed=bool(mean<=tol and sd<=tol and row.get('half_block_weight_ESS',16)>=8),half_block_weight_ESS=float(row.get('half_block_weight_ESS',16)),mean_RMS_MCSE_m=mean,SD_RMS_MCSE_m=sd,tolerance_m=tol)


def compare_moments(row,reference):
    tol=max(2e-6,.03*float(np.sqrt(np.mean(reference['variance']))))
    dm=float(np.sqrt(np.mean((row['mean']-reference['mean'])**2)));ds=float(np.sqrt(np.mean((np.sqrt(row['variance'])-np.sqrt(reference['variance']))**2)))
    return dict(passed=bool(dm<=tol and ds<=tol),mean_RMS_difference_m=dm,SD_RMS_difference_m=ds,tolerance_m=tol)


def cluster_comparison(active,baseline,*,passed,seed=2026102405,resamples=4000):
    """Resample whole object clusters; paired noise stays within its object.

    Conditional designed-cohort diagnostic CI: LHS objects are not IID draws.
    Anchor is excluded by caller. Numerical failures never drop observations.
    """
    a=np.asarray(active,float);b=np.asarray(baseline,float)
    if a.shape!=(12,16)or b.shape!=a.shape:raise ValueError('Exactly12newobjects,16pairednoise')
    delta=(a-b).mean(1);rng=np.random.default_rng(seed);boot=delta[rng.integers(12,size=(resamples,12))].mean(1)
    ci=np.quantile(boot,[.025,.975]);return dict(mean_difference_m=float(delta.mean()),designed_cohort_cluster_bootstrap_95pct_CI_m=ci.tolist(),objects=12,paired_noise_per_object=16,numerical_gates_all_passed=bool(passed),superiority_supported=bool(passed and ci[1]<0),population_calibration_claim=False,CI_scope='Conditional designed-LHS-cohort diagnostic; numerical Monte Carlo error assessed separately')
