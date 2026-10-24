"""Block delta diagnostics for a full self-normalized importance estimator.

No sampling, forward solves, truth access, or production I/O. Batch MCSE is an
asymptotic diagnostic, not a finite-sample error bound or calibration proof.
"""
from __future__ import annotations
import numpy as np


def combine(mass, mean, variance, groups=None):
    mass=np.asarray(mass,float);mean=np.asarray(mean,float);variance=np.asarray(variance,float)
    if mass.ndim!=2 or mean.shape!=variance.shape or mean.shape[:2]!=mass.shape or mean.ndim!=3:
        raise ValueError('mass[B,K], mean/variance[B,K,D] required')
    if not all(np.isfinite(x).all() for x in (mass,mean,variance)) or (mass<0).any() or (variance<0).any():
        raise ValueError('Finite nonnegative masses and variances required')
    if (mass<=0).any():raise ValueError('Zero block mass: diagnostic unresolved')
    if not np.allclose(mass.sum(0),1,atol=2e-12,rtol=0):raise ValueError('Global normalized masses required')
    if groups is None:groups=[[i] for i in range(len(mass))]
    if sorted(i for group in groups for i in group)!=list(range(len(mass))):raise ValueError('Partition of blocks required')
    masses=[];means=[];variances=[]
    for group in groups:
        m=mass[group].sum(0);w=mass[group]/m
        anchor=mean[group[0]];delta=mean[group]-anchor
        mu=np.sum(w[:,:,None]*delta,axis=0)
        vv=np.sum(w[:,:,None]*(variance[group]+(delta-mu)**2),axis=0)
        masses.append(m);means.append(anchor+mu);variances.append(vv)
    return np.asarray(masses),np.asarray(means),np.asarray(variances)


def delta_mcse(mass,mean,variance):
    """Ratio/delta MCSE, including random normalization denominator covariance.

    Blocks must be equal length. Dependence between adjacent half-chain blocks
    can invalidate a16-batch estimate; report the8-chain clustered estimate too.
    """
    mass,mean,variance=combine(mass,mean,variance)
    B=len(mass)
    if B<2:raise ValueError('At least two blocks')
    _,pooled,pv=combine(mass,mean,variance,[list(range(B))]);pooled=pooled[0];pv=pv[0]
    centered=mean-pooled
    mean_influence=B*mass[:,:,None]*centered
    variance_influence=B*mass[:,:,None]*(variance+centered**2-pv)
    # Translation-safe residuals: mean of influences need not round to zero.
    mean_influence-=mean_influence.mean(0)
    variance_influence-=variance_influence.mean(0)
    sd=np.sqrt(pv)
    if np.any((sd==0)&np.any(variance_influence!=0,axis=0)):
        raise FloatingPointError('SD derivative undefined at unresolved zero variance')
    sd_influence=np.divide(variance_influence,2*sd,out=np.zeros_like(variance_influence),where=sd>0)
    return dict(mean=pooled,variance=pv,
        mean_MCSE=np.sqrt(np.sum(mean_influence**2,axis=0)/(B*(B-1))),
        SD_MCSE=np.sqrt(np.sum(sd_influence**2,axis=0)/(B*(B-1))),
        block_weight_ESS=1/np.sum(mass**2,axis=0),maximum_block_mass=mass.max(0),blocks=B)


def assess(mass,mean,variance):
    if len(mass)!=16:raise ValueError('Exactly16 frozen half-chain blocks')
    estimates=[delta_mcse(mass,mean,variance)]
    estimates.append(delta_mcse(*combine(mass,mean,variance,[[2*i,2*i+1] for i in range(8)])))
    sd=np.sqrt(estimates[0]['variance'].mean(1));tol=np.maximum(2e-6,.03*sd)
    mean_mcse=np.maximum.reduce([np.sqrt((e['mean_MCSE']**2).mean(1)) for e in estimates])
    sd_mcse=np.maximum.reduce([np.sqrt((e['SD_MCSE']**2).mean(1)) for e in estimates])
    weight_pass=estimates[0]['block_weight_ESS']>=8
    return dict(mean_RMS_MCSE_m=mean_mcse,SD_RMS_MCSE_m=sd_mcse,tolerance_m=tol,
        predictive_RMS_SD_m=sd,block_weight_ESS=estimates[0]['block_weight_ESS'],
        maximum_block_mass=estimates[0]['maximum_block_mass'],
        mean_precision_passed=mean_mcse<=tol,SD_precision_passed=sd_mcse<=tol,
        block_weight_resolution_passed=weight_pass,
        passed=(mean_mcse<=tol)&(sd_mcse<=tol)&weight_pass,
        sixteen_half_batches=estimates[0],eight_chain_clusters=estimates[1])
