"""Particle-count-only refresh continuation and saved-draw function replay.

No new MH draws, observations, material parameters, proposal adaptation or
policy choices. The old18function moments were not archived: reconstruct those
alone from saved MH states while reusing135prediction moments/MCSE verbatim.
"""
import numpy as np
import torch
from scipy.special import logsumexp
from experiments import campaign_postpoke_refresh_v02 as model


def replay_function_moments(forward,unit,*,chunk=1024):
    """Same saved physical states, only originalA/B final nodes+fixture fields."""
    from experiments.campaign_force_prediction_precision_v01 import combine
    unit=np.asarray(unit,float)
    if unit.shape[-1]!=4 or unit.ndim not in(2,3):raise ValueError('Saved full4D states required')
    flat=unit.reshape(-1,4);counts=[];means=[];variances=[];h=model.cohort.old.original.H;m=forward.mass
    held=torch.tensor(np.stack([model.cohort.old.original.known_force(k)for k in('A','B')]).reshape(2,60,15).transpose(1,2,0))
    for first in range(0,len(flat),chunk):
        values=model.cohort.physical(flat[first:first+chunk]);n=len(values);center=np.column_stack([values[:,1],np.full(n,.02),np.zeros(n)])
        logw=np.log(forward.areas)[None]-.5*np.sum((forward.X[None]-center[:,None])**2,2)/values[:,2,None]**2
        weights=np.exp(logw-logsumexp(logw,axis=1)[:,None]);diagonal=torch.tensor(np.repeat(weights,3,axis=1));C=.05*diagonal
        K=torch.tensor(values[:,0,None,None])*forward.Kunit+torch.diag_embed(diagonal*torch.tensor(values[:,3,None]));chol=torch.linalg.cholesky(K+torch.diag_embed(m[None]/h**2+C/h))
        u=torch.zeros((n,15,2),dtype=torch.float64);v=u.clone()
        for step in range(60):
            rhs=held[step][None]+m[None,:,None]*(u/h**2+v/h)+C[:,:,None]*u/h
            new=torch.cholesky_solve(rhs,chol);v=(new-u)/h;u=new
        end=u.numpy().reshape(n,5,3,2)
        f=np.concatenate([model.cohort.ridge.fixture_functions(values,forward.areas,forward.X),end[:,1,:,0],end[:,4,:,1]],axis=1)
        mu=f.mean(0);counts.append(n);means.append(mu);variances.append(((f-mu)**2).mean(0))
    mass=np.asarray(counts,float)[:,None]/len(flat);_,mu,var=combine(mass,np.asarray(means)[:,None],np.asarray(variances)[:,None],[list(range(len(counts)))])
    return dict(function_mean=mu[0,0],function_std=np.sqrt(var[0,0]),saved_draws=len(flat),replay_only=True,new_MH_draws=0)
