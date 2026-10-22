"""Exact observation-only evaluation of immutable manufactured dynamics.

Stores only frames12/24 and selected nodes1/4, stopping at24. Future controls
cannot affect acquired data under the same causal held-force implicitEuler.
Chunks are fixed1024 for memory/cache boundedness, not an approximation.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.special import logsumexp
from experiments import campaign_compliant_continuous_reference as reference


class ObservationOnlyForward(reference.ExactVectorForward):
    chunk=1024
    def observations(self,theta):
        theta=np.asarray(theta,dtype=float)
        if theta.ndim!=2 or theta.shape[1]!=4 or not len(theta) or not np.isfinite(theta).all() or (theta[:,[0,2,3]]<=0).any():
            raise ValueError('Finite physical theta required')
        result=[]
        for first in range(0,len(theta),self.chunk):
            values=theta[first:first+self.chunk];count=len(values)
            centers=np.column_stack([values[:,1],np.full(count,.02),np.zeros(count)])
            logw=np.log(self.areas)[None]-.5*np.sum((self.X[None]-centers[:,None])**2,2)/values[:,2,None]**2
            weights=np.exp(logw-logsumexp(logw,axis=1)[:,None]);d=torch.tensor(np.repeat(weights,3,axis=1))
            K=torch.tensor(values[:,0,None,None])*self.Kunit+torch.diag_embed(d*torch.tensor(values[:,3,None]));C=.05*d
            h=reference.original.H;m=self.mass
            chol=torch.linalg.cholesky(K+torch.diag_embed(m[None]/h**2+C/h))
            u=torch.zeros((count,15),dtype=torch.float64);v=u.clone();selected=[]
            for step,F in enumerate(self.forces['A'][:24],1):
                new=torch.cholesky_solve((F[None]+m*(u/h**2+v/h)+C*u/h)[:,:,None],chol)[:,:,0]
                v=(new-u)/h;u=new
                if step in (12,24):selected.append(u.reshape(count,5,3)[:,reference.original.OBS_NODES].clone())
            result.append(torch.stack(selected,1).reshape(count,12).numpy())
        return np.concatenate(result)
    def log_likelihood(self,unit,dimensions,observation):
        pred=self.observations(reference.physical(unit,dimensions))
        sigma=reference.original.SIGMA
        return -.5*np.sum(((pred-observation)/sigma)**2,axis=1)-len(observation)*np.log(sigma*np.sqrt(2*np.pi))
