"""Broader response-ratio prior, grounded in archived external-FEM fit bounds.

The original object pilot was narrower than existing v22 fits. This revision
exposes that restriction rather than attributing its mismatch to posterior family.
"""
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.object_backend_table import GaussianTableBoundaryBackend


class WideGaussianTableBoundaryBackend(GaussianTableBoundaryBackend):
    def __init__(self,payload,device='cpu'):
        super().__init__(payload,device)
        self.lower=np.array([-9.]+[-3.]*self.groups+[0.]+[-2.5]*self.groups+[-8.])
        self.upper=np.array([1.]+[8.]*self.groups+[9.]+[2.5]*self.groups+[8.])

    def bind(self,logits):
        z=self.model.t(logits)
        if z.shape!=(self.parameter_dim,) or not bool(torch.isfinite(z).all()):raise ValueError('Finite latent vector required')
        p=self.model.t(self.lower)+(self.model.t(self.upper)-self.model.t(self.lower))*torch.sigmoid(z)
        raw=self.base_weights.clamp_min(1e-30).pow(torch.exp(p[2+self.groups:-1])[:,None])
        self.model.weights=raw/raw.sum(-1,keepdim=True)
        theta=torch.zeros(self.model.i_field+2,dtype=self.model.dtype,device=self.model.device)
        theta[:2+self.groups]=p[:2+self.groups];theta[self.model.i_field+1]=p[-1]
        return theta
