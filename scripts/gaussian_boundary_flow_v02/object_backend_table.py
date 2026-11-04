"""One additional Gaussian boundary operator: compliant tangential table support.

This is the existing conservative localized support potential, NOT Coulomb
friction. Its stiffness is inferred relative to bulk response, and fades when
material leaves the table neighborhood. No future object/contact labels enter.
"""
import numpy as np
import torch
from scripts.external_benchmark.fem import ExternalFEM
from scripts.gaussian_boundary_flow_v02.object_backend_revision import ObjectBoundaryBackendRevision


class GaussianTableBoundaryBackend(ObjectBoundaryBackendRevision):
    def __init__(self,payload,device='cpu'):
        super().__init__(payload,device,floor_mode='inferred')
        self.model=ExternalFEM(payload,device=device,field=False,contact=True,nonconvex=False,materials=1)
        self.base_weights=self.model.weights.clone()
        self.names=self.names+['log_tangential_table_material_ratio']
        self.lower=np.r_[self.lower,-8.];self.upper=np.r_[self.upper,4.]
        self.parameter_dim+=1

    def bind(self,logits):
        z=self.model.t(logits)
        if z.shape!=(self.parameter_dim,) or not bool(torch.isfinite(z).all()):raise ValueError('Finite latent vector required')
        # Parent binding uses its fixed original index map; retain standalone size.
        p=self.model.t(self._full_lower)+(self.model.t(self._full_upper)-self.model.t(self._full_lower))*torch.sigmoid(z[:-1])
        raw=self.base_weights.clamp_min(1e-30).pow(torch.exp(p[2+self.groups:])[:,None])
        self.model.weights=raw/raw.sum(-1,keepdim=True)
        theta=torch.zeros(self.model.i_field+2,dtype=self.model.dtype,device=self.model.device)
        theta[:2+self.groups]=p[:2+self.groups]
        theta[self.model.i_field+1]=-8+12*torch.sigmoid(z[-1])
        return theta
