"""Remove the static numerical gauge once inertia fixes rigid-body nullspaces.

The frozen model uses a tiny rest-position spring to fix free static modes.
Keeping that spring in release dynamics introduces an artificial restoring force,
especially when the inferred mass is small. The physical energy terms remain
unchanged; the inertial term already makes dynamic translations well posed.
"""
import torch
from scripts.external_benchmark.fem import ExternalFEM

class DynamicFEM(ExternalFEM):
    def evaluate(self,x,theta,frame,tangent=False):
        energy,residual,hessian=super().evaluate(x,theta,frame,tangent)
        if self.inertial_target is not None:
            pin=1e-5;delta=x-self.rest
            energy=energy-.5*pin*(delta**2).sum()
            residual=residual-pin*delta
            if hessian is not None:hessian=hessian-pin*torch.eye(self.ndof,device=self.device,dtype=self.dtype)
        return energy,residual,hessian
