"""Exact common-scale symmetry for the shared isotropic bulk energy.

W(F,[log s,shape,family]) = exp(log s) W_base(F,[shape,family]).
Common scale never enters the neural decoder. This preserves the zero-load,
fully co-scaled attachment equilibrium symmetry; gravity/inertia/unscaled loads
break that scene-level symmetry. No extra stability/polyconvexity claim is made.
"""
import torch
from torch import nn
from scripts.compositional_neural_mechanics.materials import SharedMaterialLaw

class ScaleSeparatedMaterialLaw(nn.Module):
    def __init__(self,hidden=16,energy_scale=12000.):
        super().__init__();self.latent_dim=3
        self.base=SharedMaterialLaw(latent_dim=2,hidden=hidden,energy_scale=energy_scale)
    def energy_density(self,F,z):
        if z.shape[-1]!=3:raise ValueError('Expected log common scale,shape,family descriptors')
        return torch.exp(z[...,0])*self.base.energy_density(F,z[...,1:])
    energy=energy_density
    forward=energy_density
    def stress(self,F,z,create_graph=True):
        with torch.enable_grad():
            if not F.requires_grad:F=F.detach().clone().requires_grad_(True)
            return torch.autograd.grad(self.energy_density(F,z).sum(),F,create_graph=create_graph)[0]
