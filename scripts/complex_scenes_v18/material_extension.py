"""Optional independent compressibility for the frozen shared material decoder.

The old descriptor remains an exact submodel when the fourth coordinate is zero.
No old weight or checkpoint is changed. This adds an identifiable physical degree
of freedom to test on development, not a claim that inference has improved.
"""
import torch
from torch import nn
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw


class IndependentBulkScale(nn.Module):
    def __init__(self,base=None):
        super().__init__();self.base=base if base is not None else ScaleSeparatedMaterialLaw();self.latent_dim=4

    def energy(self,F,z):
        if z.shape[-1]!=4:raise ValueError('Expected old 3D descriptor plus log bulk multiplier')
        old=self.base.energy(F,z[...,:3]);J=torch.linalg.det(F)
        volume=J-torch.log(J)-1
        coefficient=self.base.base.coefficients(z[...,1:3])[...,2]
        original_bulk=self.base.base.energy_scale*torch.exp(z[...,0])*coefficient*volume
        return old+torch.expm1(z[...,3])*original_bulk

    forward=energy
