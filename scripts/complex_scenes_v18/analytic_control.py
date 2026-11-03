"""Matched-family analytic Neo-Hookean control with unknown spatial parameters.

This is an oracle constitutive-family diagnostic, not a learned visual baseline.
The reference generator uses this family; that advantage must be disclosed.
"""
import torch


class AnalyticControl:
    def energy(self,F,z):
        J=torch.linalg.det(F);logJ=torch.log(J);I1=(F*F).sum((-1,-2))
        # Initial E=10 kPa, nu=.30; scale and lambda multiplier are inferred.
        E=10000*torch.exp(z[...,0]);mu=E/2.6;lam=E*.3/(1.3*.4)*torch.exp(z[...,3])
        return .5*mu*(I1-3-2*logJ)+.5*lam*logJ.square()
