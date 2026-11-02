"""Additive exact stable-NH energy with nonsingular higher autodiff at F=I.

Same energy as frozen materials.StableNeoHookean. The 3x3 determinant is evaluated
as its explicit polynomial, avoiding an SVD/LU derivative of singular G=F-I.
No existing trajectories or frozen operators are replaced by this class.
"""
import torch
from scripts.compositional_dynamics_v17.materials import StableNeoHookean

class StableNeoHookeanPolynomial(StableNeoHookean):
    def __call__(self,F,z):
        I=torch.eye(3,dtype=F.dtype,device=F.device);G=F-I;tr=G.diagonal(dim1=-2,dim2=-1).sum(-1)
        second=.5*(tr.square()-torch.einsum('...ij,...ji->...',G,G))
        detG=(G[..., :,0]*torch.linalg.cross(G[..., :,1],G[..., :,2],dim=-1)).sum(-1)
        jminus=tr+second+detG
        return self.mu*(.5*G.square().sum((-2,-1))-second-detG)+.5*self.kappa*jminus.square()
