"""Exact stable Neo-Hookean common-law control; distinct from learned MR/Yeoh."""
import torch

class StableNeoHookean:
    def __init__(self,young=12000.,poisson=.35):
        if young<=0 or not -1<poisson<.5:raise ValueError('Valid elastic constants required')
        self.young=young;self.poisson=poisson;self.mu=young/(2*(1+poisson));self.lam=young*poisson/((1+poisson)*(1-2*poisson));self.kappa=self.lam+self.mu
    def __call__(self,F,z):
        # Reference-normalized algebra identical to installed Newton stable NH.
        I=torch.eye(3,dtype=F.dtype,device=F.device);G=F-I;tr=G.diagonal(dim1=-2,dim2=-1).sum(-1)
        second=.5*(tr.square()-torch.einsum('...ij,...ji->...',G,G));detG=torch.linalg.det(G);jminus=tr+second+detG
        return self.mu*(.5*G.square().sum((-2,-1))-second-detG)+.5*self.kappa*jminus.square()
