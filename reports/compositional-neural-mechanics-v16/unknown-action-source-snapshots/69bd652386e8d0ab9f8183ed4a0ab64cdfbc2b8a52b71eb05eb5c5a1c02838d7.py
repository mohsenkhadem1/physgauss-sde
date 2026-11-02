"""Objective convex bilateral interface energy with shared learned curvature.

This is a compliant attachment, NOT unilateral contact/friction. Surface weights
must integrate to one, so K has units N/m and energy has units J. delta is the
current material-point position minus the target point on a rigid controller.
Latent coordinates: [log(K_N_per_m), nonlinear_strength, family_descriptor].
The last two coordinates are in [0,1]; unknown scene descriptors require inference.
"""
import torch
from torch import nn
from torch.nn import functional as F


def invariants(delta,z,reference_length):
    if delta.shape[-1]!=3 or z.shape[-1]!=3:raise ValueError('Three delta and latent coordinates required')
    if not bool(torch.isfinite(delta).all()) or not bool(torch.isfinite(z).all()):raise ValueError('Finite interface inputs required')
    if bool(((z[...,1:]<0)|(z[...,1:]>1)).any()):raise ValueError('Shape/family descriptors must be in [0,1]')
    if reference_length<=0:raise ValueError('Positive reference length required')
    return (delta/reference_length).square().sum(-1),z[...,0].exp(),z[...,1],z[...,2]


class AnalyticInterfaceLaw(nn.Module):
    def __init__(self,reference_length=.002):
        super().__init__();self.reference_length=float(reference_length)
    def energy_density(self,delta,z):
        s,K,strength,family=invariants(delta,z,self.reference_length)
        # Stable expansion of s-log(1+s) near the reference state.
        small=s.square()/2-s.pow(3)/3+s.pow(4)/4-s.pow(5)/5+s.pow(6)/6
        saturated=torch.where(s<1e-3,small,s-torch.log1p(s))
        curvature=(1-family)*s.square()+family*saturated
        return .5*K*self.reference_length**2*(s+2*strength*curvature)
    energy=energy_density
    def traction(self,delta,z,create_graph=True):
        x=delta if delta.requires_grad else delta.detach().requires_grad_(True)
        return torch.autograd.grad(self.energy_density(x,z).sum(),x,create_graph=create_graph)[0]


class SharedInterfaceLaw(AnalyticInterfaceLaw):
    def __init__(self,reference_length=.002,bases=6,width=16):
        super().__init__(reference_length)
        self.raw_slopes=nn.Parameter(torch.linspace(-2.,1.,bases,dtype=torch.float64))
        self.offsets=nn.Parameter(torch.linspace(-3.,2.,bases,dtype=torch.float64))
        self.conditioner=nn.Sequential(nn.Linear(2,width),nn.Tanh(),nn.Linear(width,bases+1)).double()
    def energy_density(self,delta,z):
        s,K,strength,_=invariants(delta,z,self.reference_length)
        a=F.softplus(self.raw_slopes)+1e-6;b=self.offsets
        x=s[...,None]*a;prob=torch.sigmoid(b)
        exact=F.softplus(b+x)-F.softplus(b)-prob*x
        small=.5*prob*(1-prob)*x.square()+prob*(1-prob)*(1-2*prob)*x.pow(3)/6
        phi=torch.where(x.abs()<1e-4,small,exact)
        basis=torch.cat([s.square()[...,None],phi],dim=-1)
        coefficients=F.softplus(self.conditioner(z[...,1:]))
        nonlinear=(coefficients*basis).sum(-1)
        return .5*K*self.reference_length**2*(s+2*strength*nonlinear)
    energy=energy_density


def interface_derivatives(law,delta,z,tangent=False,create_graph=False):
    x=delta if delta.requires_grad else delta.detach().requires_grad_(True)
    W=law.energy_density(x,z);traction=torch.autograd.grad(W.sum(),x,create_graph=True)[0]
    H=None
    if tangent:
        H=torch.stack([torch.autograd.grad(traction[...,i].sum(),x,create_graph=create_graph,retain_graph=True)[0]
                       for i in range(3)],dim=-2)
    return W,traction,H
