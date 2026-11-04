"""Finite-range Gaussian cohesive attachment, not a Coulomb contact model.

U=k sigma^2 (1-exp(-||x-target||^2/(2 sigma^2))). Its restoring force is
k exp(-r^2/(2sigma^2))*(target-x): linear near zero and vanishing far away.
Gravity can overcome its finite peak force, allowing dynamic release without
an inverse fit or a manually chosen release frame. Surface/material patch weights
remain a separate spatial prior. This operator is deliberately nonconvex.
"""
import torch
from scripts.external_benchmark.dynamic_fem import DynamicFEM


def gaussian_bond(delta,weight,sigma):
    if bool((sigma<=0).any()) or not bool(torch.isfinite(sigma).all()):raise ValueError('Positive finite interaction ranges required')
    s2=sigma.square();e=torch.exp(-.5*delta.square().sum(-1)/s2)
    energy=(weight*s2*(-torch.expm1(-.5*delta.square().sum(-1)/s2))).sum()
    residual=weight[...,None]*e[...,None]*delta
    I=torch.eye(3,dtype=delta.dtype,device=delta.device)
    H=(weight*e)[...,None,None]*(I-delta[..., :,None]*delta[...,None,:]/s2[...,None,None])
    return energy,residual,H


class FiniteRangeAttachmentFEM(DynamicFEM):
    def __init__(self,*args,interaction_range=.10,**kwargs):
        super().__init__(*args,**kwargs)
        self.interaction_range=self.t(interaction_range)
        if self.interaction_range.ndim==0:self.interaction_range=self.interaction_range.repeat(len(self.weights))
        if self.interaction_range.shape!=(len(self.weights),) or bool((self.interaction_range<=0).any()):raise ValueError('One positive dimensionless range per attachment required')

    def evaluate(self,x,theta,frame,tangent=False):
        energy,r,H=super().evaluate(x,theta,frame,tangent)
        delta=x[None]-self.targets[frame]
        weight=self.weights*theta[1:1+len(self.weights)].exp()[:,None]*self.active[frame,:,None]
        U,newr,blocks=gaussian_bond(delta,weight,self.interaction_range[:,None])
        energy=energy+U-.5*(weight[...,None]*delta.square()).sum()
        r=r+(newr-weight[...,None]*delta).sum(0)
        if tangent:
            blocks=blocks.sum(0)-weight.sum(0)[:,None,None]*self.eye
            ids=torch.arange(self.ndof,device=self.device).reshape(-1,3)
            H=H.index_put((ids[:,:,None].expand_as(blocks).reshape(-1),ids[:,None,:].expand_as(blocks).reshape(-1)),blocks.reshape(-1),accumulate=True)
        return energy,r,H
