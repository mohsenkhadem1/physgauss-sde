"""Bounded-quadrature-memory FP64 compatible P2 mechanics.

Only constructor temporaries use a shallow one-point geometry proxy; input
geometry is never mutated. Every physical quadrature point is used by the
streamed energy/gradient/full-HVP/node-block operators. No full-Q gradient,
F/cofactor or physical Hessian cache is stored. Constant-material polynomial
stable NH only; no contact or changed force law is introduced.
"""
from copy import copy
import numpy as np
import torch
from experiments.campaign_p2_coarse_preconditioner import TwoLevelP2
from experiments.campaign_p2_torch import material_torch
from experiments.campaign_static_p2_block import EDGES


class StreamingTwoLevelP2(TwoLevelP2):
    def __init__(self,model,device='cpu',maximum_cg_iterations=4000,quadrature_chunk_size=8):
        if type(quadrature_chunk_size) is not int or quadrature_chunk_size<1:raise ValueError('Positive integer quadrature chunk size required')
        bary=np.asarray(model.bary);weights=np.asarray(model.weights);checks=np.asarray(model.check_bary)
        if bary.ndim!=2 or bary.shape[1]!=4 or weights.shape!=(len(bary),) or checks.ndim!=2 or checks.shape[1]!=4 or len(bary)==0 or len(checks)==0 or not all(np.isfinite(x).all() for x in [bary,weights,checks]) or np.any(weights<=0):raise ValueError('Finite positive actual quadrature and checks required')
        proxy=copy(model);proxy.bary=bary[:1];proxy.weights=weights[:1];proxy.check_bary=checks[:1]
        # Parent construction assembles geometry/mass/coarse preconditioner only;
        # it does not evaluate the proxy's one-point physical energy/operator.
        super().__init__(proxy,device,maximum_cg_iterations)
        del self.grad,self.check_grad,self.weights
        tensor=lambda x,dtype=self.dtype:torch.as_tensor(np.asarray(x).copy(),dtype=dtype,device=self.device)
        self.bary=tensor(bary);self.quadrature_weights=tensor(weights);self.check_bary=tensor(checks);self.linear_gradients=tensor(model.linear_gradients)
        self.edge_i=tensor([e[0] for e in EDGES],torch.int64);self.edge_j=tensor([e[1] for e in EDGES],torch.int64);self.quadrature_chunk_size=quadrature_chunk_size

    def shape_gradient_chunk(self,bary):
        g=self.linear_gradients;v=(4*bary[:,None,:,None]-1)*g[None]
        edge=4*(bary[:,self.edge_i][:,None,:,None]*g[None,:,self.edge_j,:]+bary[:,self.edge_j][:,None,:,None]*g[None,:,self.edge_i,:])
        return torch.cat([v,edge],dim=2)

    def chunks(self,checking=False):
        points=self.check_bary if checking else self.bary
        for start in range(0,len(points),self.quadrature_chunk_size):
            stop=min(start+self.quadrature_chunk_size,len(points));grad=self.shape_gradient_chunk(points[start:stop])
            weight=None if checking else self.quadrature_weights[start:stop,None]*self.volumes[None]
            yield grad,weight

    def deformation(self,local_u,grad):return self.identity+torch.einsum('eai,qeaj->qeij',local_u,grad)

    def guard_J(self,J):
        if not bool(torch.isfinite(J).all()) or float(J.min())<=self.minimum_J:raise ValueError('Positive Jacobian guard failed; no repair')
        return J.min()

    def evaluate(self,u):
        if u.shape!=self.rest.shape or u.dtype!=self.dtype or u.device!=self.rest.device or not bool(torch.isfinite(u).all()):raise ValueError('Finite device-matched FP64 nodal displacement required')
        local_u=u[self.elements];energy=torch.zeros((),dtype=self.dtype,device=self.device);gradient=torch.zeros_like(self.rest);minimum=torch.full((),float('inf'),dtype=self.dtype,device=self.device)
        for grad,w in self.chunks():
            F=self.deformation(local_u,grad);W,P,_,_,J=material_torch(F,self.young,self.poisson);minimum=torch.minimum(minimum,self.guard_J(J));energy=energy+(w*W).sum();local=torch.einsum('qe,qeij,qeaj->eai',w,P,grad);gradient=gradient+self.scatter(local)
        for grad,_ in self.chunks(checking=True):minimum=torch.minimum(minimum,self.guard_J(torch.linalg.det(self.deformation(local_u,grad))))
        if not bool(torch.isfinite(energy)) or not bool(torch.isfinite(gradient).all()):raise ValueError('Nonfinite original energy/force; no repair')
        return dict(local_u=local_u,energy=energy,gradient=gradient,minimum_J=minimum)

    def tangent(self,cache,v):
        if v.shape!=self.rest.shape or v.dtype!=self.dtype or v.device!=self.rest.device or not bool(torch.isfinite(v).all()):raise ValueError('Finite device-matched FP64 direction required')
        out=torch.zeros_like(self.rest);local_v=v[self.elements];mu=self.young/(2*(1+self.poisson));lam=self.young*self.poisson/((1+self.poisson)*(1-2*self.poisson));kappa=lam+mu
        for grad,w in self.chunks():
            F=self.deformation(cache['local_u'],grad);_,_,cof,s,_=material_torch(F,self.young,self.poisson);dF=torch.einsum('eai,qeaj->qeij',local_v,grad)
            dc=torch.stack([torch.linalg.cross(dF[..., :,1],F[..., :,2],dim=-1)+torch.linalg.cross(F[..., :,1],dF[..., :,2],dim=-1),torch.linalg.cross(dF[..., :,2],F[..., :,0],dim=-1)+torch.linalg.cross(F[..., :,2],dF[..., :,0],dim=-1),torch.linalg.cross(dF[..., :,0],F[..., :,1],dim=-1)+torch.linalg.cross(F[..., :,0],dF[..., :,1],dim=-1)],-1)
            dP=mu*dF+kappa*(cof*dF).sum((-1,-2))[...,None,None]*cof+s[...,None,None]*dc
            out=out+self.scatter(torch.einsum('qe,qeij,qeaj->eai',w,dP,grad))
        return out

    def nodal_blocks(self,cache,mass_factor=0.):
        mu=self.young/(2*(1+self.poisson));lam=self.young*self.poisson/((1+self.poisson)*(1-2*self.poisson));kappa=lam+mu;local=torch.zeros((len(self.elements),10,3,3),dtype=self.dtype,device=self.device)
        for grad,w in self.chunks():
            _,_,cof,_,_=material_torch(self.deformation(cache['local_u'],grad),self.young,self.poisson);cg=torch.einsum('qeij,qeaj->qeai',cof,grad);iso=(w[...,None]*mu*(grad*grad).sum(-1)).sum(0)
            local=local+kappa*torch.einsum('qe,qeai,qeaj->eaij',w,cg,cg)+iso[...,None,None]*self.identity
        blocks=torch.zeros((len(self.rest),3,3),dtype=self.dtype,device=self.device).index_add_(0,self.elements.reshape(-1),local.reshape(-1,3,3))
        if mass_factor:blocks=blocks+mass_factor*self.mass_diagonal[:,0,None,None]*self.identity
        return blocks

    def diagonal(self,cache):return self.nodal_blocks(cache).diagonal(dim1=-2,dim2=-1)
