"""Exact position-only state moves with Gaussian velocity refreshes.

For constant internal amplitude, covariance/contact depend on positions but
not velocities. At fixed positions, all velocities and Gaussian load variables
are jointly Gaussian. QR eliminates them without a subtractive Schur precision.
The original full-dimensional physical posterior is restored by a conditional
Gaussian draw after the reversible marginal position move. No prior changes.
"""
import math
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pacnerf_hybrid import ConditionalStateEvaluator
from pacnerf_reflections import state_transition


def indices(model):
    m=model.modes
    positions=torch.tensor([2+m+t*2*m+j for t in range(13) for j in range(m)])
    velocities=torch.tensor([2+j for j in range(m)]+
        [2+m+t*2*m+m+j for t in range(13) for j in range(m)])
    return positions,velocities


class GaussianVelocities:
    def __init__(self,model,z):
        if model.noise!='constant':
            raise ValueError('Exact velocity integration requires velocity-independent covariance')
        self.model=model;self.q_index,self.v_index=indices(model)
        b=len(z);m=model.modes;k=model.force_dimension;nv=len(self.v_index)
        zero=z.clone();zero[:,self.v_index]=0
        if model.force=='joint':zero[:,model.base_dimension:model.core_dimension]=0
        residual,covariance,mapping,_=model.terms(zero)
        transition=model.transition(zero,model.dt)
        velocity_map=z.new_zeros(b,13,2*m,nv)
        for t in range(13):
            velocity_map[:,t,:,t*m:(t+1)*m]=-transition[:,:,m:]*.5
            velocity_map[:,t,m:,(t+1)*m:(t+2)*m]+=.5*torch.eye(m,dtype=z.dtype)
        chol=torch.linalg.cholesky(covariance)
        wb=torch.linalg.solve_triangular(chol,velocity_map,upper=False).flatten(1,2)
        wg=torch.linalg.solve_triangular(chol,mapping,upper=False).flatten(1,2)
        wr=torch.linalg.solve_triangular(chol,residual[...,None],upper=False).squeeze(-1).flatten(1)
        # Unknown ordering [all force coefficients, all normalized velocities].
        matrix=torch.cat((-wg,wb),-1)
        force_prior=z.new_zeros(b,k,k+nv);force_prior[:,:,:k]=torch.eye(k,dtype=z.dtype)
        initial_prior=z.new_zeros(b,m,k+nv);initial_prior[:,:,k:k+m]=torch.eye(m,dtype=z.dtype)
        matrix=torch.cat((matrix,force_prior,initial_prior),1)
        rhs=torch.cat((-wr,z.new_zeros(b,k+m)),1)
        q,root=torch.linalg.qr(matrix,mode='reduced')
        projected=(q.transpose(-1,-2)@rhs[...,None])
        mean=torch.linalg.solve_triangular(root,projected,upper=True).squeeze(-1)
        error=(matrix@mean[...,None]).squeeze(-1)-rhs
        logdet_q=2*chol.diagonal(dim1=-2,dim2=-1).log().sum((-1,-2))
        logdet_bundle=2*root.diagonal(dim1=-2,dim2=-1).abs().log().sum(-1)
        prior=.5*zero[:,:2].square().sum(-1)+.5*zero[:,-3:].square().sum(-1)
        self.marginal_nll=(.5*(error.square().sum(-1)+logdet_q+logdet_bundle+
            (residual.shape[1]*residual.shape[2]-nv)*math.log(2*math.pi))+
            prior-model.scale[2+m:].log().sum())
        self.mean=mean;self.root=root;self.force_dimension=k;self.velocity_dimension=nv
        self.velocity_map=velocity_map;self.zero=zero

    def conditional_nll(self,z):
        k=self.force_dimension
        if self.model.force=='joint':
            value=torch.cat((z[:,self.model.base_dimension:self.model.core_dimension],z[:,self.v_index]),-1)
            root=self.root;mean=self.mean
        else:
            value=z[:,self.v_index];root=self.root[:,k:,k:];mean=self.mean[:,k:]
        white=(root@(value-mean)[...,None]).squeeze(-1)
        return .5*(white.square().sum(-1)+value.shape[-1]*math.log(2*math.pi))-root.diagonal(dim1=-2,dim2=-1).abs().log().sum(-1)

    def draw(self,rng):
        white=torch.randn(self.mean.shape,dtype=self.mean.dtype,generator=rng)
        value=self.mean+torch.linalg.solve_triangular(self.root,white[...,None],upper=True).squeeze(-1)
        full=self.zero.clone();full[:,self.v_index]=value[:,self.force_dimension:]
        if self.model.force=='joint':full[:,self.model.base_dimension:self.model.core_dimension]=value[:,:self.force_dimension]
        return full


class CollapsedStateContext:
    def __init__(self,target,fixed,core):
        self.target=target;self.fixed=fixed.clone();self.core=core
        self.model=target.model.model
        if self.model.noise!='constant':raise ValueError('Constant-noise collapsed move only')
        self.q_index,self.v_index=indices(self.model)
        self.parameters=ConditionalStateEvaluator(target,fixed,core).context
        with torch.no_grad():
            root=self.parameters['gaussian'][1]
            mapping=torch.linalg.solve_triangular(root.transpose(-1,-2),
                target.transform[2:core,2:core].expand(len(fixed),-1,-1),upper=True)
            positions=mapping[:,self.q_index-2]
            covariance=positions@positions.transpose(-1,-2)
            self.position_root=torch.linalg.cholesky((covariance+covariance.transpose(-1,-2))/2)
            zero=fixed.clone();zero[:,2:core]=0
            self.template=target.decode(fixed)
            self.offset=target.decode(zero)[:,self.q_index]
            self.initial=torch.linalg.solve_triangular(self.position_root,
                (self.template[:,self.q_index]-self.offset)[...,None],upper=False).squeeze(-1)

    def physical(self,position):
        full=self.template.clone()
        full[:,self.q_index]=self.offset+(self.position_root@position[...,None]).squeeze(-1)
        return full

    @torch.no_grad()
    def geometry(self):
        model=self.model;m=model.modes
        mapping=(self.position_root*model.scale[self.q_index][None,:,None]).reshape(len(self.fixed),13,m,-1)
        normals=torch.einsum('gm,btms->btgs',model.floor_basis,mapping).flatten(1,2)
        offset=model.floor_heights(self.physical(torch.zeros_like(self.initial))).flatten(1)
        return normals,offset

    @torch.enable_grad()
    def __call__(self,position):
        variable=position.detach().requires_grad_()
        try:
            if self.model._conditional_context is not None:raise ValueError('Nested conditional cache')
            self.model._conditional_context=self.parameters
            full=self.physical(variable);gaussian=GaussianVelocities(self.model,full)
            loss=gaussian.marginal_nll+.5*self.model.image_residual(full).square().sum(-1)+self.model.image_normalizer(full)
            value=-loss+.5*variable.square().sum(-1)
            gradient,=torch.autograd.grad(value.sum(),variable)
        finally:self.model._conditional_context=None
        bad=~torch.isfinite(value)|~torch.isfinite(gradient).all(-1)
        return value.detach(),torch.nan_to_num(gradient.detach()),bad


@torch.no_grad()
def collapsed_transition(target,white,core,epsilon,steps,rng):
    context=CollapsedStateContext(target,white,core)
    normals,offset=context.geometry()
    position,info=state_transition(context.initial,context,normals,offset,.002,epsilon,steps,rng)
    try:
        if context.model._conditional_context is not None:raise ValueError('Nested conditional draw')
        context.model._conditional_context=context.parameters
        bundle=GaussianVelocities(context.model,context.physical(position))
        physical=bundle.draw(rng)
    finally:context.model._conditional_context=None
    native,_=target.model.transform(physical,inverse=True)
    result=torch.linalg.solve(target.transform,(native-target.mean).T).T
    return result,info
