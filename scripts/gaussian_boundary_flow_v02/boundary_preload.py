"""Localized attachment preloads: external boundary reactions, not body cancellation.

For fixed MATERIAL attachment weights and a prescribed vector b_g, each active
kernel contributes nodal force w_gi/sum(w_g) * b_g. The potential is linear in
position, -sum_i f_i.(x_i-X_i), so its residual is -f and x-tangent is zero.
Vectors use the same nondimensional force/material units as ExternalFEM; they
are not measured Newtons unless a separate physical energy/length scale is known.
"""
import torch
from scripts.external_benchmark.fem import ExternalFEM
from scripts.gaussian_boundary_flow_v02.object_prestrain import InitialPrestrainFEM


def attachment_preload_force(weights,active,vectors,*,rotations=None):
    """Return [N,3] forces from [G,N] weights, [G] activation and [G,3] vectors.

    Optional prescribed rotations [G,3,3] map task-frame column vectors to world.
    These rotations may depend on command time, NEVER on the unknown current x;
    a state-dependent follower force would need a different potential/tangent.
    """
    if weights.ndim!=2 or active.shape!=(len(weights),) or vectors.shape!=(len(weights),3):
        raise ValueError('Expected weights [G,N], active [G], vectors [G,3]')
    if not bool(torch.isfinite(weights).all() and torch.isfinite(active).all() and torch.isfinite(vectors).all()):
        raise ValueError('Finite preload inputs required')
    if bool((weights<0).any()) or bool(((active<0)|(active>1)).any()):
        raise ValueError('Nonnegative weights and activation in [0,1] required')
    totals=weights.sum(-1)
    if bool(((totals<=0)&(active>0)).any()):
        raise ValueError('Active preload kernel requires nonzero spatial support')
    normalized=weights/totals.clamp_min(torch.finfo(weights.dtype).tiny)[:,None]
    if rotations is not None:
        if rotations.shape!=(len(weights),3,3):raise ValueError('Expected task rotations [G,3,3]')
        vectors=torch.einsum('gij,gj->gi',rotations,vectors)
    return torch.einsum('gn,g,gj->nj',normalized,active,vectors)


class AttachmentPreloadMixin:
    """Append 3*G actual preload components to a compatible FEM parameter vector.

    full_parameter_count includes these new components; base_parameter_count
    leaves an inherited prestrain model's own parameter_count untouched.
    The default added vector is zero. A caller must declare an inference prior
    and any bounds; this class does not claim the initial reaction is identified.
    """
    def __init__(self,*args,preload_frame='world',preload_rotations=None,**kwargs):
        super().__init__(*args,**kwargs)
        if preload_frame not in ('world','task'):
            raise ValueError('Declare preload_frame as world or task')
        self.preload_frame=preload_frame
        self.base_parameter_count=getattr(self,'parameter_count',self.i_field+1+int(self.contact))
        self.i_preload=self.base_parameter_count
        self.full_parameter_count=self.base_parameter_count+3*len(self.weights)
        if preload_frame=='world':
            if preload_rotations is not None:raise ValueError('World vectors do not take task rotations')
            self.preload_rotations=None
        else:
            if preload_rotations is None:raise ValueError('Task vectors require prescribed rotations')
            R=self.t(preload_rotations);expected=(len(self.weights),3,3)
            if R.shape!=expected and R.shape!=(len(self.targets),*expected):
                raise ValueError('Task rotations must be [G,3,3] or [T,G,3,3]')
            if not bool(torch.isfinite(R).all()) or not torch.allclose(R@R.transpose(-1,-2),self.eye.expand_as(R),atol=1e-8,rtol=1e-8) or not bool((torch.linalg.det(R)>0).all()):
                raise ValueError('Proper prescribed task rotations required')
            self.preload_rotations=R

    def preload_force(self,theta,frame):
        if theta.ndim!=1 or len(theta)!=self.full_parameter_count:
            raise ValueError('Append one three-component preload per attachment')
        vectors=theta[self.i_preload:].reshape(len(self.weights),3)
        R=self.preload_rotations
        if R is not None and R.ndim==4:R=R[frame]
        return attachment_preload_force(self.weights,self.active[frame],vectors,rotations=R)

    def evaluate(self,x,theta,frame,tangent=False):
        force=self.preload_force(theta,frame)
        energy,residual,H=super().evaluate(x,theta[:self.base_parameter_count],frame,tangent)
        energy=energy-(force*(x-self.rest)).sum()
        residual=residual-force
        return energy,residual,H


class PreloadedExternalFEM(AttachmentPreloadMixin,ExternalFEM):
    """Original nonlinear mechanics plus localized initial attachment reaction."""


class InitialPrestrainPreloadFEM(AttachmentPreloadMixin,InitialPrestrainFEM):
    """Objective bulk natural strain composed with localized boundary preloads."""
