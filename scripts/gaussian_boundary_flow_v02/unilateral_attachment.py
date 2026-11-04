"""Isolated conservative Gaussian patch / unilateral tool-plane operator.

The plane, outward unit normal, and material Gaussian weights are prescribed,
not discovered. Positive gap is free space. No tangential force, adhesion,
friction, impact law, or hard nonpenetration is supplied. Existing bilateral FEM
is unchanged; the subclass replaces only explicitly selected attachment groups.
"""
import torch
from scripts.external_benchmark.fem import ExternalFEM


def unilateral_plane(x, weights, stiffness, active, origins, normals):
    """Return energy, energy gradient [N,3], and nodal tangent blocks [N,3,3].

    E = .5 sum_gi active_g*k_g*w_gi*min(n_g.(x_i-q_g),0)^2.
    Force is MINUS the returned gradient. Weights are fixed material quantities,
    so no spatial-gate derivative is missing. At gap=0 choose the inactive
    generalized tangent (zero); the potential is C1 but not C2 there.
    """
    if x.ndim!=2 or x.shape[1]!=3 or weights.ndim!=2 or weights.shape[1]!=len(x):
        raise ValueError('Expected x[N,3] and weights[G,N]')
    groups=len(weights)
    if stiffness.shape!=(groups,) or active.shape!=(groups,) or origins.shape!=(groups,3) or normals.shape!=(groups,3):
        raise ValueError('Expected stiffness/active[G], origins/normals[G,3]')
    if not all(bool(torch.isfinite(t).all()) for t in (x,weights,stiffness,active,origins,normals)):
        raise ValueError('Finite unilateral-plane inputs required')
    if bool((weights<0).any() or (stiffness<0).any() or (active<0).any() or (active>1).any()):
        raise ValueError('Nonnegative weights/stiffness and activation in [0,1] required')
    if not torch.allclose(normals.norm(dim=-1),torch.ones_like(active),rtol=1e-7,atol=1e-7):
        raise ValueError('Prescribed normals must be unit vectors')
    gap=torch.einsum('gni,gi->gn',x[None]-origins[:,None],normals)
    negative=torch.where(gap<0,gap,torch.zeros_like(gap))
    weight=weights*(stiffness*active)[:,None]
    energy=.5*(weight*negative.square()).sum()
    residual=torch.einsum('gn,gi->ni',weight*negative,normals)
    blocks=torch.einsum('gn,gi,gj->nij',weight*(gap<0),normals,normals)
    return energy,residual,blocks


class UnilateralAttachmentFEM(ExternalFEM):
    """Explicitly declared tool planes replace selected bilateral springs.

    plane_origins are SI world positions [T,G,3], plane_normals world unit
    vectors [T,G,3]. Provide complete known command schedules. False entries of
    unilateral_groups retain the original bilateral attachments exactly.
    Planes must come from supplied geometry or a separately frozen prefix-only
    hypothesis; this class never estimates a contact label or tool orientation.
    """
    def __init__(self,data,*args,plane_origins,plane_normals,unilateral_groups,**kwargs):
        if 'evaluation' in data:raise ValueError('Evaluation data forbidden')
        super().__init__(data,*args,**kwargs)
        self.unilateral_groups=torch.as_tensor(unilateral_groups,device=self.device)
        if self.unilateral_groups.dtype!=torch.bool or self.unilateral_groups.shape!=(len(self.weights),):
            raise ValueError('Explicit boolean unilateral group mask required')
        self.plane_origins=self.t((plane_origins-self.origin)/self.scale)
        self.plane_normals=self.t(plane_normals)
        expected=(len(self.targets),len(self.weights),3)
        if self.plane_origins.shape!=expected or self.plane_normals.shape!=expected:
            raise ValueError('Complete prescribed plane schedule [T,G,3] required')
        if not bool(torch.isfinite(self.plane_origins).all() and torch.isfinite(self.plane_normals).all()):
            raise ValueError('Finite plane schedule required')
        if not torch.allclose(self.plane_normals.norm(dim=-1),torch.ones(expected[:2],device=self.device,dtype=self.dtype),atol=1e-7,rtol=1e-7):
            raise ValueError('Unit world normals required')

    def evaluate(self,x,theta,frame,tangent=False):
        energy,residual,H=super().evaluate(x,theta,frame,tangent)
        select=self.unilateral_groups
        if not bool(select.any()):return energy,residual,H
        weights=self.weights[select];stiffness=torch.exp(theta[1:1+len(self.weights)])[select]
        active=self.active[frame,select];w=weights*(stiffness*active)[:,None]
        delta=x[None]-self.targets[frame,select]
        old_energy=.5*(w[:,:,None]*delta.square()).sum()
        old_residual=(w[:,:,None]*delta).sum(0)
        new_energy,new_residual,blocks=unilateral_plane(x,weights,stiffness,active,
                              self.plane_origins[frame,select],self.plane_normals[frame,select])
        energy=energy-old_energy+new_energy;residual=residual-old_residual+new_residual
        if tangent:
            blocks=blocks-w.sum(0)[:,None,None]*self.eye
            ids=torch.arange(self.ndof,device=self.device).reshape(-1,3)
            H=H.index_put((ids[:,:,None].expand_as(blocks).reshape(-1),ids[:,None,:].expand_as(blocks).reshape(-1)),blocks.reshape(-1),accumulate=True)
        return energy,residual,H
