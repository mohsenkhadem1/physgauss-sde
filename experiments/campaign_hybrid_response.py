"""Causal force-linear Gaussian response prior given a physical hypothesis.

The spatial coefficients depend only on reference geometry, material, boundary
hypothesis and full rigid hardware. Known control history enters through fixed
linear convolutions, never through an unconstrained recurrent/nonlinear head.
Superposition is exact for nodal displacement and handle rotation vectors;
the finite SO(3) attachment decoder is intentionally nonlinear. This is a
learned conditional forward approximation, not an integrator or posterior.
"""
from __future__ import annotations

import torch
from torch import nn

try:
    from campaign_hybrid_predictor import adjacency,prepare_features,skew
except ImportError:
    from experiments.campaign_hybrid_predictor import adjacency,prepare_features,skew


def causal_history_basis(controls,support_offsets,times,*,dt=.05,
                         frequencies=(0.,2.,6.,12.),decays=(0.,.5,2.)):
    """Return [B,T,15*K] features on the fixed 20 Hz public query clock.

    Row j>0 is the known drive for the preceding interval (t[j-1],t[j]].
    Row zero has no preceding interval and is ignored. Controls contain the
    twelve literal mean/centered-moment channels; their units and quadrature
    interpretation remain those of the dataset. Support displacement is three
    further linear drive channels, with exact fixture positions set separately.
    Filters use interval-midpoint lag. Impulse and double integral are exact for
    an interval-constant drive; oscillatory filters use midpoint quadrature.
    No learned or target-dependent temporal basis is used.
    """
    if controls.ndim!=3 or controls.shape[-1]!=12 or support_offsets.shape!=(*controls.shape[:2],3):
        raise ValueError('Expected twelve control and three support channels')
    T=controls.shape[1]
    if times.ndim!=1 or len(times)!=T or T<1:
        raise ValueError('One nonempty common public query clock required')
    expected=torch.arange(T,dtype=times.dtype,device=times.device)*dt
    if not bool(torch.allclose(times,expected,atol=2e-6,rtol=0)):
        raise ValueError('Response basis requires the fixed .05 s clock starting at zero')
    if bool((support_offsets[:,0]!=0).any()):
        raise ValueError('Fresh rest hypothesis requires zero initial support offset')
    drive=torch.cat([controls/.1,support_offsets/.01],-1)
    drive=torch.cat([torch.zeros_like(drive[:,:1]),drive[:,1:]],1)
    i=torch.arange(T,device=times.device)
    admissible=(i[:,None]>=i[None,:])&(i[None,:]>0)
    lag=(i[:,None]-i[None,:]).to(times.dtype)*dt+dt/2
    kernels=[torch.ones_like(lag)*dt,lag*dt]
    for decay in decays:
        for frequency in frequencies:
            response=lag if frequency==0 else torch.sin(frequency*lag)/frequency
            kernels.append(dt*torch.exp(-decay*lag)*response)
    kernel=torch.stack(kernels,-1)*admissible[...,None]
    history=torch.einsum('ijk,bjc->bick',kernel,drive)
    return history.flatten(-2)


class HybridGaussianResponse(nn.Module):
    """Static Gaussian graph coefficients contracted with causal linear drive.

    Material covariance is pushed forward externally from the returned shared
    nodal field. It is physical shape, never uncertainty. Exact supports and
    the two finite rigid patches retain the first model's decoder conventions.
    """
    def __init__(self,reference,tets,faces,*,width=64,layers=3,spatial_exchange=True,
                 frequencies=(0.,2.,6.,12.),decays=(0.,.5,2.)):
        super().__init__()
        self.width=width;self.layers=layers;self.spatial_exchange=bool(spatial_exchange)
        self.frequencies=tuple(frequencies);self.decays=tuple(decays)
        self.history_size=15*(2+len(self.frequencies)*len(self.decays))
        self.register_buffer('tets',tets.long());self.register_buffer('faces',faces.long())
        self.register_buffer('adjacency',adjacency(reference,tets))
        self.node_encoder=nn.Sequential(nn.Linear(23,width),nn.SiLU(),nn.Linear(width,width))
        self.global_encoder=nn.Sequential(nn.Linear(116,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width)) for _ in range(layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.body_coefficients=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,3*self.history_size))
        # One head shared across the two handles, each with its own Gaussian-
        # pooled spatial embedding and 52 local full-hardware/patch features.
        self.handle_coefficients=nn.Sequential(nn.Linear(width+52,width),nn.SiLU(),nn.Linear(width,6*self.history_size))

    def encode(self,node,global_features):
        h=self.node_encoder(node)+self.global_encoder(global_features)[:,None]
        for block,norm in zip(self.blocks,self.norms):
            exchange=torch.matmul(self.adjacency,h) if self.spatial_exchange else torch.zeros_like(h)
            h=norm(h+block(h+exchange))
        return h

    def forward(self,data,*,time_indices=None,prepared=None):
        node,g=prepare_features(data,self.tets,self.faces) if prepared is None else prepared
        h=self.encode(node,g);B,N,W=h.shape
        history=causal_history_basis(data['controls'],data['support_offsets'],data['times'],
            frequencies=self.frequencies,decays=self.decays)
        offsets=data['support_offsets']
        if time_indices is not None:
            history=history[:,time_indices];offsets=offsets[:,time_indices]
        coefficients=self.body_coefficients(h).reshape(B,N,3,self.history_size)
        displacement=.01*torch.einsum('bnok,btk->btno',coefficients,history)
        origin=data['rest'].mean(1)
        local=[];pooled=[]
        for j in range(2):
            ids=data['handle_patches'][:,j]
            ref=torch.gather(data['rest'],1,ids[...,None].expand(-1,-1,3))-data['handle_centers'][:,j,None]
            local.append(torch.cat([data['handle_masses'][:,j:j+1]/.001,
                (data['handle_inertias'][:,j]/1e-6).flatten(1),data['handle_com_offsets'][:,j]/.02,
                data['handle_initial_rotations'][:,j].flatten(1),(ref/.24).flatten(1),
                (data['handle_centers'][:,j]-origin)/.24],-1))
            weights=node[:,:,5+j]/N
            pooled.append((h*weights[...,None]).sum(1))
        handle_features=torch.cat([torch.stack(pooled,1),torch.stack(local,1)],-1)
        coefficients=self.handle_coefficients(handle_features).reshape(B,2,6,self.history_size)
        pose=torch.einsum('bhoz,btz->btho',coefficients,history)
        translation=.01*pose[...,:3];rotvec=.3*pose[...,3:]
        rotation=torch.matrix_exp(skew(rotvec))
        positions=data['rest'][:,None]+displacement
        for j in range(2):
            ids=data['handle_patches'][:,j]
            ref=torch.gather(data['rest'],1,ids[...,None].expand(-1,-1,3))-data['handle_centers'][:,j,None]
            patch=torch.einsum('btij,bnj->btni',rotation[:,:,j],ref)+data['handle_centers'][:,None,j,None]+translation[:,:,j,None]
            positions=positions.scatter(2,ids[:,None,:,None].expand(-1,history.shape[1],-1,3),patch)
        targets=data['rest'][:,None]+offsets[:,:,None]
        positions=torch.where(data['fixture_mask'][:,None,:,None],targets,positions)
        return {'positions_m':positions,'handle_centers_m':data['handle_centers'][:,None]+translation,
            'handle_rotations':rotation@data['handle_initial_rotations'][:,None],
            'latent_displacements_m':displacement,'handle_translations_m':translation,
            'handle_rotation_vectors_rad':rotvec,
            'role':'force_linear_latent_conditional_forward_not_posterior'}
