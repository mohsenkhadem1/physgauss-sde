"""Causal conditional deformation operator for material/boundary Gaussians.

This is a learned forward approximation given a *physical hypothesis*, not an
image encoder or calibrated posterior. A shared nodal deformation field drives
volumetric material kernels; their F and covariance are never independent heads.
Exact support membership and finite rigid attachments belong to the supplied
hypothesis/hardware. Gaussian boundary weights are interaction features, not a
replacement for the reference solver's force/torque law.
"""
from __future__ import annotations

import math
import torch
from torch import nn

MATERIAL_FIELDS=('youngs_modulus_Pa','density_kg_m3','poissons_ratio','linear_damping_s_inverse')
def skew(value):
    x, y, z = value.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(*value.shape[:-1],3,3)


def adjacency(rest, tets):
    """Reference Gaussian-neighbour exchange on the supplied mesh edges."""
    n = len(rest)
    matrix = torch.eye(n, dtype=rest.dtype, device=rest.device)
    for i in range(4):
        for j in range(i+1,4):
            a,b=tets[:,i],tets[:,j]
            weight=torch.exp(-((rest[a]-rest[b])/.04).square().sum(-1)/2)
            matrix[a,b]=weight;matrix[b,a]=weight
    return matrix/matrix.sum(-1,keepdim=True)


def differentiable_surface_weights(rest, faces, centers, covariances, active):
    """Torch counterpart of the independent area-lumped boundary quadrature.

    Keeps hypothesis locations/bandwidths differentiable on CPU or GPU. The
    Gaussian normalizing constant cancels in each discrete patch normalization.
    """
    triangles=rest[faces]
    area=torch.linalg.vector_norm(torch.linalg.cross(triangles[:,1]-triangles[:,0],
        triangles[:,2]-triangles[:,0],dim=-1),dim=-1)/2
    lumped=rest.new_zeros(len(rest))
    for i in range(3):lumped=lumped.index_add(0,faces[:,i],area/3)
    delta=rest[None]-centers[:,None]
    precision=torch.linalg.inv(covariances)
    distance=(delta[..., :,None]*precision[:,None]*delta[...,None,:]).sum((-2,-1))
    log_area=torch.where(lumped>0,lumped.clamp_min(torch.finfo(rest.dtype).tiny).log(),-torch.inf)
    weights=torch.softmax(log_area[None]-.5*distance,dim=-1)
    return torch.where(active[:,None],weights,torch.zeros_like(weights)),lumped


def prepare_features(data, tets, faces):
    """Pure feature adapter: labels, IDs, solver clocks and future q are unused.

    Material/support fields are supplied candidate hypotheses. Only hardware,
    rest geometry and known controls could be public image-conditioning inputs.
    """
    rest=data['rest']; B,N,_=rest.shape
    if data['handle_patches'].shape!=(B,2,9):
        raise ValueError('Exactly two ordered nine-node physical handles required')
    origin=rest.mean(1,keepdim=True)
    handle=data['handle_centers']; support=data['support_centers']
    centers=torch.cat([support,handle],1)
    active=torch.cat([data['support_active'].bool(),torch.ones(B,2,dtype=torch.bool,device=rest.device)],1)
    weights=[]; volumes=[];patch_features=[];patch_relative=[]
    for b in range(B):
        cov=[]
        for j in range(4):
            if j<2:
                cov.append(torch.eye(3,device=rest.device,dtype=rest.dtype)*.018**2)
            else:
                points=rest[b,data['handle_patches'][b,j-2]]-handle[b,j-2]
                cov.append(points.T@points/len(points)+torch.eye(3,device=rest.device,dtype=rest.dtype)*.005**2)
        w,_=differentiable_surface_weights(rest[b],faces,centers[b],torch.stack(cov),active[b])
        weights.append(w.T)
        vertices=rest[b,tets]
        volume=torch.linalg.det((vertices[:,1:]-vertices[:,:1]).transpose(-1,-2)).abs()/6
        lumped=rest.new_zeros(N)
        for k in range(4):lumped.index_add_(0,tets[:,k],volume/4)
        volumes.append(lumped/lumped.mean())
        membership=rest.new_zeros(N,2)
        for j in range(2):membership[data['handle_patches'][b,j],j]=1
        if bool((membership.sum(-1)>1).any()) or bool((membership.sum(-1).bool()&data['fixture_mask'][b]).any()):
            raise ValueError('Rigid patches must be disjoint from each other and exact fixtures')
        patch_features.append(membership)
        patch_relative.append((rest[b,data['handle_patches'][b]]-handle[b,:,None])/.24)
    # Weights are normalized quadrature couplings. Scaling N is geometry-only.
    gaussian=torch.stack(weights)*N
    delta=(rest[:,:,None]-centers[:,None])/.24
    node=torch.cat([(rest-origin)/.24,gaussian,data['fixture_mask'][...,None].to(rest.dtype),
        torch.stack(patch_features),delta.flatten(-2),torch.stack(volumes)[...,None]],-1)
    m=data['material']  # Explicit pilot order MATERIAL_FIELDS; not a legacy vector.
    if m.shape!=(B,4) or bool((m[:,[0,1,3]]<=0).any()) or bool(((m[:,2]<=0)|(m[:,2]>=.5)).any()):
        raise ValueError('Material order must be E, rho, nu, drag with positive coefficients and 0<nu<.5')
    material=torch.stack([torch.log(m[:,0]/5000),torch.log(m[:,1]/625),(m[:,2]-.35)/.1,torch.log(m[:,3]/.6)],-1)
    global_features=torch.cat([material,((support-origin)/.24).flatten(1),data['support_active'].to(rest.dtype),
        ((handle-origin)/.24).flatten(1),data['handle_masses']/.001,
        (data['handle_inertias']/1e-6).flatten(1),(data['handle_com_offsets']/.02).flatten(1),
        data['handle_initial_rotations'].flatten(1),torch.stack(patch_relative).flatten(1)],-1)
    if node.shape[-1]!=23 or global_features.shape[-1]!=116:
        raise ValueError('Complete fixed-capacity feature schema changed')
    return node,global_features


class HybridGaussianPredictor(nn.Module):
    """Shared mechanical-feature network plus causal known-actuator memory.

    The no-exchange ablation has identical trainable parameter count and inputs.
    Neither model is a conservative integrator or an uncertainty distribution.
    """
    def __init__(self, reference, tets, faces, *, width=64, layers=3, spatial_exchange=True):
        super().__init__()
        self.width=width;self.layers=layers;self.spatial_exchange=bool(spatial_exchange)
        self.register_buffer('tets',tets.long());self.register_buffer('faces',faces.long())
        self.register_buffer('adjacency',adjacency(reference,tets))
        self.node_encoder=nn.Sequential(nn.Linear(23,width),nn.SiLU(),nn.Linear(width,width))
        self.global_encoder=nn.Sequential(nn.Linear(116,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width)) for _ in range(layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.memory=nn.GRU(15,width,batch_first=True)
        self.body_head=nn.Sequential(nn.Linear(2*width+3,width),nn.SiLU(),nn.Linear(width,3))
        self.handle_head=nn.Sequential(nn.Linear(2*width+3,width),nn.SiLU(),nn.Linear(width,12))

    def encode(self,node,global_features):
        h=self.node_encoder(node)+self.global_encoder(global_features)[:,None]
        for block,norm in zip(self.blocks,self.norms):
            exchange=torch.matmul(self.adjacency,h) if self.spatial_exchange else torch.zeros_like(h)
            h=norm(h+block(h+exchange))
        return h

    def forward(self,data, *, time_indices=None, prepared=None):
        node,g=prepare_features(data,self.tets,self.faces) if prepared is None else prepared
        h=self.encode(node,g);B,N,W=h.shape
        controls=data['controls']/.1
        support=data['support_offsets']/.01
        if bool((support[:,0]!=0).any()):raise ValueError('Fresh rest hypothesis requires zero initial support offset')
        drive=torch.cat([controls,support],-1)
        # Frame zero has no preceding action interval. A force value at the
        # initial instant cannot cause an instantaneous position jump.
        drive=torch.cat([torch.zeros_like(drive[:,:1]),drive[:,1:]],1)
        memory,_=self.memory(drive)
        reference_memory,_=self.memory(torch.zeros_like(drive))
        times=data['times']
        if times.ndim!=1 or len(times)!=drive.shape[1]:raise ValueError('One explicit common public query clock required')
        if time_indices is not None:
            memory=memory[:,time_indices];reference_memory=reference_memory[:,time_indices]
            times=times[time_indices];support=support[:,time_indices]
        T=len(times)
        tf=torch.stack([times/3,torch.sin(math.pi*times/3),torch.cos(math.pi*times/3)],-1)
        static=h[:,None].expand(-1,T,-1,-1)
        def body(m):
            return self.body_head(torch.cat([static,m[:,:,None].expand(-1,-1,N,-1),tf[None,:,None].expand(B,-1,N,-1)],-1))
        displacement=.01*(body(memory)-body(reference_memory))
        positions=data['rest'][:,None]+displacement
        # Predict finite handle poses, not nine unrelated attachment motions.
        pooled=h.mean(1)[:,None].expand(-1,T,-1)
        def handle(m):return self.handle_head(torch.cat([pooled,m,tf[None].expand(B,-1,-1)],-1))
        pose=(handle(memory)-handle(reference_memory)).reshape(B,T,2,6)
        translation=.01*pose[...,:3];rotation=torch.matrix_exp(skew(.3*pose[...,3:]))
        for j in range(2):
            ids=data['handle_patches'][:,j]
            ref=torch.gather(data['rest'],1,ids[...,None].expand(-1,-1,3))-data['handle_centers'][:,j,None]
            patch=torch.einsum('btij,bnj->btni',rotation[:,:,j],ref)+data['handle_centers'][:,None,j,None]+translation[:,:,j,None]
            positions=positions.scatter(2,ids[:,None,:,None].expand(-1,T,-1,3),patch)
        targets=data['rest'][:,None]+.01*support[:,:,None]
        positions=torch.where(data['fixture_mask'][:,None,:,None],targets,positions)
        return {'positions_m':positions,'handle_centers_m':data['handle_centers'][:,None]+translation,
            'handle_rotations':rotation@data['handle_initial_rotations'][:,None],
            'role':'conditional_forward_given_physical_hypothesis_not_posterior'}

