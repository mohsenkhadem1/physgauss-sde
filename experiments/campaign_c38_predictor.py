"""Existing Gaussian graph/causal-GRU pattern adapted to actual P2/nohandle data.

This is a conditional forward surrogate given known/candidate physical fields,
NOT a public sparse-view encoder or a calibrated posterior. Sparse aggregation
is algebraically the old normalized graph exchange, without an N×N allocation.
Consistent signed P2 force basis is input; Gaussian features never replace it.
"""
from __future__ import annotations
import torch
from torch import nn
from experiments.campaign_hybrid_predictor import differentiable_surface_weights

def sparse_adjacency(rest,elements,bandwidth=.04):
    if rest.ndim!=2 or rest.shape[1]!=3 or elements.ndim!=2 or elements.shape[1]!=10 or bandwidth<=0:raise ValueError('P2 reference nodes/elements and positive graph bandwidth required')
    if not bool(torch.isfinite(rest).all()) or elements.dtype not in (torch.int32,torch.int64) or not len(elements) or bool((elements<0).any()) or bool((elements>=len(rest)).any()):raise ValueError('Finite P2 reference geometry and valid integer connectivity required')
    n=len(rest);pairs=[]
    for i in range(10):
        for j in range(i+1,10):pairs.extend([elements[:,[i,j]],elements[:,[j,i]]])
    ids=torch.arange(n,device=rest.device);pairs.append(torch.stack([ids,ids],1))
    edges=torch.unique(torch.cat(pairs,0),dim=0).T.long()
    values=torch.exp(-((rest[edges[0]]-rest[edges[1]])/bandwidth).square().sum(-1)/2)
    total=rest.new_zeros(n).index_add(0,edges[0],values)
    return torch.sparse_coo_tensor(edges,values/total[edges[0]],(n,n),device=rest.device,check_invariants=True).coalesce()

def exchange(adjacency,node_features):
    B,N,W=node_features.shape
    value=node_features.transpose(0,1).reshape(N,B*W)
    return torch.sparse.mm(adjacency,value).reshape(N,B,W).transpose(0,1)

class BlockGaussianPredictor(nn.Module):
    def __init__(self,reference,elements,surface_triangles,control_scales,*,width=64,layers=3,spatial_exchange=True,time_scale=.5,force_scale_N=1.):
        super().__init__()
        scales=torch.as_tensor(control_scales,dtype=reference.dtype,device=reference.device)
        if scales.ndim!=1 or not 1<=len(scales)<=15 or not bool(torch.isfinite(scales).all()) or bool((scales<=0).any()) or width<4 or layers<1 or time_scale<=0 or force_scale_N<=0:raise ValueError('Fixed physical control scales and model budget required')
        self.channels=len(scales);self.width=width;self.spatial_exchange=bool(spatial_exchange);self.time_scale=float(time_scale);self.force_scale=float(force_scale_N)
        self.length=float((reference.max(0).values-reference.min(0).values).max())
        if self.length<=0:raise ValueError('Nonzero reference geometry required')
        self.register_buffer('faces',surface_triangles.long());self.register_buffer('control_scales',scales)
        self.register_buffer('adjacency',sparse_adjacency(reference,elements.long()))
        self.node_encoder=nn.Sequential(nn.Linear(12+3*self.channels,width),nn.SiLU(),nn.Linear(width,width))
        self.global_encoder=nn.Sequential(nn.Linear(43,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width)) for _ in range(layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.memory=nn.GRU(self.channels+4,width,batch_first=True)
        self.body_head=nn.Sequential(nn.Linear(2*width+3,width),nn.SiLU(),nn.Linear(width,3))

    def prepare(self,context):
        if context.get('information_role') not in ('known_physics_oracle_A','candidate_physics_B'):raise ValueError('Explicit known/candidate physics role required; unknown truth is not public B input')
        rest=context['rest'];B,N,_=rest.shape;m=context['material'];mask=context['fixture_mask'];basis=context['load_basis_N']
        centers=context['load_centers_material_m'];cov=context['load_covariances_material_m2'];active=context['load_active']
        if m.shape!=(B,4) or mask.shape!=(B,N) or mask.dtype!=torch.bool or basis.shape!=(B,self.channels,N,3) or centers.shape!=(B,2,3) or cov.shape!=(B,2,3,3) or active.shape!=(B,2) or active.dtype!=torch.bool:raise ValueError('Actual P2/nohandle material/fixture/load context shapes required')
        if not all(bool(torch.isfinite(x).all()) for x in (rest,m,basis,centers,cov)) or bool((m[:,:2]<=0).any()) or bool(((m[:,2]<=0)|(m[:,2]>=.5)).any()) or bool((m[:,3]<0).any()):raise ValueError('Finite physical context; E/rho>0,0<nu<.5,drag>=0 required')
        origin=rest.mean(1,keepdim=True);gaussian=[]
        for b in range(B):
            if bool((torch.linalg.cholesky_ex(cov[b])[1]!=0).any()):raise ValueError('SPD material-frame load Gaussian covariance required')
            w,_=differentiable_surface_weights(rest[b],self.faces,centers[b],cov[b],active[b]);gaussian.append(w.T*N)
        physical_basis=(basis*self.control_scales[None,:,None,None]/self.force_scale).permute(0,2,3,1).flatten(-2)
        delta=((rest[:,:,None]-centers[:,None])/self.length).flatten(-2)
        node=torch.cat([(rest-origin)/self.length,mask[...,None].to(rest.dtype),physical_basis,torch.stack(gaussian),delta],-1)
        count=mask.sum(1,keepdim=True);safe=count.clamp_min(1);centroid=(rest*mask[...,None]).sum(1)/safe
        relative=(rest-centroid[:,None])/self.length
        support_cov=torch.einsum('bn,bni,bnj->bij',mask.to(rest.dtype),relative,relative)/safe[...,None]
        material=torch.stack([torch.log(m[:,0]/5000),torch.log(m[:,1]/625),(m[:,2]-.35)/.1,torch.log1p(m[:,3]/.6)],-1)
        global_features=torch.cat([material,count.to(rest.dtype)/N,(centroid-origin[:,0])/self.length,support_cov.flatten(1),
            ((centers-origin)/self.length).flatten(1),(cov/self.length**2).flatten(1),active.to(rest.dtype)],-1)
        return node,global_features

    def forward(self,context,coefficients,times,*,support_offsets=None,time_indices=None,prepared=None):
        """Coefficient row i is the held drive on (times[i-1],times[i]].

        Row zero is ignored because the episode starts at known rest. This is
        a conditional full-history decoder, not assimilation of observed RGBD.
        """
        node,g=self.prepare(context) if prepared is None else prepared
        B,N,_=node.shape;T=len(times)
        if coefficients.shape!=(B,T,self.channels) or times.ndim!=1 or T<1 or not bool(torch.isfinite(coefficients).all()) or not bool(torch.isfinite(times).all()) or abs(float(times[0]))>1e-12 or (T>1 and bool((times[1:]<=times[:-1]).any())):raise ValueError('Finite causal known interval drive and ordered clock from0 required')
        offsets=coefficients.new_zeros(B,T,3) if support_offsets is None else support_offsets
        if offsets.shape!=(B,T,3) or not bool(torch.isfinite(offsets).all()) or bool((offsets[:,0]!=0).any()):raise ValueError('Known exact fixture offset from rest required')
        h=self.node_encoder(node)+self.global_encoder(g)[:,None]
        for block,norm in zip(self.blocks,self.norms):
            message=exchange(self.adjacency,h) if self.spatial_exchange else torch.zeros_like(h)
            h=norm(h+block(h+message))
        dt=torch.cat([times.new_zeros(1),times[1:]-times[:-1]])/self.time_scale
        drive=torch.cat([coefficients/self.control_scales,offsets/self.length,dt[None,:,None].expand(B,-1,-1)],-1)
        # Row0 has no preceding interval; known future drives may affect only
        # predictions at/after their interval, never earlier memory states.
        drive=drive.clone();drive[:,0,:-1]=0
        baseline=torch.cat([torch.zeros_like(drive[...,:-1]),drive[...,-1:]],-1)
        memory,_=self.memory(drive);reference_memory,_=self.memory(baseline)
        selected=torch.arange(T,device=times.device) if time_indices is None else time_indices
        memory=memory[:,selected];reference_memory=reference_memory[:,selected];ts=times[selected]/self.time_scale
        tf=torch.stack([ts,torch.sin(ts),torch.cos(ts)],-1)
        def body(mem):
            count=mem.shape[1]
            return self.body_head(torch.cat([h[:,None].expand(-1,count,-1,-1),mem[:,:,None].expand(-1,-1,N,-1),tf[None,:,None].expand(B,-1,N,-1)],-1))
        displacement=.01*(body(memory)-body(reference_memory))
        positions=context['rest'][:,None]+displacement
        target=context['rest'][:,None]+offsets[:,selected,None]
        positions=torch.where(context['fixture_mask'][:,None,:,None],target,positions)
        return dict(positions_m=positions,latent_displacements_m=displacement,
            role='conditional_forward_given_physical_hypothesis_not_sparseview_posterior',physical_shape_covariance_is_uncertainty=False)
