"""Differentiable adapter from the paper's Q2 mechanics to PIDG's rasterizer.
No learned motion field, target positions, or independent Gaussian deformation.
"""
import numpy as np
import torch
from scripts.phystwin.tensor_fixture import basis
from scripts.compositional_dynamics_v17.gaussian_transport import bind_gaussians

class Q2Binding:
    def __init__(self, model, centers, covariance, device='cpu', dtype=torch.float64):
        b=bind_gaussians(model.geometry,centers,covariance);q=b.coordinates
        old=q*np.array([.24,.08,.08])+[0,-.04,-.04]
        cell=np.stack([np.clip(np.searchsorted(a,old[:,i],side='right')-1,0,len(a)-2) for i,a in enumerate(model.axes)],1)
        lo=np.stack([a[cell[:,i]] for i,a in enumerate(model.axes)],1);width=np.stack([np.diff(a)[cell[:,i]] for i,a in enumerate(model.axes)],1)
        N,G=basis(2*(old-lo)/width-1);ids=model.elements[np.ravel_multi_index(cell.T,tuple(model.cells))]
        jac=np.einsum('pai,paJ->piJ',model.rest[ids],G);grad=np.einsum('paJ,pJi->pai',G,np.linalg.inv(jac))
        self.ids=torch.as_tensor(ids,device=device,dtype=torch.long)
        for name,value in [('N',N),('grad',grad),('offset',b.offset_m),('covariance',b.covariance_m2)]:setattr(self,name,torch.as_tensor(value,device=device,dtype=dtype))
    def transport(self, nodes):
        x=nodes[self.ids];F=torch.einsum('pai,paJ->piJ',x,self.grad)
        det=torch.linalg.det(F)
        if not torch.isfinite(F).all() or (det<=0).any():raise ValueError('Nonpositive material Jacobian: reject, do not silently repair')
        centers=torch.einsum('pa,pai->pi',self.N,x)+torch.einsum('pij,pj->pi',F,self.offset)
        covariance=F@self.covariance@F.transpose(-1,-2)
        # Newton polar iteration avoids undefined SVD gradients at repeated singular values.
        rotation=F
        for _ in range(12):rotation=.5*(rotation+torch.linalg.inv(rotation).transpose(-1,-2))
        if not torch.allclose(rotation.transpose(-1,-2)@rotation,torch.eye(3,device=nodes.device,dtype=nodes.dtype).expand_as(rotation),atol=1e-5,rtol=1e-5):raise ValueError('Polar iteration failed')
        return {'centers':centers,'covariance':covariance,'rotation':rotation,'determinant':det}

def render(camera, appearance, state, background):
    """Reuse upstream rasterizer; radiance directions are pulled into material frame."""
    from types import SimpleNamespace
    from gaussian_renderer import render as native_render
    from utils.sh_utils import eval_sh
    class BoundAppearance:
        active_sh_degree=appearance.active_sh_degree
        max_sh_degree=appearance.max_sh_degree
        get_xyz=state['centers']
        get_opacity=appearance.get_opacity
        get_features=appearance.get_features
        def get_covariance(self, scaling_modifier=1.):
            c=state['covariance']*scaling_modifier**2
            return torch.stack([c[:,0,0],c[:,0,1],c[:,0,2],c[:,1,1],c[:,1,2],c[:,2,2]],-1)
    direction=state['centers']-camera.camera_center
    direction=torch.einsum('pji,pj->pi',state['rotation'],direction)
    direction=direction/torch.linalg.vector_norm(direction,dim=-1,keepdim=True).clamp_min(1e-12)
    sh=appearance.get_features.transpose(1,2).reshape(-1,3,(appearance.max_sh_degree+1)**2)
    color=(eval_sh(appearance.active_sh_degree,sh,direction)+.5).clamp_min(0)
    return native_render(camera,BoundAppearance(),SimpleNamespace(compute_cov3D_python=True,convert_SHs_python=False,debug=False),background,None,0.,0.,False,override_color=color)
