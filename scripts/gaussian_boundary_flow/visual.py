"""Use the existing differentiable GaussianRenderer; no displacement magnification."""
import torch
from physgauss.render import GaussianRenderer


def cameras(mesh,times,orbit_degrees=25.,offset=25.):
    angle=(times/times[-1]-.5)*torch.pi*orbit_degrees/180+offset*torch.pi/180
    center=mesh.rest.mean(0)
    eye=center+torch.stack([.30*angle.sin(),torch.ones_like(angle)*.16,.30*angle.cos()],-1)
    forward=torch.nn.functional.normalize(center-eye,dim=-1)
    up=torch.zeros_like(forward);up[:,1]=1
    right=torch.nn.functional.normalize(torch.linalg.cross(forward,up),dim=-1)
    camera_up=torch.linalg.cross(right,forward)
    return eye,torch.stack([right,-camera_up,forward],-2)


def render(block,u,size=24,offset=25.,orbit_degrees=25.):
    r=GaussianRenderer(block.mesh,size)
    # Existing renderer uses a .65 m cube default: scale appearance footprints.
    r.sigma0=r.sigma0.to(u)*(.1/.65)**2
    r.bary=r.bary.to(u);r.colors=r.colors.to(u);r.pixels=r.pixels.to(u)
    times=torch.linspace(0,1,u.shape[1],device=u.device,dtype=u.dtype)
    eye,R=cameras(block.mesh,times,orbit_degrees,offset)
    return r(u+block.mesh.rest,eye,R)
