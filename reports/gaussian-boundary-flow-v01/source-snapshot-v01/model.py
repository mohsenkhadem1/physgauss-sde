"""Controlled identification development harness; reuses production mesh/renderer.

Small-strain elastic tangent of physgauss.physics.material_response at rest.
No claim of a qualified finite-strain/contact solver or real RGB-only inference.
"""
import torch
from physgauss.physics import cube_mesh, material_response
from physgauss.gaussian_boundary import (surface_quadrature,gaussian_gate,
    robin_matrices,traction_force,LinearBoundarySystem)

PARAMETERS=['log_young_Pa','log_support_N_per_m3','log_load_N','support_x_m','load_x_m','log_damping_seconds']
LOW=torch.tensor([7.6,11.0,-1.6,-.035,-.035,-4.6])
HIGH=torch.tensor([9.9,13.4,.0,.035,.035,-2.3])


class Block:
    def __init__(self,dtype=torch.float32,device='cpu'):
        # Reuse existing tetrahedral mesh, linearize existing elastic response.
        source=cube_mesh(dtype=torch.float64,density=500.,side=.1)
        x=source.rest.flatten().requires_grad_(True)
        jac=torch.autograd.functional.jacobian(lambda q:material_response(q.reshape(-1,3),source,torch.tensor(1.,dtype=x.dtype))[0].flatten(),x)
        self.K=(-jac).to(dtype=dtype,device=device)
        self.mesh=cube_mesh(dtype=dtype,density=500.,side=.1)
        for key,value in vars(self.mesh).items():
            if isinstance(value,torch.Tensor):setattr(self.mesh,key,value.to(device))
        self.p,self.a,self.N=surface_quadrature(self.mesh)
        self.bottom=(self.p[:,1]-self.mesh.rest[:,1].min()).abs()<1e-6
        self.top=(self.p[:,1]-self.mesh.rest[:,1].max()).abs()<1e-6
        self.dtype=dtype;self.device=device

    def parameters(self,z):
        low=LOW.to(z);high=HIGH.to(z)
        # z is an unconstrained logit coordinate, sampled via logit(U).
        return low+(high-low)*torch.sigmoid(z)

    def operators(self,z,dt):
        theta=self.parameters(z);b=len(z);zero=torch.zeros(b,device=z.device,dtype=z.dtype)
        cov=torch.diag(z.new_tensor([.027**2,.02**2,.045**2])).expand(b,-1,-1)
        center=torch.stack((theta[:,3],zero+self.mesh.rest[:,1].min(),zero),-1)
        gate=gaussian_gate(self.p,center,cov)*self.bottom
        stiffness=torch.eye(3,device=z.device,dtype=z.dtype)[None]*theta[:,1].exp()[:,None,None]
        Ks,Ds=robin_matrices(self.N,self.a,gate,stiffness,stiffness*theta[:,5].exp()[:,None,None])
        K=self.K[None]*theta[:,0].exp()[:,None,None]+Ks
        mass=self.mesh.mass.repeat_interleave(3)[None].expand(b,-1)
        D=Ds+torch.diag_embed(mass*.25)
        center=torch.stack((theta[:,4],zero+self.mesh.rest[:,1].max(),zero),-1)
        loadgate=gaussian_gate(self.p,center,cov)*self.top
        # Applied resultant is a force in N; spatial traction normalized by area.
        loadgate=loadgate/(loadgate*self.a).sum(-1,keepdim=True)
        direction=z.new_tensor([.35,-1.,.15]);direction=direction/direction.norm()
        load=traction_force(self.N,self.a,loadgate,theta[:,2].exp()[:,None]*direction).flatten(1)
        gravity=(self.mesh.mass[:,None]*z.new_tensor([0.,-9.81,0.])).flatten().expand(b,-1)
        return LinearBoundarySystem(mass,K,D,dt),load,gravity

    def rollout(self,z,*,kind='sine',dt=.005,duration=1.,stride=5):
        system,load,gravity=self.operators(z,dt)
        u=torch.zeros_like(load);v=u.clone();states=[u]
        for step in range(1,round(duration/dt)+1):
            t=step*dt
            # All programs share the observed prefix; only unseen future differs.
            if t<=.4 or kind=='sine': command=.5*(1-torch.cos(z.new_tensor(2*torch.pi*2*t)))
            elif kind=='release': command=0.
            elif kind=='triangle': command=max(0.,1-abs((t-.4)/.3-1))
            elif kind=='reverse': command=-.8
            else:raise ValueError(kind)
            u,v,_=system.step(u,v,gravity+load*command)
            if step%stride==0:states.append(u)
        return torch.stack(states,1).reshape(len(z),-1,len(self.mesh.rest),3)


def observe(u,noise=True,generator=None):
    """Known-camera orthographic 2D tracks, 9 prefix frames. Not raw RGB input."""
    # First 0.4s only at 0.05s spacing; output histories at .025s spacing.
    basis=u.new_tensor([[.8660254,0.,.5],[0.,1.,0.]])
    tracks=torch.einsum('btni,ki->btnk',u[:,0:17:2],basis)
    if noise: tracks=tracks+torch.randn(tracks.shape,device=tracks.device,dtype=tracks.dtype,generator=generator)*.00005
    return tracks.flatten(1)
