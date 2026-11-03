"""Author Spring-Gaus internal dynamics with differentiable scene boundaries.

The original scalar adapter remains available for parity tests and initialization.
This enables native per-edge stiffness/damping optimization without detaching
contact or fixture state derivatives. It remains an explicitly adapted baseline.
"""
import torch
from torch import nn
from scripts.complex_scenes_v18.spring_scene import SpringScene
from scripts.complex_scenes_v18.torch_boundaries import TorchBoundaries


class DifferentiableSpringScene:
    def __init__(self,case,variant=0,log10_k=-.3,per_edge=True):
        self.base=SpringScene(case,variant,log10_k=log10_k);self.native=self.base.native
        self.boundary=TorchBoundaries(self.base.scene,dtype=torch.float64);self.count=len(self.native);self.names=list(self.base.rigid)
        if per_edge:
            for model in self.native:
                model.global_k=nn.Parameter(torch.full_like(model.origin_len,log10_k));model.damp.requires_grad_(True)
        self.parameters=[p for model in self.native for p in (model.global_k,model.damp)if p.requires_grad]
        self.surface=[torch.tensor(b.B.toarray(),device='cuda',dtype=torch.float32)for b in self.base.sample]

    def initial(self):
        x=[a.detach().clone().requires_grad_(True)for a in self.base.x];v=[a.detach().clone()for a in self.base.v]
        centers=[torch.tensor(self.base.rigid[n]['center'],dtype=torch.float32,device='cuda',requires_grad=True)for n in self.names]
        rv=[torch.tensor(self.base.rigid[n]['velocity'],dtype=torch.float32,device='cuda')for n in self.names]
        return tuple(x+v+centers+rv)

    def step(self,state,index,dt,training=True):
        n=self.count;r=len(self.names);x=state[:n];v=state[n:2*n];centers=state[2*n:2*n+r];rv=state[2*n+r:]
        with torch.enable_grad():_,force,rf=self.boundary.evaluate([a.double()for a in x],{name:c.double()for name,c in zip(self.names,centers)},index*dt,create_graph=training)
        xx=[];vv=[];cc=[];rr=[]
        with torch.set_grad_enabled(training):
            for model,xi,vi,f in zip(self.native,x,v,force):
                model.scene_external_force=f.to(xi.dtype)
                K=10**model.global_k.expand_as(model.origin_len)/(model.origin_len+model.eps);damp=10**model.damp/(model.origin_len+model.eps)
                xn,vn=model.step(xi,vi,K,10**model.global_m,torch.sigmoid(model.rebound_k),torch.sigmoid(model.fric_k),damp,dt)
                xx.append(xn);vv.append(vn)
            for name,c,velocity in zip(self.names,centers,rv):
                body=self.base.scene.rigid[name];gravity=c.new_tensor(self.base.scene.models[0].gravity)
                newv=velocity+dt*(rf[name].to(c.dtype)/body.mass+gravity);rr.append(newv);cc.append(c+dt*newv)
        if not training:
            xx=[a.detach().requires_grad_(True)for a in xx];cc=[a.detach().requires_grad_(True)for a in cc]
            vv=[a.detach()for a in vv];rr=[a.detach()for a in rr]
        return tuple(xx+vv+cc+rr)

    def positions(self,state):return torch.stack([B@x for B,x in zip(self.surface,state[:self.count])])
