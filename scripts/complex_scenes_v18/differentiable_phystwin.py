"""Native Warp spring-force adjoints plus differentiable common boundaries.

Training uses the algebraically equivalent no-drag/no-ground semi-implicit step
in Torch; sealed forecasts use the author Warp integration kernels separately.
"""
import numpy as np
import torch
from torch import nn
from scripts.complex_scenes_v18.phystwin_scene import PhysTwinScene
from scripts.complex_scenes_v18.phystwin_autograd import spring_force
from scripts.complex_scenes_v18.explicit_boundaries import ExplicitTorchBoundaries


class DifferentiablePhysTwinScene:
    def __init__(self,case,variant=0,log10_k=-.3,per_edge=True,query=None):
        self.base=PhysTwinScene(case,variant,log10_k,query=query);self.count=len(self.base.x);self.names=list(self.base.rigid)
        self.boundary=ExplicitTorchBoundaries(self.base.scene,dtype=torch.float64)
        self.native=[];self.edges=[];self.rest=[];self.mass=[]
        for b in self.base.buffers:
            model=nn.Module();model.global_k=nn.Parameter(torch.full((len(b['edges_numpy']),),log10_k,device='cuda'))
            model.register_buffer('damp',torch.tensor(-10.,device='cuda'));self.native.append(model)
            self.edges.append(torch.tensor(b['edges_numpy'],dtype=torch.int32,device='cuda'))
            self.rest.append(torch.tensor(b['rest'].numpy(),device='cuda'));self.mass.append(torch.tensor(b['mass_numpy'],dtype=torch.float32,device='cuda'))
        self.parameters=[model.global_k for model in self.native]
        self.surface=[torch.tensor(b.B.toarray(),device='cuda',dtype=torch.float32)for b in self.base.sample]

    def initial(self):
        x=[a.to(device='cuda',dtype=torch.float32).detach().clone().requires_grad_(True)for a in self.base.x]
        v=[a.to(device='cuda',dtype=torch.float32).detach().clone()for a in self.base.v]
        centers=[torch.tensor(self.base.rigid[n]['center'],dtype=torch.float32,device='cuda',requires_grad=True)for n in self.names]
        rv=[torch.tensor(self.base.rigid[n]['velocity'],dtype=torch.float32,device='cuda')for n in self.names]
        return tuple(x+v+centers+rv)

    def step(self,state,index,dt,training=True):
        n=self.count;r=len(self.names);x=state[:n];v=state[n:2*n];centers=state[2*n:2*n+r];rv=state[2*n+r:]
        _,force,rf=self.boundary.evaluate([a.double()for a in x],{name:c.double()for name,c in zip(self.names,centers)},index*dt,create_graph=training)
        xx=[];vv=[];cc=[];rr=[]
        for i,(xi,vi,f,model)in enumerate(zip(x,v,force,self.native)):
            internal=spring_force(xi,vi,model.global_k*np.log(10.),self.edges[i],self.rest[i])
            gravity=xi.new_tensor(self.base.scene.models[i].gravity)
            vn=vi+dt*((internal+f.to(xi.dtype))/self.mass[i][:,None]+gravity)
            xx.append(xi+dt*vn);vv.append(vn)
        for name,c,velocity in zip(self.names,centers,rv):
            gravity=c.new_tensor(self.base.scene.models[0].gravity)
            vn=velocity+dt*(rf[name].to(c.dtype)/self.base.scene.rigid[name].mass+gravity)
            rr.append(vn);cc.append(c+dt*vn)
        result=tuple(xx+vv+cc+rr)
        return result if training else tuple(a.detach()for a in result)

    def positions(self,state):return torch.stack([B@x for B,x in zip(self.surface,state[:self.count])])

    def native_replay(self):
        for model,b,edges in zip(self.native,self.base.buffers,self.edges):
            if not np.array_equal(edges.cpu().numpy(),b['edges_numpy']):raise ValueError('Graph changed before native replay')
            b['Y'].assign((model.global_k.detach().cpu().numpy()*np.log(10.)).astype(np.float32))
        return self.base
