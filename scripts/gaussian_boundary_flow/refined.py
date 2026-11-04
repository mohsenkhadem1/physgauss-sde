"""Finer, small-strain development model; preserves the first pilot unchanged."""
import torch
from scripts.gaussian_boundary_flow.model import Block,LOW,HIGH
from scripts.gaussian_boundary_flow.qualification import refined_block


class RefinedBlock(Block):
    def __init__(self,dtype=torch.float32,device='cpu',n=3):
        b=refined_block(n)
        self.__dict__.update(b.__dict__)
        for name in ('K','p','a','N'):setattr(self,name,getattr(self,name).to(device=device,dtype=dtype))
        self.bottom=self.bottom.to(device);self.top=self.top.to(device)
        for name,value in vars(self.mesh).items():
            if isinstance(value,torch.Tensor):setattr(self.mesh,name,value.to(device=device,dtype=dtype if value.is_floating_point() else value.dtype))
        self.dtype=dtype;self.device=device;self.nominal_dt=.0025
        coarse=Block(dtype=dtype,device=device)
        self.corners=torch.cdist(coarse.mesh.rest,self.mesh.rest).argmin(-1)

    def parameters(self,z):
        # Regime fixed prospectively for a NEW independent evaluation cohort.
        shift=z.new_tensor([1.5,.8,0.,0.,0.,0.])
        return LOW.to(z)+shift+(HIGH-LOW).to(z)*torch.sigmoid(z)

    def rollout(self,z,*,kind='sine',dt=.0025,duration=1.,stride=10,corners_only=True):
        u=super().rollout(z,kind=kind,dt=dt,duration=duration,stride=stride)
        return u[:,:,self.corners] if corners_only else u
