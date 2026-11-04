"""Conditional rectified flow over physical program parameters, not physical time."""
import torch
from torch import nn


class ConditionalFlow(nn.Module):
    def __init__(self,context_dim,parameter_dim=6,width=192):
        super().__init__()
        self.context=nn.Sequential(nn.Linear(context_dim,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.vector=nn.Sequential(nn.Linear(width+parameter_dim+3,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,parameter_dim))
        self.parameter_dim=parameter_dim

    def forward(self,z,s,context):
        c=self.context(context)
        return self.vector(torch.cat([z,s,torch.sin(torch.pi*s),torch.cos(torch.pi*s),c],-1))

    def loss(self,target,context):
        noise=torch.randn_like(target);s=torch.rand(len(target),1,device=target.device)
        point=(1-s)*noise+s*target
        return (self(point,s,context)-(target-noise)).square().mean()

    @torch.no_grad()
    def sample(self,context,count=32,steps=32,generator=None):
        context=context.repeat_interleave(count,0)
        z=torch.randn(len(context),self.parameter_dim,device=context.device,generator=generator)
        for i in range(steps):
            s=z.new_full((len(z),1),i/steps)
            k1=self(z,s,context)
            k2=self(z+k1/steps,s+1/steps,context)
            z=z+(k1+k2)/(2*steps)
        return z.reshape(-1,count,self.parameter_dim)
