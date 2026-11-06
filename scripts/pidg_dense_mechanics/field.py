"""Dense image-connected mechanics with a frozen learned bulk law and Gaussian loads.
This is a soft mechanics constraint, not a certified forward contact simulator.
Spatial/temporal units remain those of the reconstruction, not calibrated SI.
"""
from pathlib import Path
import torch
from scripts.pidg_comparison.mechanics import ConstitutivePrior

class DenseMechanics(torch.nn.Module):
 def __init__(self,particles=384,kernels=16,knots=16):
  super().__init__();self.prior=ConstitutivePrior();self.particles=particles;self.kernels=kernels;self.knots=knots;self.ready=False;self.calls=0;self.last={};self.optimizer=None
 def initialize(self,deform,gaussians):
  from utils.general_utils import quaternion_to_matrix
  with torch.no_grad():
   # Opacity-ranked deterministic support; no scene-specific object segmentation.
   order=torch.argsort(gaussians.get_opacity.detach().flatten(),descending=True);pool=order[:max(self.particles,len(order)//2)];ids=pool[torch.linspace(0,len(pool)-1,min(self.particles,len(pool)),device='cuda').long()];q=gaussians.get_xyz.detach()[ids].clone()
   d=deform.step(q,torch.zeros(len(q),1,device=q.device))['d_xyz'];X=(quaternion_to_matrix(d[:,:4])@q[...,None]).squeeze(-1)+d[:,4:]
   origin=X.mean(0);scale=(X-origin).square().sum(1).mean().sqrt().clamp_min(.01);X=(X-origin)/scale
   dist=torch.cdist(X,X);near=dist.topk(min(13,len(X)),largest=False).indices[:,1:];dx=X[near]-X[:,None];weight=torch.exp(-dx.square().sum(-1)/(dx.square().sum(-1).mean(1,keepdim=True).clamp_min(1e-5)));A=torch.einsum('nk,nki,nkj->nij',weight,dx,dx);ridge=.001*A.diagonal(dim1=-2,dim2=-1).sum(1).clamp_min(1e-6);inv=torch.linalg.inv(A+ridge[:,None,None]*torch.eye(3,device=X.device));centers=X[torch.linspace(0,len(X)-1,self.kernels,device=X.device).long()]
   accelerations=[]
   for t in [.1,.3,.5,.7,.9]:
    ys=[]
    for offset in [-.01,0,.01]:
     pk=deform.step(q,torch.full((len(q),1),t+offset,device=q.device))['d_xyz'];ys.append(((quaternion_to_matrix(pk[:,:4])@q[...,None]).squeeze(-1)+pk[:,4:]-origin)/scale)
    accelerations.append((ys[2]-2*ys[1]+ys[0])/.0001)
   dynamic_scale=torch.stack(accelerations).square().mean().sqrt().clamp_min(1.)
  for name,value in dict(q=q,X=X,origin=origin,unit_scale=scale,near=near,dx=dx,weight=weight,inv=inv,centers=centers,dynamic_scale=dynamic_scale).items():self.register_buffer(name,value)
  self.load=torch.nn.Parameter(torch.zeros(self.knots,self.kernels,3,device='cuda'));self.targets=torch.nn.Parameter(centers[None].repeat(self.knots,1,1));self.stiffness=torch.nn.Parameter(torch.full((self.kernels,),-4.,device='cuda'));self.damping=torch.nn.Parameter(torch.full((self.kernels,),-4.,device='cuda'));self.gravity=torch.nn.Parameter(torch.zeros(3,device='cuda'));self.material=torch.nn.Parameter(torch.tensor([0.,0.,-2.],device='cuda'));self.optimizer=torch.optim.Adam([self.load,self.targets,self.stiffness,self.damping,self.gravity,self.material],lr=.002);self.ready=True
 def internal(self,y):
  delta=(y[self.near]-y[:,None])-self.dx;F=torch.eye(3,device=y.device)+torch.einsum('nk,nki,nkj->nij',self.weight,delta,self.dx)@self.inv;J=torch.linalg.det(F);valid=J>.05;safe=torch.where(valid[:,None,None],F,torch.eye(3,device=y.device));z=torch.stack([self.material[0].tanh(),self.material[1].tanh(),self.material[2].tanh()]).expand(len(y),3)
  energy=self.prior.law.energy(safe,z)/12000.+10*torch.relu(.1-J).square();force=-torch.autograd.grad(energy.sum(),y,create_graph=True,retain_graph=True)[0]
  return force,F,J,energy.mean()
 def external(self,y,v,t):
  s=t.clamp(0,1)*(self.knots-1);i=min(int(s.detach()),self.knots-2);a=s-i;load=(1-a)*self.load[i]+a*self.load[i+1];target=(1-a)*self.targets[i]+a*self.targets[i+1]
  raw=torch.exp(-.5*torch.cdist(y,self.centers).square()/.65**2);w=raw/(1+raw.sum(1,keepdim=True));k=torch.nn.functional.softplus(self.stiffness);c=torch.nn.functional.softplus(self.damping)
  return self.gravity+torch.einsum('nk,nkj->nj',w,load[None]+k[None,:,None]*(target[None]-y[:,None])-c[None,:,None]*v[:,None])
 def forward(self,deform,gaussians,fid):
  from utils.general_utils import quaternion_to_matrix
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
   if not self.ready:self.initialize(deform,gaussians)
   self.optimizer.zero_grad(set_to_none=True);t=fid.reshape(()).clamp(.01,.99);h=.01
   def mapping(time):
    p=deform.step(self.q,time.expand(len(self.q),1))['d_xyz'];return ((quaternion_to_matrix(p[:,:4])@self.q[...,None]).squeeze(-1)+p[:,4:]-self.origin)/self.unit_scale
   ym=mapping(t-h);y=mapping(t);yp=mapping(t+h);v=(yp-ym)/(2*h);acc=(yp-2*y+ym)/(h*h);internal,F,J,energy=self.internal(y);external=self.external(y,v,t)
   # Relative acceleration scaling is detached; it prevents large arbitrary units
   # from overpowering images, but does not calibrate physical mass or force.
   normalizer=self.dynamic_scale;residual=acc/normalizer-internal-external;balance=torch.nn.functional.smooth_l1_loss(residual,torch.zeros_like(residual));barrier=torch.relu(.1-J).square().mean();prior=.0001*(self.load.square().mean()+(self.targets[1:]-self.targets[:-1]).square().mean()+self.material.square().mean())
   loss=balance+.01*energy+.1*barrier+prior;self.calls+=1;self.last={'balance':float(balance.detach()),'energy':float(energy.detach()),'minJ':float(J.min().detach()),'inversion_fraction':float((J<=0).float().mean()),'loss':float(loss.detach()),'calls':self.calls};return loss
 def step(self,loss_weight=1.):
  if self.ready:
   for p in [self.load,self.targets,self.stiffness,self.damping,self.gravity,self.material]:
    if p.grad is not None:p.grad.div_(max(loss_weight,1e-12))
   torch.nn.utils.clip_grad_norm_([self.load,self.targets,self.stiffness,self.damping,self.gravity,self.material],1.);self.optimizer.step()
 def save(self,path):
  if self.ready:torch.save({'state_dict':self.state_dict(),'optimizer':self.optimizer.state_dict(),'last':self.last,'scope':'Dense renderer with soft learned-mechanics/Gaussian-load constraint; not solved forward dynamics or calibrated forces.'},Path(path))
