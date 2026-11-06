"""Prefix-identified Gaussian interactions driving a general tetrahedral FEM.
Reconstruction units; inferred body load is not claimed as calibrated gravity.
"""
import torch,numpy as np
from types import SimpleNamespace
from scripts.solver_driven_gaussians.dynamics import DynamicFEM
from scripts.mechanics_gaussian_bridge.bridge import render

class SceneFEM(DynamicFEM):
 def __init__(self,adapter,appearance):
  super().__init__(cells=(1,1,1),device='cuda',dtype=torch.float32)
  self.rest=self.t(adapter['rest']);self.elements=torch.as_tensor(adapter['elements'],device='cuda',dtype=torch.long);dm=(self.rest[self.elements[:,1:]]-self.rest[self.elements[:,:1]]).transpose(-1,-2);self.volume=torch.linalg.det(dm)/6;inv=torch.linalg.inv(dm);self.grad=torch.cat([-inv.sum(1,keepdim=True),inv],1);assert bool((self.volume>0).all());self.mass=torch.zeros(len(self.rest),device='cuda').index_add(0,self.elements.flatten(),(self.volume[:,None]/4).expand(-1,4).flatten());assert bool((self.mass>0).all())
  self.ids=torch.tensor(adapter['ids'],device='cuda');self.N=self.t(adapter['N']);self.origin=self.t(adapter['origin']);self.length=float(adapter['length']);self.centers=appearance['centers'].cuda();self.cov=appearance['covariance'].cuda();self.radiance_frame=appearance.get('radiance_frame','material');self.appearance=SimpleNamespace(active_sh_degree=3,max_sh_degree=3,get_features=appearance['features'].cuda(),get_opacity=appearance['opacity'].cuda());self.t0=float(adapter['times'][0]);self.end=float(adapter['times'][-1]);self.times=self.t(adapter['times']);self.observed=self.t(adapter['x']);self.observed_v=self.t(adapter['v'])
  self.raw_v=self.observed_v.clone();self.rigid_v=torch.zeros_like(self.raw_v)
  labels=np.asarray(adapter['component_labels']) if 'component_labels' in adapter else np.zeros(len(self.elements),int)
  with torch.no_grad():
   for label in np.unique(labels):
    ids=torch.unique(self.elements[torch.tensor(labels==label,device='cuda')]);w=self.mass[ids];w=w/w.sum();xx=self.observed[:,ids];vv=self.raw_v[:,ids];center=(xx*w[None,:,None]).sum(1);vc=(vv*w[None,:,None]).sum(1);q=xx-center[:,None];I=torch.eye(3,device='cuda')[None]*(q.square().sum(-1)*w[None]).sum(1)[:,None,None]-torch.einsum('tni,tnj,n->tij',q,q,w);torque=(torch.linalg.cross(q,vv-vc[:,None])*w[None,:,None]).sum(1);omega=torch.linalg.solve(I+1e-8*torch.eye(3,device='cuda'),torque[...,None])[...,0];self.rigid_v[:,ids]=vc[:,None]+torch.linalg.cross(omega[:,None].expand_as(q),q)
  # Deterministic farthest-point kernel centers, including each disconnected volume.
  pts=self.rest.detach();chosen=[0];dist=(pts-pts[0]).square().sum(1)
  for _ in range(11):j=int(dist.argmax());chosen.append(j);dist=torch.minimum(dist,(pts-pts[j]).square().sum(1))
  centers=pts[chosen];raw=torch.exp(-.5*((pts[:,None]-centers[None])/.22).square().sum(-1));self.weights=raw/(.1+raw.sum(-1,keepdim=True));self.param=torch.nn.ParameterDict({'log_scale':torch.nn.Parameter(self.t(0.)),'log_damping':torch.nn.Parameter(self.t(-1.)),'log_stiffness':torch.nn.Parameter(torch.full((12,),-4.,device='cuda')),'loads':torch.nn.Parameter(torch.zeros(8,12,3,device='cuda')),'body':torch.nn.Parameter(torch.zeros(3,device='cuda'))})
 def set_velocity_factor(self,factor):
  self.observed_v=self.rigid_v+factor*(self.raw_v-self.rigid_v)
 def external(self,x,v,t,mode='constant'):
  u=(t-self.t0)/(self.end-self.t0)*7;idx=max(0,min(6,int(u)));alpha=max(0,min(1,float(u-idx)));loads=(1-alpha)*self.param['loads'][idx]+alpha*self.param['loads'][idx+1]
  if t>self.end:
   if mode=='zero':loads=loads*0
   elif mode=='linear':loads=loads+(t-self.end)/(self.end-self.t0)*7*(self.param['loads'][-1]-self.param['loads'][-2])
   elif mode=='decay':loads=loads*float(np.exp(-(t-self.end)/.08))
  stiffness=self.weights@torch.nn.functional.softplus(self.param['log_stiffness']);a=self.weights@loads+self.param['body']-stiffness[:,None]*(x-self.rest)-torch.nn.functional.softplus(self.param['log_damping'])*v
  return self.mass[:,None]*a
 def step(self,x,v,t,dt,mode='constant',ad=False):
  internal,_,J=self.bulk(x,scale=self.param['log_scale'].exp()/12000,kind='learned',create_graph=ad);force=internal+self.external(x,v,t,mode);v=v+dt*force/self.mass[:,None];x=x+dt*v;return x,v
 def integrate(self,x,v,start,end,mode='constant',ad=False,maxdt=.001):
  from torch.utils.checkpoint import checkpoint
  n=max(1,int(np.ceil((end-start)/maxdt)));dt=(end-start)/n
  for begin in range(0,n,10):
   stop=min(n,begin+10)
   def segment(xx,vv,begin=begin,stop=stop):
    for j in range(begin,stop):xx,vv=self.step(xx,vv,start+j*dt,dt,mode,ad)
    return xx,vv
   if ad:x,v=checkpoint(segment,x,v,use_reentrant=False)
   else:x,v=segment(x,v)
  return x,v
 def state(self,x):
  dx=(x-self.rest)[self.ids];F=torch.eye(3,device='cuda')+torch.einsum('pai,paj->pij',dx,self.grad[self.adapter_tet]);J=torch.linalg.det(F)
  if not bool(torch.isfinite(J).all()) or float(J.detach().min())<=0:raise ValueError('Invalid Gaussian transport')
  R=F
  for _ in range(10):R=.5*(R+torch.linalg.inv(R).transpose(-1,-2))
  if not torch.allclose(R.transpose(-1,-2)@R,torch.eye(3,device='cuda').expand_as(R),atol=1e-3,rtol=1e-3):raise ValueError('Unconverged polar rotation')
  state=dict(centers=self.centers+self.length*torch.einsum('pa,pai->pi',self.N,dx),covariance=F@self.cov@F.transpose(-1,-2),rotation=R if self.radiance_frame=='material' else torch.eye(3,device='cuda').expand_as(R),determinant=J)
  return state
 def image(self,x,camera):
  return render(camera,self.appearance,self.state(x),torch.zeros(3,device='cuda'))['render'].clamp(0,1)
 def set_binding_tets(self):
  # Exact lookup of each binding's ordered element IDs.
  lookup={tuple(e):i for i,e in enumerate(self.elements.cpu().numpy())};self.adapter_tet=torch.tensor([lookup[tuple(e)] for e in self.ids.cpu().numpy()],device='cuda')
 def initialize_loads(self):
  from scipy.signal import savgol_filter
  x=self.observed.detach();v=self.observed_v.detach();a=self.t(savgol_filter(x.cpu().numpy(),9,3,deriv=2,delta=float(self.times[1]-self.times[0]),axis=0));rows=[]
  with torch.no_grad():
   for i in range(len(x)):
    f,_,_=self.bulk(x[i],scale=1/12000);res=a[i]-f/self.mass[:,None]+torch.nn.functional.softplus(self.param['log_damping'])*v[i];design=torch.cat([self.weights,torch.ones(len(x[i]),1,device='cuda')],1);coef=torch.linalg.solve(design.T@design+.1*torch.eye(13,device='cuda'),design.T@res);rows.append(coef)
   rows=torch.stack(rows);self.param['body'].copy_(rows[:,-1].mean(0))
   for k in range(8):
    idx=round(k*(len(x)-1)/7);self.param['loads'][k].copy_(rows[max(0,idx-2):min(len(x),idx+3),:12].mean(0))
 def clamp(self):
  with torch.no_grad():
   self.param['log_scale'].clamp_(-4,3);self.param['log_damping'].clamp_(-6,4);self.param['log_stiffness'].clamp_(-10,5);self.param['loads'].clamp_(-200,200);self.param['body'].clamp_(-100,100)
