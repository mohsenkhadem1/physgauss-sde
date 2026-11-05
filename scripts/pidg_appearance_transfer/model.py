"""Transferred fitted appearance, bound to solved mechanics at one fitting frame."""
import os,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from scipy.spatial import cKDTree
from scripts.pidg_mechanics.fit_mechanics import payload
from scripts.real_transfer_v19.neural_fem import NeuralFEM
from scripts.mechanics_gaussian_bridge.bridge import render
from scripts.mechanics_gaussian_bridge.implicit_boundary import BoundaryEquilibrium
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/os.environ.get('PIDG_TRANSFER_ROOT','runs/pidg-appearance-transfer-v01')

def polar(F):
 R=F
 for _ in range(12):R=.5*(R+torch.linalg.inv(R).transpose(-1,-2))
 return R

def rotation(v):
 theta=torch.linalg.vector_norm(v);x,y,z=v;zero=v.new_zeros(());K=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)
 return torch.eye(3,device=v.device,dtype=v.dtype)+torch.sinc(theta/torch.pi)*K+.5*torch.sinc(theta/(2*torch.pi))**2*(K@K)

class Transfer(torch.nn.Module):
 active_sh_degree=3;max_sh_degree=3
 def __init__(self):
  super().__init__();self.source=torch.load(BASE/'transferred-appearance.pt',weights_only=False);g=np.load(ROOT/'runs/pidg-mechanics-v04/training-geometry.npz');cache=np.load(BASE/'physical-cache.npz');self.models=[];self.binding=[];self.cache={};self.reference=int(self.source['reference_frame']);xyz=self.source['xyz'].numpy();cov=self.source['covariance'].numpy();ref=cache['nodes'][self.reference];self.register_buffer('nodes',torch.tensor(cache['nodes'],device='cuda',dtype=torch.double));self.register_buffer('targets',torch.tensor(cache['targets'],device='cuda',dtype=torch.double));self.register_buffer('theta',torch.tensor(cache['theta'],device='cuda',dtype=torch.double))
  centers=[]
  for b in range(6):
   m=NeuralFEM(payload(g,b,np.arange(150)/149),device='cuda:0',materials=1);self.models.append(m);centers.append(ref[b].mean(0)*m.scale+m.origin)
  # Canonical 3-D nearest-body identity; retain every Gaussian and original order.
  label=np.argmin(np.sum(((xyz[:,None]-np.array(centers)[None])/g['radii'][None,:,None])**2,axis=-1),axis=1)
  if (BASE/'body-labels.npy').exists():label=np.load(BASE/'body-labels.npy')
  self.register_buffer('body_ids',torch.tensor(label,device='cuda'))
  for b,m in enumerate(self.models):
   selected=np.where(label==b)[0];v=ref[b][m.elements.cpu().numpy()];dm=np.stack([v[:,i]-v[:,0] for i in [1,2,3]],-1);inv=np.linalg.inv(dm);grad=np.concatenate([-inv.sum(1,keepdims=True),inv],1);points=(xyz[selected]-m.origin)/m.scale;elem=cKDTree(v.mean(1)).query(points)[1];bar=np.einsum('nij,nj->ni',inv[elem],points-v[elem,0]);bar=np.c_[1-bar.sum(1),bar]
   self.binding.append(dict(indices=torch.tensor(selected,device='cuda'),element=torch.tensor(elem,device='cuda'),weights=m.t(bar),grad=m.t(grad),cov=m.t(cov[selected]),radius=float(g['radii'][b])))
  self.features=torch.nn.Parameter(self.source['features'].cuda().clone());self.opacity_raw=torch.nn.Parameter(torch.logit(self.source['opacity'].cuda().clamp(1e-6,1-1e-6)));self.offset=torch.nn.Parameter(torch.zeros(len(xyz),3,device='cuda'));self.scale=torch.nn.Parameter(torch.zeros(len(xyz),1,device='cuda'));self.controls=torch.nn.Parameter(torch.zeros(12,6,7,device='cuda',dtype=torch.double));self.patch_controls=torch.nn.Parameter(torch.zeros(12,6,6,3,device='cuda',dtype=torch.double))
 @property
 def get_features(self):return self.features
 @property
 def get_opacity(self):return self.opacity_raw.sigmoid()
 def solve(self,frame):
  result=[];p=frame/149*11;i=min(int(p),10)
  for b,m in enumerate(self.models):
   q=(1-(p-i))*self.controls[i,b]+(p-i)*self.controls[i+1,b];original=self.targets[frame,b];center=original.mean((0,1),keepdim=True);R=rotation(.8*q[4:].tanh());target=(original-center)@R.T*torch.exp(.3*q[3].tanh())+center+.5*q[:3].tanh()[None,None]
   patch=(1-(p-i))*self.patch_controls[i,b]+(p-i)*self.patch_controls[i+1,b];target=target+.3*patch.tanh()[:,None,:]
   targets=m.targets.clone();targets[frame]=target;nodes=BoundaryEquilibrium.apply(self.theta[b],targets,m,frame,target.mean(0).detach());result.append(nodes)
  return torch.stack(result)
 def raw(self,frame,nodes=None):
  if nodes is None and frame in self.cache:return self.cache[frame]
  fixed=nodes is None;nodes=self.nodes[frame] if fixed else nodes;n=len(self.features);centers=nodes.new_zeros(n,3);cov=nodes.new_zeros(n,3,3);rot=nodes.new_zeros(n,3,3);gradient=nodes.new_zeros(n,3,3)
  for b,(m,bind) in enumerate(zip(self.models,self.binding)):
   x=nodes[b][m.elements];F=torch.einsum('eai,eaJ->eiJ',x,bind['grad']);assert torch.linalg.det(F).min()>0
   Fe=F[bind['element']];Re=polar(F)[bind['element']];pos=torch.einsum('na,nai->ni',bind['weights'],x[bind['element']])*m.scale+m.t(m.origin)
   centers=centers.index_copy(0,bind['indices'],pos);cov=cov.index_copy(0,bind['indices'],Fe@bind['cov']@Fe.transpose(-1,-2));rot=rot.index_copy(0,bind['indices'],Re);gradient=gradient.index_copy(0,bind['indices'],Fe*bind['radius'])
  state={k:v.float() for k,v in dict(centers=centers,covariance=cov,rotation=rot,gradient=gradient).items()}
  if fixed:self.cache[frame]={k:v.detach() for k,v in state.items()}
  return state
 def state(self,frame,live=False):
  state=self.raw(frame,self.solve(frame) if live else None)
  if os.environ.get('TRANSFER_RADIANCE_WORLD')=='1':state=dict(state,rotation=torch.eye(3,device='cuda').expand(len(self.features),3,3))
  return dict(centers=state['centers']+.03*torch.einsum('nij,nj->ni',state['gradient'],self.offset.tanh()),covariance=state['covariance']*torch.exp(.4*self.scale.tanh())[:,:,None],rotation=state['rotation'])
 def image(self,camera,frame,live=False):return render(camera,self,self.state(frame,live),torch.zeros(3,device='cuda'))['render']
 def parameters_only(self):return {k:getattr(self,k).detach().cpu().clone() for k in ['features','opacity_raw','offset','scale','controls','patch_controls']}
 def restore(self,params):
  with torch.no_grad():
   for k,v in params.items():getattr(self,k).copy_(v.to(getattr(self,k)))
