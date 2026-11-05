"""Fixed arbitrary-mesh material binding for the paper's NeuralFEM adapter."""
import numpy as np
import torch
from scipy.spatial import cKDTree

class TetraBinding:
 def __init__(self, model, centers, covariance):
  nodes=model.rest.detach().cpu().numpy();elements=model.elements.detach().cpu().numpy()
  points=(np.asarray(centers)-model.origin)/model.scale
  # Fixed nearest-element association. Affine extrapolation is explicit outside mesh.
  ids=cKDTree(nodes[elements].mean(1)).query(points)[1]
  v=nodes[elements[ids]];dm=np.stack([v[:,i]-v[:,0] for i in [1,2,3]],-1)
  bary=np.einsum('nij,nj->ni',np.linalg.inv(dm),points-v[:,0]);bary=np.c_[1-bary.sum(1),bary]
  self.ids=model.elements[torch.as_tensor(ids,device=model.device)];self.N=model.t(bary);self.grad=model.grad[torch.as_tensor(ids,device=model.device)];self.cov=model.t(covariance);self.scale=model.scale;self.origin=model.t(model.origin)
  self.extrapolated_fraction=float(np.mean((bary<0).any(1)))
 def transport(self,nodes):
  x=nodes[self.ids];F=torch.einsum('pai,paJ->piJ',x,self.grad);J=torch.linalg.det(F)
  if not torch.isfinite(F).all() or (J<=0).any():raise ValueError('Invalid tetrahedral transport')
  R=F
  for _ in range(12):R=.5*(R+torch.linalg.inv(R).transpose(-1,-2))
  if not torch.allclose(R.transpose(-1,-2)@R,torch.eye(3,device=nodes.device,dtype=nodes.dtype).expand_as(R),atol=1e-5,rtol=1e-5):raise ValueError('Polar iteration failed')
  return dict(centers=torch.einsum('pa,pai->pi',self.N,x)*self.scale+self.origin,covariance=F@self.cov@F.transpose(-1,-2),rotation=R,determinant=J)
