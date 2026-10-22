"""Compatible T10 reference-area compliant-plane contact, float64.

Positive Duffy weights integrate the true six-node quadratic boundary trace.
Uc=.5*kappa_A*integral_boundary max(floor_z-z(X),0)^2 dA_reference.
The plane is static, frictionless, undamped; kappa_A has units N/m^3.
This is finite-compliance contact, not exact nonpenetration or CCD. Parent
elasticity, full consistent mass, constraints and Newton-CG objective are intact.
"""
from __future__ import annotations
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from experiments.campaign_p2_torch import TorchP2
from experiments.campaign_static_p2_block import EDGES

TRI_EDGES=((0,1),(0,2),(1,2))

def triangle_rule(order):
 if type(order) is not int or not 2<=order<=32:raise ValueError('Positive triangle order2..32 required')
 t,w=leggauss(order);t=(t+1)/2;w=w/2
 a,b=np.meshgrid(t,t,indexing='ij');wa,wb=np.meshgrid(w,w,indexing='ij')
 r=a.ravel();s=((1-a)*b).ravel();bary=np.stack([1-r-s,r,s],-1)
 N=np.concatenate([bary*(2*bary-1),np.stack([4*bary[:,i]*bary[:,j] for i,j in TRI_EDGES],-1)],-1)
 weights=(2*wa*wb*(1-a)).ravel()
 return bary,N,weights

def boundary_trace(model):
 """Return oriented boundaryface6 global IDs and positive reference areas."""
 rest=np.asarray(model.rest,float);faces=np.asarray(model.faces,int);elements=np.asarray(model.elements,int)
 edge={}
 for tet in elements:
  for k,(i,j) in enumerate(EDGES):
   key=tuple(sorted((int(tet[i]),int(tet[j]))));mid=int(tet[4+k])
   if key in edge and edge[key]!=mid:raise ValueError('Incompatible shared P2 edge')
   edge[key]=mid
 if len(np.unique(np.sort(faces,1),axis=0))!=len(faces):raise ValueError('Duplicate boundaryface')
 six=np.array([list(map(int,f))+[edge[tuple(sorted((int(f[i]),int(f[j]))))] for i,j in TRI_EDGES] for f in faces],dtype=np.int64)
 x=rest[six[:,:3]];areas=.5*np.linalg.norm(np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]),axis=1)
 if np.any(areas<=0) or not np.isfinite(areas).all():raise ValueError('Positive finite boundaryface areas required')
 for k,(i,j) in enumerate(TRI_EDGES):
  if not np.allclose(rest[six[:,3+k]],(x[:,i]+x[:,j])/2,atol=1e-14,rtol=1e-13):raise ValueError('Affine rest geometry required')
 return six,areas

def quadratic_face_minimum(values):
 """Analytic global minimum of a scalar P2 triangle, evaluated in float64.

 Candidate completeness: three vertices, convex edge stationary points and
 positive-definite interior stationary point. Degenerate interior minima extend
 to an edge. This is not an outward-rounded interval certificate.
 Input order: vertices0/1/2, midpoints01/02/12. Returns(face minima, arg bary).
 """
 z=np.asarray(values,dtype=float)
 if z.ndim!=2 or z.shape[1]!=6 or not np.isfinite(z).all():raise ValueError('Finite face6 values required')
 v0,v1,v2,e01,e02,e12=z.T
 a=2*(v0+v1)-4*e01;b=2*(v0+v2)-4*e02;c=4*(v0-e01-e02+e12);d=4*e01-3*v0-v1;e=4*e02-3*v0-v2;f=v0
 vals=[v0,v1,v2];barys=[np.broadcast_to(x,(len(z),3)).copy() for x in np.eye(3)]
 def add(u,v,valid):
  value=a*u*u+b*v*v+c*u*v+d*u+e*v+f
  vals.append(np.where(valid,value,np.inf));barys.append(np.stack([1-u-v,u,v],-1))
 with np.errstate(divide='ignore',invalid='ignore'):
  u=np.divide(-d,2*a,out=np.zeros_like(a),where=a!=0);add(u,np.zeros_like(u),(a>0)&(u>0)&(u<1))
  v=np.divide(-e,2*b,out=np.zeros_like(b),where=b!=0);add(np.zeros_like(v),v,(b>0)&(v>0)&(v<1))
  aa=a+b-c;dd=c+d-2*b-e;u=np.divide(-dd,2*aa,out=np.zeros_like(aa),where=aa!=0);add(u,1-u,(aa>0)&(u>0)&(u<1))
  det=4*a*b-c*c;u=np.divide(c*e-2*b*d,det,out=np.zeros_like(det),where=det!=0);v=np.divide(c*d-2*a*e,det,out=np.zeros_like(det),where=det!=0);add(u,v,(a>0)&(det>0)&(u>0)&(v>0)&(u+v<1))
 V=np.stack(vals,1);B=np.stack(barys,1);idx=np.argmin(V,axis=1);return V[np.arange(len(z)),idx],B[np.arange(len(z)),idx]

class PlaneContactP2(TorchP2):
 def __init__(self,model,device='cpu',*,kappa_A=1e9,floor_z=0.,surface_order=6):
  if not np.isfinite([kappa_A,floor_z]).all() or kappa_A<=0:raise ValueError('Positive finite compliant-plane stiffness required')
  super().__init__(model,device)
  six,areas=boundary_trace(model);_,N,w=triangle_rule(surface_order)
  tensor=lambda a,dtype=self.dtype:torch.as_tensor(np.asarray(a).copy(),dtype=dtype,device=self.device)
  self.surface_elements=tensor(six,torch.int64);self.surface_shapes=tensor(N);self.surface_weights=tensor(w[:,None]*areas[None]);self.reference_surface_areas=tensor(areas)
  self.kappa_A=float(kappa_A);self.floor_z=float(floor_z);self.surface_order=surface_order

 def scatter_surface(self,local):
  out=torch.zeros_like(self.rest);return out.index_add_(0,self.surface_elements.reshape(-1),local.reshape(-1,3))

 def evaluate(self,u):
  cache=super().evaluate(u)
  gap=torch.einsum('qa,fa->qf',self.surface_shapes,(self.rest+u)[self.surface_elements,2])-self.floor_z
  negative=gap.clamp_max(0.);active=gap<0
  U=.5*self.kappa_A*(self.surface_weights*negative.square()).sum()
  gz=self.kappa_A*torch.einsum('qf,qa->fa',self.surface_weights*negative,self.surface_shapes)
  local=torch.zeros((*gz.shape,3),dtype=self.dtype,device=self.device);local[...,2]=gz
  cache.update(energy=cache['energy']+U,gradient=cache['gradient']+self.scatter_surface(local),contact_energy=U,contact_active=active,contact_gap=gap)
  return cache

 def tangent(self,cache,v):
  out=super().tangent(cache,v)
  dz=torch.einsum('qa,fa->qf',self.surface_shapes,v[self.surface_elements,2])
  hz=self.kappa_A*torch.einsum('qf,qa->fa',self.surface_weights*cache['contact_active']*dz,self.surface_shapes)
  local=torch.zeros((*hz.shape,3),dtype=self.dtype,device=self.device);local[...,2]=hz
  return out+self.scatter_surface(local)

 def diagonal(self,cache):
  out=super().diagonal(cache)
  hz=self.kappa_A*torch.einsum('qf,qa->fa',self.surface_weights*cache['contact_active'],self.surface_shapes.square())
  local=torch.zeros((*hz.shape,3),dtype=self.dtype,device=self.device);local[...,2]=hz
  return out+self.scatter_surface(local)

 def contact_diagnostics(self,u,cache=None):
  cache=self.evaluate(u) if cache is None else cache
  z=(self.rest+u)[self.surface_elements,2].detach().cpu().numpy()
  minimum,bary=quadratic_face_minimum(z)
  force=self.kappa_A*(self.surface_weights*(-cache['contact_gap']).clamp_min(0)).sum()
  return dict(contact_energy_J=float(cache['contact_energy']),contact_normal_resultant_N=float(force),maximum_quadrature_penetration_m=max(0.,-float(cache['contact_gap'].min())),maximum_analytic_P2_face_penetration_m=max(0.,self.floor_z-float(minimum.min())),minimum_face_index=int(np.argmin(minimum)),minimum_face_barycentric=bary[np.argmin(minimum)].tolist(),reference_surface_area_m2=float(self.reference_surface_areas.sum()),active_quadrature_count=int(cache['contact_active'].sum()),quadrature_count=cache['contact_active'].numel(),law='Finite reference-area compliantplane; staticfloor; mu=kd=0',minimum_semantics='Analytic P2 simplex minimization in float64, not intervalarithmeticcertificate')
