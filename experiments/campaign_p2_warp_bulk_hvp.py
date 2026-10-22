"""Prospective FP64 elastic-only HVP with on-the-fly compatible T10 gradients.

No solver integration, force/energy/J guard replacement, contact, mass, or
preconditioner change. The caller retains its original evaluate/guard/cache and
passes its accepted local displacement plus a global direction. Storage is
O(E*10 + E*4*3 + Q*4 + N*3), never a Q*E*10*3 gradient tensor.
"""
from __future__ import annotations
import numpy as np
import warp as wp

@wp.func
def edge_pair(a: int):
 i=int(0);j=int(1)
 if a==5:j=2
 elif a==6:j=3
 elif a==7:i=1;j=2
 elif a==8:i=1;j=3
 elif a==9:i=2;j=3
 return i,j

@wp.func
def shape_gradient(a:int,q:int,e:int,bary:wp.array2d(dtype=wp.float64),linear:wp.array3d(dtype=wp.float64)):
 g=wp.vec3d(wp.float64(0.0))
 if a<4:
  for j in range(3):g[j]=(wp.float64(4.0)*bary[q,a]-wp.float64(1.0))*linear[e,a,j]
 else:
  i,k=edge_pair(a)
  for j in range(3):g[j]=wp.float64(4.0)*(bary[q,i]*linear[e,k,j]+bary[q,k]*linear[e,i,j])
 return g

@wp.func
def columns(a:wp.vec3d,b:wp.vec3d,c:wp.vec3d):
 return wp.mat33d(a[0],b[0],c[0],a[1],b[1],c[1],a[2],b[2],c[2])

@wp.kernel
def bulk_hvp(local_u:wp.array3d(dtype=wp.float64),v:wp.array2d(dtype=wp.float64),elements:wp.array2d(dtype=wp.int64),linear:wp.array3d(dtype=wp.float64),bary:wp.array2d(dtype=wp.float64),weights:wp.array(dtype=wp.float64),volumes:wp.array(dtype=wp.float64),mu:wp.float64,kappa:wp.float64,out:wp.array2d(dtype=wp.float64)):
 q,e=wp.tid();G=wp.mat33d(wp.float64(0.0));dF=wp.mat33d(wp.float64(0.0))
 for a in range(10):
  g=shape_gradient(a,q,e,bary,linear);node=elements[e,a]
  for i in range(3):
   for j in range(3):G[i,j]+=local_u[e,a,i]*g[j];dF[i,j]+=v[node,i]*g[j]
 I=wp.mat33d(wp.float64(1.0),wp.float64(0.0),wp.float64(0.0),wp.float64(0.0),wp.float64(1.0),wp.float64(0.0),wp.float64(0.0),wp.float64(0.0),wp.float64(1.0));F=I+G
 f0=wp.vec3d(F[0,0],F[1,0],F[2,0]);f1=wp.vec3d(F[0,1],F[1,1],F[2,1]);f2=wp.vec3d(F[0,2],F[1,2],F[2,2])
 d0=wp.vec3d(dF[0,0],dF[1,0],dF[2,0]);d1=wp.vec3d(dF[0,1],dF[1,1],dF[2,1]);d2=wp.vec3d(dF[0,2],dF[1,2],dF[2,2])
 cof=columns(wp.cross(f1,f2),wp.cross(f2,f0),wp.cross(f0,f1))
 dcof=columns(wp.cross(d1,f2)+wp.cross(f1,d2),wp.cross(d2,f0)+wp.cross(f2,d0),wp.cross(d0,f1)+wp.cross(f0,d1))
 tr=wp.trace(G);jm=tr+wp.float64(.5)*(tr*tr-wp.trace(G*G))+wp.determinant(G)
 dJ=wp.float64(0.0)
 for i in range(3):
  for j in range(3):dJ+=cof[i,j]*dF[i,j]
 # FULL determinant variation: cross-node/cofactor terms must not be omitted.
 dP=mu*dF+kappa*dJ*cof+(kappa*jm-mu)*dcof
 for a in range(10):
  g=shape_gradient(a,q,e,bary,linear);node=elements[e,a]
  for i in range(3):
   val=wp.float64(0.0)
   for j in range(3):val+=dP[i,j]*g[j]
   wp.atomic_add(out,node,i,weights[q]*volumes[e]*val)

class WarpBulkHVP:
 """Elastic HVP primitive only; caller keeps original J/force/final-R checks.

 API uses NumPy copies deliberately. CUDA interoperability and a CG wrapper
 require separate source review/parity/benchmark before any production use.
 """
 def __init__(self,*,elements,linear_gradients,bary,weights,volumes,node_count,young,poisson,device='cpu'):
  if type(node_count) is not int or node_count<10 or not np.isfinite([young,poisson]).all() or young<=0 or not -1<poisson<.5:raise ValueError('Valid finite material and node count required')
  a=dict(elements=np.asarray(elements),linear_gradients=np.asarray(linear_gradients),bary=np.asarray(bary),weights=np.asarray(weights),volumes=np.asarray(volumes));e=len(a['elements']);q=len(a['bary'])
  if e<1 or q<1 or a['elements'].shape!=(e,10) or a['linear_gradients'].shape!=(e,4,3) or a['bary'].shape!=(q,4) or a['weights'].shape!=(q,) or a['volumes'].shape!=(e,):raise ValueError('Compatible T10 geometry/quadrature shapes required')
  if not np.issubdtype(a['elements'].dtype,np.integer) or np.any(a['elements']<0) or np.any(a['elements']>=node_count):raise ValueError('Exact valid global T10 indices required')
  if not all(np.isfinite(x).all() for x in a.values()) or np.any(a['weights']<=0) or np.any(a['volumes']<=0) or np.any(a['bary']<0) or not np.allclose(a['bary'].sum(1),1,rtol=0,atol=2e-14):raise ValueError('Finite positive quadrature and barycentric points required')
  self.device=wp.get_device(device);self.n=node_count;self.e=e;self.q=q
  self.data={k:wp.array(np.ascontiguousarray(x),dtype=wp.int64 if k=='elements' else wp.float64,device=self.device) for k,x in a.items()}
  self.mu=young/(2*(1+poisson));self.kappa=young*poisson/((1+poisson)*(1-2*poisson))+self.mu
  self.local_u=wp.empty((e,10,3),dtype=wp.float64,device=self.device);self.v=wp.empty((node_count,3),dtype=wp.float64,device=self.device);self.output=wp.empty((node_count,3),dtype=wp.float64,device=self.device)
  self.geometry_bytes=sum(x.nbytes for x in a.values())
 def tangent(self,local_u,v):
  for name,value,shape in [('local_u',local_u,(self.e,10,3)),('direction',v,(self.n,3))]:
   if not isinstance(value,np.ndarray) or value.dtype!=np.float64 or value.shape!=shape or not np.isfinite(value).all():raise ValueError('Finite float64 '+name+' shape required')
  wp.copy(self.local_u,wp.array(np.ascontiguousarray(local_u),dtype=wp.float64,device=self.device));wp.copy(self.v,wp.array(np.ascontiguousarray(v),dtype=wp.float64,device=self.device));self.output.zero_();a=self.data
  wp.launch(bulk_hvp,dim=(self.q,self.e),inputs=[self.local_u,self.v,a['elements'],a['linear_gradients'],a['bary'],a['weights'],a['volumes'],self.mu,self.kappa,self.output],device=self.device)
  result=self.output.numpy().copy()
  if not np.isfinite(result).all():raise ValueError('Nonfinite full elastic HVP')
  return result
