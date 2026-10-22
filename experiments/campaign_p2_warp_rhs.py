"""Independent FP64 force-only Warp prototype; no integrator or mass change.

Arrays are exported NumPy snapshots of the existing compatible T10 model.
Bulk W/P use the same cancellation-stable G=F-I polynomial as TorchP2.
All existing bulk and check Jacobian points and all six contact nodes remain.
Atomic assembly can reorder FP64 sums; parity must precede any trajectory use.
This prototype neither imports Torch/Newton nor mutates their runtimes.
"""
from __future__ import annotations
import numpy as np
import warp as wp

@wp.func
def material(G:wp.mat33d,mu:wp.float64,kappa:wp.float64):
 I=wp.mat33d(wp.float64(1.0),wp.float64(0.0),wp.float64(0.0),wp.float64(0.0),wp.float64(1.0),wp.float64(0.0),wp.float64(0.0),wp.float64(0.0),wp.float64(1.0))
 F=I+G
 tr=wp.trace(G);second=wp.float64(0.5)*(tr*tr-wp.trace(G*G));detG=wp.determinant(G);jm=tr+second+detG
 c0=wp.cross(wp.vec3d(F[0,1],F[1,1],F[2,1]),wp.vec3d(F[0,2],F[1,2],F[2,2]))
 c1=wp.cross(wp.vec3d(F[0,2],F[1,2],F[2,2]),wp.vec3d(F[0,0],F[1,0],F[2,0]))
 c2=wp.cross(wp.vec3d(F[0,0],F[1,0],F[2,0]),wp.vec3d(F[0,1],F[1,1],F[2,1]))
 cof=wp.mat33d(c0[0],c1[0],c2[0],c0[1],c1[1],c2[1],c0[2],c1[2],c2[2])
 norm=wp.float64(0.0)
 for i in range(3):
  for j in range(3):norm+=G[i,j]*G[i,j]
 W=mu*(wp.float64(.5)*norm-second-detG)+wp.float64(.5)*kappa*jm*jm
 P=mu*G+(kappa*jm-mu)*(cof-I)+kappa*jm*I
 return W,P,wp.float64(1.0)+jm

@wp.kernel
def bulk(u:wp.array2d(dtype=wp.float64),elements:wp.array2d(dtype=wp.int64),grad:wp.array4d(dtype=wp.float64),weights:wp.array2d(dtype=wp.float64),mu:wp.float64,kappa:wp.float64,E:wp.array2d(dtype=wp.float64),J:wp.array2d(dtype=wp.float64),g:wp.array2d(dtype=wp.float64)):
 q,e=wp.tid();G=wp.mat33d(wp.float64(0.0))
 for a in range(10):
  n=elements[e,a]
  for i in range(3):
   for j in range(3):G[i,j]+=u[n,i]*grad[q,e,a,j]
 W,P,jac=material(G,mu,kappa);E[q,e]=weights[q,e]*W;J[q,e]=jac
 for a in range(10):
  n=elements[e,a]
  for i in range(3):
   val=wp.float64(0.0)
   for j in range(3):val+=P[i,j]*grad[q,e,a,j]
   wp.atomic_add(g,n,i,weights[q,e]*val)

@wp.kernel
def check_jacobian(u:wp.array2d(dtype=wp.float64),elements:wp.array2d(dtype=wp.int64),grad:wp.array4d(dtype=wp.float64),J:wp.array2d(dtype=wp.float64)):
 q,e=wp.tid();F=wp.mat33d(wp.float64(1.0),wp.float64(0.0),wp.float64(0.0),wp.float64(0.0),wp.float64(1.0),wp.float64(0.0),wp.float64(0.0),wp.float64(0.0),wp.float64(1.0))
 for a in range(10):
  n=elements[e,a]
  for i in range(3):
   for j in range(3):F[i,j]+=u[n,i]*grad[q,e,a,j]
 J[q,e]=wp.determinant(F)

@wp.kernel
def surface(u:wp.array2d(dtype=wp.float64),rest:wp.array2d(dtype=wp.float64),faces:wp.array2d(dtype=wp.int64),N:wp.array2d(dtype=wp.float64),weights:wp.array2d(dtype=wp.float64),kappa:wp.float64,floor:wp.float64,E:wp.array2d(dtype=wp.float64),g:wp.array2d(dtype=wp.float64)):
 q,f=wp.tid();gap=-floor
 for a in range(6):
  n=faces[f,a];gap+=N[q,a]*(rest[n,2]+u[n,2])
 negative=wp.min(gap,wp.float64(0.0));E[q,f]=wp.float64(.5)*kappa*weights[q,f]*negative*negative
 for a in range(6):wp.atomic_add(g,faces[f,a],2,kappa*weights[q,f]*negative*N[q,a])

class WarpP2Force:
 """Force/energy/J only. Caller retains exact unchanged full mass solve."""
 REQUIRED=('rest','elements','grad','check_grad','weights','surface_elements','surface_shapes','surface_weights')
 def __init__(self,arrays,*,young,poisson,kappa_A=1e9,floor_z=0.,minimum_J=.1,device='cpu'):
  if not np.isfinite([young,poisson,kappa_A,floor_z,minimum_J]).all() or young<=0 or not -1<poisson<.5 or kappa_A<=0:raise ValueError('Finite positive material/contact inputs required')
  self.device=wp.get_device(device);self.arrays={k:np.ascontiguousarray(arrays[k]) for k in self.REQUIRED};a=self.arrays;n=len(a['rest']);e=len(a['elements']);q=len(a['grad']);f=len(a['surface_elements']);qs=len(a['surface_shapes'])
  if a['rest'].shape!=(n,3) or a['elements'].shape!=(e,10) or a['grad'].shape!=(q,e,10,3) or a['check_grad'].shape[1:]!=(e,10,3) or a['weights'].shape!=(q,e) or a['surface_elements'].shape!=(f,6) or a['surface_shapes'].shape!=(qs,6) or a['surface_weights'].shape!=(qs,f):raise ValueError('Exact compatible T10/face6 shapes required')
  for k in ('elements','surface_elements'):
   if not np.issubdtype(a[k].dtype,np.integer) or np.any(a[k]<0) or np.any(a[k]>=n):raise ValueError('Valid exact global node indices required')
  for k,v in a.items():
   if not np.isfinite(v).all():raise ValueError('Finite immutable inputs required')
  if np.any(a['weights']<=0) or np.any(a['surface_weights']<=0):raise ValueError('Positive reference quadrature weights required')
  self.data={k:wp.array(v,dtype=wp.int64 if k in ('elements','surface_elements') else wp.float64,device=self.device) for k,v in a.items()};self.n=n;self.mu=young/(2*(1+poisson));self.kappa=young*poisson/((1+poisson)*(1-2*poisson))+self.mu;self.kappa_A=kappa_A;self.floor_z=floor_z;self.minimum_J=minimum_J
  self.u=wp.empty((n,3),dtype=wp.float64,device=self.device);self.gradient=wp.zeros((n,3),dtype=wp.float64,device=self.device);self.bulk_E=wp.empty((q,e),dtype=wp.float64,device=self.device);self.J=wp.empty((q,e),dtype=wp.float64,device=self.device);self.Jcheck=wp.empty(a['check_grad'].shape[:2],dtype=wp.float64,device=self.device);self.contact_E=wp.empty((qs,f),dtype=wp.float64,device=self.device)
 def evaluate(self,u):
  u=np.asarray(u)
  if u.shape!=(self.n,3) or u.dtype!=np.float64 or not np.isfinite(u).all():raise ValueError('Finite float64 displacement required')
  wp.copy(self.u,wp.array(np.ascontiguousarray(u),dtype=wp.float64,device=self.device));self.gradient.zero_();a=self.data
  wp.launch(bulk,dim=self.bulk_E.shape,inputs=[self.u,a['elements'],a['grad'],a['weights'],self.mu,self.kappa,self.bulk_E,self.J,self.gradient],device=self.device)
  wp.launch(check_jacobian,dim=self.Jcheck.shape,inputs=[self.u,a['elements'],a['check_grad'],self.Jcheck],device=self.device)
  wp.launch(surface,dim=self.contact_E.shape,inputs=[self.u,a['rest'],a['surface_elements'],a['surface_shapes'],a['surface_weights'],self.kappa_A,self.floor_z,self.contact_E,self.gradient],device=self.device)
  J=self.J.numpy();Jc=self.Jcheck.numpy();minimum=float(min(J.min(),Jc.min()))
  if not np.isfinite(J).all() or not np.isfinite(Jc).all() or minimum<=self.minimum_J:raise ValueError('Positive Jacobian guard failed; no repair')
  UE=float(self.bulk_E.numpy().sum());UC=float(self.contact_E.numpy().sum());g=self.gradient.numpy().copy()
  if not np.isfinite(g).all() or not np.isfinite([UE,UC]).all():raise ValueError('Nonfinite original force/energy')
  return dict(energy=UE+UC,elastic_energy=UE,contact_energy=UC,gradient=g,minimum_J=minimum)
