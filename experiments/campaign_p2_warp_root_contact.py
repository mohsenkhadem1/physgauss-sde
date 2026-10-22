"""Prospective force/energy-only FP64 Warp root-split T6 plane integration.

Fixed64 endpoint-mapped outer Gauss, exact quartic inner integration. This is a
NEW standalone surface operator; no old Duffy/bulk/mass/solver source is edited.
No production Hessian and no trajectory qualification are supplied here.
"""
import numpy as np
import warp as wp
Vec8d=wp.types.vector(length=8,dtype=wp.float64)
Vec6d=wp.types.vector(length=6,dtype=wp.float64)
Vec4d=wp.types.vector(length=4,dtype=wp.float64)

@wp.func
def quadratic_roots(c0:wp.float64,c1:wp.float64,c2:wp.float64):
 roots=wp.vec2d(wp.float64(0.0));n=int(0);scale=wp.max(wp.abs(c0),wp.max(wp.abs(c1),wp.abs(c2)))
 if scale>wp.float64(0.0):
  a=c2/scale;b=c1/scale;c=c0/scale
  if a==wp.float64(0.0):
   if b!=wp.float64(0.0):roots[0]=-c/b;n=1
  else:
   D=b*b-wp.float64(4.0)*a*c
   if D==wp.float64(0.0):roots[0]=-b/(wp.float64(2.0)*a);n=1
   elif D>wp.float64(0.0):
    sign=wp.float64(1.0)
    if b<wp.float64(0.0):sign=wp.float64(-1.0)
    q=wp.float64(-.5)*(b+sign*wp.sqrt(D))
    if q!=wp.float64(0.0):roots[0]=q/a;roots[1]=c/q;n=2
    else:roots[0]=-b/(wp.float64(2.0)*a);n=1
 return roots,n

@wp.func
def add_event(events:Vec8d,n:int,x:wp.float64):
 duplicate=bool(False)
 for i in range(8):
  if i<n and events[i]==x:duplicate=True
 if wp.isfinite(x) and x>wp.float64(0.0) and x<wp.float64(1.0) and not duplicate:
  events[n]=x;n=n+1
 return events,n

@wp.func
def shape6(r:wp.float64,s:wp.float64):
 b=wp.float64(1.0)-r-s
 return Vec6d(b*(wp.float64(2.0)*b-wp.float64(1.0)),r*(wp.float64(2.0)*r-wp.float64(1.0)),s*(wp.float64(2.0)*s-wp.float64(1.0)),wp.float64(4.0)*b*r,wp.float64(4.0)*b*s,wp.float64(4.0)*r*s)

@wp.kernel
def root_surface(u:wp.array2d(dtype=wp.float64),rest:wp.array2d(dtype=wp.float64),faces:wp.array2d(dtype=wp.int64),areas:wp.array(dtype=wp.float64),outer_t:wp.array(dtype=wp.float64),outer_w:wp.array(dtype=wp.float64),kappa:wp.float64,floor:wp.float64,E:wp.array2d(dtype=wp.float64),gradient:wp.array2d(dtype=wp.float64)):
 q,f=wp.tid();z=Vec6d(wp.float64(0.0))
 for nodea in range(6):z[nodea]=rest[faces[f,nodea],2]+u[faces[f,nodea],2]-floor
 c00=z[0];c10=wp.float64(4.0)*z[3]-wp.float64(3.0)*z[0]-z[1];c01=wp.float64(4.0)*z[4]-wp.float64(3.0)*z[0]-z[2]
 c20=wp.float64(2.0)*(z[0]+z[1]-wp.float64(2.0)*z[3]);c02=wp.float64(2.0)*(z[0]+z[2]-wp.float64(2.0)*z[4]);c11=wp.float64(4.0)*(z[0]-z[3]-z[4]+z[5])
 events=Vec8d(wp.float64(0.0));events[1]=wp.float64(1.0);n=int(2)
 for poly in range(3):
  a=c00;b=c10;c=c20
  if poly==1:a=c00+c01+c02;b=c10-c01+c11-wp.float64(2.0)*c02;c=c20-c11+c02
  elif poly==2:a=c01*c01-wp.float64(4.0)*c02*c00;b=wp.float64(2.0)*c01*c11-wp.float64(4.0)*c02*c10;c=c11*c11-wp.float64(4.0)*c02*c20
  roots,count=quadratic_roots(a,b,c)
  for k in range(2):
   if k<count:events,n=add_event(events,n,roots[k])
 for i in range(8):
  for j in range(7):
   if j<n-1 and events[j]>events[j+1]:tmp=events[j];events[j]=events[j+1];events[j+1]=tmp
 energy=wp.float64(0.0);g=Vec6d(wp.float64(0.0));t=outer_t[q];pi=wp.float64(3.1415926535897932384626433832795)
 for segment in range(7):
  if segment<n-1:
   lo=events[segment];hi=events[segment+1];sn=wp.sin(pi*t/wp.float64(2.0));r=lo+(hi-lo)*sn*sn;jac=(hi-lo)*pi/wp.float64(2.0)*wp.sin(pi*t);upper=wp.float64(1.0)-r
   p0=c00+c10*r+c20*r*r;p1=c01+c11*r;p2=c02;cuts=Vec4d(wp.float64(0.0));cuts[1]=upper;nc=int(2);roots,count=quadratic_roots(p0,p1,p2)
   for k in range(2):
    if k<count and roots[k]>wp.float64(0.0) and roots[k]<upper:
     duplicate=bool(False)
     for j in range(4):
      if j<nc and cuts[j]==roots[k]:duplicate=True
     if not duplicate:cuts[nc]=roots[k];nc=nc+1
   for i in range(4):
    for j in range(3):
     if j<nc-1 and cuts[j]>cuts[j+1]:tmp=cuts[j];cuts[j]=cuts[j+1];cuts[j+1]=tmp
   for interval in range(3):
    if interval<nc-1:
     a=cuts[interval];b=cuts[interval+1];mid=(a+b)/wp.float64(2.0)
     if p0+mid*(p1+mid*p2)<wp.float64(0.0):
      for point in range(3):
       x=wp.float64(0.0);w=(wp.float64(8.0)/wp.float64(9.0))
       if point==0:x=wp.float64(-0.77459666924148337704);w=(wp.float64(5.0)/wp.float64(9.0))
       elif point==2:x=wp.float64(0.77459666924148337704);w=(wp.float64(5.0)/wp.float64(9.0))
       s=mid+(b-a)/wp.float64(2.0)*x;h=p0+s*(p1+s*p2);N=shape6(r,s);weight=wp.float64(2.0)*areas[f]*kappa*outer_w[q]*jac*(b-a)/wp.float64(2.0)*w
       energy+=wp.float64(.5)*weight*h*h
       for j in range(6):g[j]+=weight*h*N[j]
 E[q,f]=energy
 for nodea in range(6):wp.atomic_add(gradient,faces[f,nodea],2,g[nodea])

class WarpRootContact:
 """Standalone surface E/g; no material/bulk/J/mass/solver changes."""
 def __init__(self,*,rest,surface_elements,reference_areas,kappa_A=1e9,floor_z=0.,outer_order=64,device='cpu'):
  rest=np.asarray(rest);faces=np.asarray(surface_elements);area=np.asarray(reference_areas)
  if rest.ndim!=2 or rest.shape[1]!=3 or rest.dtype!=np.float64 or faces.ndim!=2 or faces.shape[1]!=6 or not np.issubdtype(faces.dtype,np.integer) or np.any(faces<0) or np.any(faces>=len(rest)) or area.shape!=(len(faces),) or not np.isfinite(rest).all() or not np.isfinite(area).all() or np.any(area<=0):raise ValueError('Finite FP64 T6 surface/reference areas required')
  if not np.isfinite([kappa_A,floor_z]).all() or kappa_A<=0 or type(outer_order)is not int or outer_order!=64:raise ValueError('Explicit fixed64/finite positive same-law parameters required')
  self.device=wp.get_device(device);self.n=len(rest);self.kappa_A=kappa_A;self.floor_z=floor_z;self.order=outer_order
  x,w=np.polynomial.legendre.leggauss(outer_order);self.rest=wp.array(np.ascontiguousarray(rest),dtype=wp.float64,device=self.device);self.faces=wp.array(np.ascontiguousarray(faces),dtype=wp.int64,device=self.device);self.areas=wp.array(np.ascontiguousarray(area),dtype=wp.float64,device=self.device);self.t=wp.array((x+1)/2,dtype=wp.float64,device=self.device);self.w=wp.array(w/2,dtype=wp.float64,device=self.device);self.u=wp.empty((self.n,3),dtype=wp.float64,device=self.device);self.E=wp.empty((outer_order,len(faces)),dtype=wp.float64,device=self.device);self.gradient=wp.empty((self.n,3),dtype=wp.float64,device=self.device)
 def evaluate(self,u):
  if not isinstance(u,np.ndarray) or u.shape!=(self.n,3) or u.dtype!=np.float64 or not np.isfinite(u).all():raise ValueError('Finite FP64 displacement required')
  wp.copy(self.u,wp.array(np.ascontiguousarray(u),dtype=wp.float64,device=self.device));self.gradient.zero_();wp.launch(root_surface,dim=self.E.shape,inputs=[self.u,self.rest,self.faces,self.areas,self.t,self.w,self.kappa_A,self.floor_z,self.E,self.gradient],device=self.device);energy=float(self.E.numpy().sum());g=self.gradient.numpy().copy()
  if not np.isfinite(energy) or not np.isfinite(g).all():raise ValueError('Nonfinite unchanged-law contact integration')
  return dict(contact_energy=energy,gradient=g,outer_order=64,production_Hessian=False,limitations='FixedstateE/ginterfaceonly; no temporal/spatial/hardcontact/trajectory qualification')
