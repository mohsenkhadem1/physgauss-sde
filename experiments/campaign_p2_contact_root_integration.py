"""CPU reference: root-split contact integration on one compatible T6 face.

The physical law is unchanged: .5*kappa*integral(min(z-floor,0)^2)dA_ref.
At fixed r, split s at quadratic height roots; 3-point Gauss integrates the
resulting quartic energy/gradient/Hessian polynomials exactly in real arithmetic.
Outer integration is adaptive, split at edge and discriminant topology events.
Returned error estimates are numerical estimates, never interval certificates.
No GPU, dynamics, floor projection, friction/damping, or law change.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import quad_vec
from numpy.polynomial.legendre import leggauss
INNER_X,INNER_W=leggauss(3)
PILOT_X,PILOT_W=leggauss(16)


def triangle_shapes(r,s):
 r,s=np.broadcast_arrays(r,s);b=np.stack([1-r-s,r,s],axis=-1)
 return np.concatenate([b*(2*b-1),np.stack([4*b[...,0]*b[...,1],4*b[...,0]*b[...,2],4*b[...,1]*b[...,2]],axis=-1)],axis=-1)


def height_coefficients(height6):
 z=np.asarray(height6,dtype=np.longdouble)
 return np.array([z[0],4*z[3]-3*z[0]-z[1],4*z[4]-3*z[0]-z[2],2*(z[0]+z[1]-2*z[3]),4*(z[0]-z[3]-z[4]+z[5]),2*(z[0]+z[2]-2*z[4])],dtype=np.longdouble)


def real_roots(coefficients):
 """Finite real roots, scaling + cancellation-safe quadratic formula.

 Exactly-zero leading coefficients select linear/constant cases. No positive
 active interval is intentionally discarded using an area or force threshold.
 Duplicates are merged only if their represented float64 values are identical.
 """
 c=np.asarray(coefficients,dtype=np.longdouble);scale=max(abs(c))
 if scale==0:return []
 c=c/scale;c0,c1,c2=c
 if c2==0:
  return [] if c1==0 else [float(-c0/c1)]
 D=c1*c1-4*c2*c0
 if D<0:return []
 if D==0:return [float(-c1/(2*c2))]
 root=np.sqrt(D);q=-np.longdouble(.5)*(c1+np.copysign(root,c1))
 values=[q/c2,c0/q] if q!=0 else [-c1/(2*c2)]
 return sorted(set(float(x) for x in values if np.isfinite(x)))


def topology_events(c):
 a,b,d,e,f,g=c
 # z(r,0), z(r,1-r), and s-quadratic discriminant.
 polynomials=[[a,b,e],[a+d+g,b-d+f-2*g,e-f+g],[d*d-4*g*a,2*d*f-4*g*b,f*f-4*g*e]]
 return sorted(set([0.,1.]+[x for poly in polynomials for x in real_roots(poly) if 0<x<1]))


def inner_contact(r,coefficients):
 """Unscaled [integral z-²/2, integral z-N(6), integral activeNN(36), activearea]."""
 a,b,d,e,f,g=coefficients;upper=1-r;out=np.zeros(44)
 if upper<=0:return out
 p=np.asarray([a+b*r+e*r*r,d+f*r,g],dtype=np.longdouble)
 cuts=sorted(set([0.,upper]+[x for x in real_roots(p) if 0<x<upper]))
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  midpoint=np.longdouble(lo)+(np.longdouble(hi)-lo)/2
  if p[0]+midpoint*(p[1]+midpoint*p[2])>=0:continue
  s=(lo+hi)/2+(hi-lo)/2*INNER_X;w=(hi-lo)/2*INNER_W
  # Extended-precision coefficient evaluation reduces cancellation near roots;
  # shape integration and final sums remain ordinary float64.
  z=np.asarray(p[0]+np.asarray(s,dtype=np.longdouble)*(p[1]+np.asarray(s,dtype=np.longdouble)*p[2]),dtype=float);N=triangle_shapes(np.full_like(s,r),s)
  out[0]+=.5*np.dot(w,z*z);out[1:7]+=N.T@(w*z);out[7:43]+=np.einsum('q,qa,qb->ab',w,N,N).ravel();out[43]+=hi-lo
 return out


def integrate_face(height6,reference_area,*,kappa_A=1e9,floor_z=0.,energy_atol_J=1e-12,gradient_atol_N=1e-10,hessian_atol_N_m=1e-7,area_atol_m2=1e-13,rtol=1e-10,maximum_subintervals=4096):
 """Return conservative E/g/fullH plus simultaneous outer error estimates.

 g/H differentiate with respect to the SIX nodal WORLD heights, in their
 compatible vertex01/2-edge01/02/12 order. Actual reference metric=2*area.
 At a whole face exactly at z=floor, E/g are zero and H uses inactive-side
 selection (zero); no unique classical Hessian exists there.
 """
 z=np.asarray(height6)
 pars=[reference_area,kappa_A,floor_z,energy_atol_J,gradient_atol_N,hessian_atol_N_m,area_atol_m2,rtol]
 if z.shape!=(6,) or z.dtype!=np.float64 or not np.isfinite(z).all() or not np.isfinite(pars).all() or min(reference_area,kappa_A,energy_atol_J,gradient_atol_N,hessian_atol_N_m,area_atol_m2)<=0 or rtol<0 or type(maximum_subintervals)is not int or maximum_subintervals<8:raise ValueError('Finite FP64 compatible heights and positive integration settings required')
 c=height_coefficients(z-np.float64(floor_z));events=topology_events(c);physical_scale=np.r_[np.full(43,2*reference_area*kappa_A),2*reference_area]
 def physical(r):return inner_contact(r,c)*physical_scale
 # Deterministic pilot ONLY sets componentwise relative-error units; final
 # acceptance uses supplied absolute+relative tolerances against FINAL values.
 pilot=np.zeros(44)
 for lo,hi in zip(events[:-1],events[1:]):
  for x,w in zip(PILOT_X,PILOT_W):pilot+=(hi-lo)/2*w*physical((lo+hi)/2+(hi-lo)/2*x)
 absolute=np.r_[energy_atol_J,np.full(6,gradient_atol_N),np.full(36,hessian_atol_N_m),area_atol_m2];units=absolute+rtol*abs(pilot)
 values,error,info=quad_vec(lambda r:physical(r)/units,0.,1.,epsabs=.2,epsrel=0.,norm='max',points=events[1:-1],limit=maximum_subintervals,full_output=True)
 final=values*units;estimated_error=float(error)*units;tolerance=absolute+rtol*abs(final);passed=bool(info.success and np.isfinite(final).all() and np.all(estimated_error<=tolerance))
 if not passed:raise RuntimeError('Unchanged contact integration tolerance failed: '+str(dict(status=int(info.status),normalized_error=float(error),maximum_component_error_to_tolerance=float(np.max(estimated_error/tolerance)))))
 return dict(energy_J=float(final[0]),gradient_N=final[1:7].copy(),hessian_N_m=final[7:43].reshape(6,6).copy(),active_reference_area_m2=float(final[43]),estimated_energy_error_J=float(estimated_error[0]),estimated_gradient_errors_N=estimated_error[1:7].copy(),estimated_hessian_errors_N_m=estimated_error[7:43].reshape(6,6).copy(),estimated_active_area_error_m2=float(estimated_error[43]),outer_intervals=int(len(info.intervals)),outer_evaluations=int(info.neval),topology_r_events=events,passed=passed,whole_face_zero_H_selection='inactive-side zero; no unique classicalH atwholefacez=floor' if np.all(z==floor_z) else None,limitations='Root/area/adaptiveerror estimates are floating point, not interval-certified. Same reference-area finite compliant plane law; no hardfloor/CCD/dynamics claim.')
