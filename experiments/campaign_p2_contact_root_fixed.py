"""CPU prospective fixed outer rule for the unchanged root-split plane law.

Each topology interval uses r=lo+(hi-lo)*sin²(pi*t/2), INCLUDING its Jacobian.
Inner polynomial integration/root splitting is shared with the adaptive CPU
reference; outer Gauss-Legendre rule is numerically independent. No GPU calls.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from experiments.campaign_p2_contact_root_integration import height_coefficients,topology_events,inner_contact

def integrate_face_fixed(height6,reference_area,*,kappa_A=1e9,floor_z=0.,outer_order=16):
 z=np.asarray(height6)
 if z.shape!=(6,) or z.dtype!=np.float64 or not np.isfinite(z).all() or not np.isfinite([reference_area,kappa_A,floor_z]).all() or min(reference_area,kappa_A)<=0 or type(outer_order)is not int or outer_order<2:raise ValueError('Finite FP64 heights/positive area/kappa/integer rule required')
 c=height_coefficients(z-floor_z);events=topology_events(c);x,w=leggauss(outer_order);t=(x+1)/2;w=w/2;theta=np.pi*t/2;values=np.zeros(44)
 for lo,hi in zip(events[:-1],events[1:]):
  r=lo+(hi-lo)*np.sin(theta)**2;jac=(hi-lo)*np.pi/2*np.sin(np.pi*t)
  for rr,ww,jj in zip(r,w,jac):values+=ww*jj*inner_contact(float(rr),c)
 values*=np.r_[np.full(43,2*reference_area*kappa_A),2*reference_area]
 if not np.isfinite(values).all():raise ValueError('Nonfinite same-law fixedouter integration')
 return dict(energy_J=float(values[0]),gradient_N=values[1:7],hessian_N_m=values[7:43].reshape(6,6),active_reference_area_m2=float(values[43]),outer_order=outer_order,topology_r_events=events,outer_evaluations=outer_order*(len(events)-1),limitations='No errorcertificate; compare with refinedfixedrule and adaptiveCPUreference before use. E/g/H integrate samecontinuous law; atfiniteorder topology-dependent quadrature does not imply exact derivative of itsapproximateE.')
