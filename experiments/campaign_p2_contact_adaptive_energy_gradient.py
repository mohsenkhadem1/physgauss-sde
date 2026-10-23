"""Strict adaptive E/g-only plane reference; no Hessian control or claim.

Same frozen root-split inner polynomial and physical reference-area penalty.
Frozen inner evaluation may form H entries, which are discarded here.
Removing unrelated H components from the adaptive norm changes only this NEW
reference's scope, not its E/g tolerances or any production force operator.
"""
import numpy as np
from scipy.integrate import quad_vec
from experiments.campaign_p2_contact_root_integration import height_coefficients,topology_events,inner_contact,PILOT_X,PILOT_W

def integrate_face_energy_gradient(height6,reference_area,*,kappa_A=1e9,floor_z=0.,energy_atol_J=1e-12,gradient_atol_N=1e-10,rtol=1e-10,maximum_subintervals=4096):
 z=np.asarray(height6)
 if z.shape!=(6,) or z.dtype!=np.float64 or not np.isfinite(z).all() or not np.isfinite([reference_area,kappa_A,floor_z,energy_atol_J,gradient_atol_N,rtol]).all() or min(reference_area,kappa_A,energy_atol_J,gradient_atol_N)<=0 or rtol<0 or type(maximum_subintervals)is not int or maximum_subintervals<8:raise ValueError('Finite FP64 heights and original positive E/g integration settings required')
 c=height_coefficients(z-np.float64(floor_z));events=topology_events(c);scale=2*reference_area*kappa_A
 def physical(r):return inner_contact(r,c)[:7]*scale
 pilot=np.zeros(7)
 for lo,hi in zip(events[:-1],events[1:]):
  for x,w in zip(PILOT_X,PILOT_W):pilot+=(hi-lo)/2*w*physical((lo+hi)/2+(hi-lo)/2*x)
 absolute=np.r_[energy_atol_J,np.full(6,gradient_atol_N)];units=absolute+rtol*abs(pilot)
 values,error,info=quad_vec(lambda r:physical(r)/units,0.,1.,epsabs=.2,epsrel=0.,norm='max',points=events[1:-1],limit=maximum_subintervals,full_output=True)
 final=values*units;estimated=float(error)*units;tolerance=absolute+rtol*abs(final)
 if not (info.success and int(info.status)==0 and np.isfinite(final).all() and np.isfinite(estimated).all() and np.all(estimated<=tolerance)):raise RuntimeError('Strict adaptive E/g reference failed: '+str(dict(status=int(info.status),normalized_error=float(error),maximum_component_error_to_tolerance=float(np.max(estimated/tolerance)))))
 return dict(energy_J=float(final[0]),gradient_N=final[1:].copy(),estimated_energy_error_J=float(estimated[0]),estimated_gradient_errors_N=estimated[1:].copy(),outer_intervals=int(len(info.intervals)),outer_evaluations=int(info.neval),passed=True,Hessian_computed_or_qualified=False,scope='E/g only; no status2 waiver; floating point adaptive error estimates, not interval certificate')
