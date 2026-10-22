"""Exact scalar reference for potential-only plane contact under physical BDF2.

This pure module proves the discrete elimination, not general frictional contact.
The plane is static; normal potential is .5*k*max(r-z,0)^2; kd=mu=0.
Physical velocity is independently obtained from momentum, with native positions
retained by callers. Momentum residual constructed from it is not independent.
"""
from __future__ import annotations
import numpy as np

def scalar_step(z,v,h,*,mass,k,radius=0.,gravity=-9.81,oldz=None,oldv=None):
 if not np.isfinite([z,v,h,mass,k,radius,gravity]).all() or min(h,mass,k)<=0:raise ValueError('Finite positive step/mass/stiffness required')
 if (oldz is None)!=(oldv is None):raise ValueError('Paired BDF2 history required')
 if oldz is None:
  effective=h;helper=v;vmemory=v
 else:
  if not np.isfinite([oldz,oldv]).all():raise ValueError('Finite history required')
  effective=2*h/3;helper=(z-oldz)/(2*h)+(4*v-oldv)/3;vmemory=(4*v-oldv)/3
 y=z+effective*helper+effective**2*gravity
 q=y if y>=radius else (mass*y+effective**2*k*radius)/(mass+effective**2*k)
 contact=k*max(radius-q,0)
 velocity=vmemory+effective*(gravity+contact/mass)
 return q,velocity

def discrete_momentum_impulse(velocities,mass):
 """BE-start+BDF2 telescoped h*D2v, not ordinary rectangle physical impulse."""
 v=np.asarray(velocities,float)
 if v.ndim<1 or len(v)<2 or not np.isfinite(v).all() or mass<=0:raise ValueError('Finite full velocity history required')
 if len(v)==2:return mass*(v[1]-v[0])
 return mass*(3*v[-1]-v[-2]-v[1]-v[0])/2
