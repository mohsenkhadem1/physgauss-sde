import numpy as np
from experiments.campaign_newton_contact_bdf2 import scalar_step,discrete_momentum_impulse

def test_exact_momentum_kinematic_and_positional_objective():
 z=.005;v=-.4;oldz=.0053;oldv=-.3;h=.001;m=.01;k=10000.;r=.005;g=-9.81
 for history in [{},{'oldz':oldz,'oldv':oldv}]:
  q,w=scalar_step(z,v,h,mass=m,k=k,radius=r,gravity=g,**history)
  e=h if not history else 2*h/3
  helper=v if not history else (z-oldz)/(2*h)+(4*v-oldv)/3
  y=z+e*helper+e*e*g
  np.testing.assert_allclose(m*(q-y)/e**2,k*max(r-q,0),atol=2e-14)
  if history:
   np.testing.assert_allclose((3*q-4*z+oldz)/(2*h),w,atol=2e-15)
   acceleration=(3*w-4*v+oldv)/(2*h)
  else:
   np.testing.assert_allclose((q-z)/h,w,atol=2e-15)
   acceleration=(w-v)/h
  np.testing.assert_allclose(m*acceleration,m*g+k*max(r-q,0),atol=2e-14)

def test_exact_discrete_contact_impulse_and_gravity_equilibrium():
 z=.05;v=0.;zs=[z];vs=[v];forces=[];h=1/960;m=.01;k=10000.
 for i in range(192):
  args={} if i==0 else dict(oldz=zs[-2],oldv=vs[-2])
  z,v=scalar_step(z,v,h,mass=m,k=k,**args);zs.append(z);vs.append(v);forces.append(m*(-9.81)+k*max(-z,0))
 np.testing.assert_allclose(discrete_momentum_impulse(vs,m),h*sum(forces),atol=1e-13)
 q,w=scalar_step(-m*9.81/k,0,h,mass=m,k=k,oldz=-m*9.81/k,oldv=0)
 np.testing.assert_allclose([q,w],[-m*9.81/k,0],atol=1e-15)

def test_force_free_particle_has_exact_uniform_velocity():
 q,v=scalar_step(.1,.2,.01,mass=.01,k=10000.,gravity=0,oldz=.098,oldv=.2)
 np.testing.assert_allclose([q,v],[.102,.2],atol=1e-16)
