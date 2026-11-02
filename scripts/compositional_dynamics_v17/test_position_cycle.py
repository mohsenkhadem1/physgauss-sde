import numpy as np
from scripts.compositional_dynamics_v17.position_cycle import build,controls,tool_wrenches,controlled_step

def test_profile_endpoints_and_fixed_state_work_sign():
 np.testing.assert_allclose(controls(0),0);np.testing.assert_allclose(controls(.3),0)
 np.testing.assert_allclose(controls(.1),controls(.2));assert np.isclose(controls(.15)[0,0],.018)
 m,e=build(1);u=np.zeros_like(m.rest);u[:,0]=.002*np.sin(m.rest[:,0]*10);a=controls(.03);b=controls(.031);m.set_controls(a);U0=m.evaluate(u)['potential_J'];f0=tool_wrenches(m,u);m.set_controls(b);U1=m.evaluate(u)['potential_J'];f1=tool_wrenches(m,u)
 W=-.5*np.sum((f0[:,:3]+f1[:,:3])*(b[:,:3]-a[:,:3]));np.testing.assert_allclose(U1-U0,W,rtol=1e-10,atol=1e-12)

def test_time_dependent_verlet_work_convergence():
 errors=[];ends=[]
 for dt in [80e-6,40e-6,20e-6]:
  m,e=build(1);s=e.initial_state();W=0.;maximum=0.;peak=0.
  for i in range(1,round(.004/dt)+1):
   pose=controls(.1*(i*dt)/.004)*(.0015/.018);s,dW,_=controlled_step(e,s,dt,pose);W+=dW;maximum=max(maximum,abs(e.energy(s)-W));peak=max(peak,abs(W))
  errors.append(maximum/peak);ends.append(s.u)
 assert errors[0]/errors[1]>3 and errors[1]/errors[2]>3
 assert errors[-1]<.005
 assert np.linalg.norm(ends[0]-ends[1])/np.linalg.norm(ends[1]-ends[2])>3
