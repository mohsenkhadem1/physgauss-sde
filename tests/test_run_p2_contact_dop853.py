import numpy as np
from scipy.linalg import eigh
from experiments.run_p2_contact_dop853 import integrate,accepted_partition,CONFIGS

def test_independent_coupled_oscillator_and_accepted_partition():
 M=np.array([[2.,.3],[.3,1.]]);K=np.array([[900.,-100.],[-100.,500.]]);q0=np.array([.01,-.03]);v0=np.array([.2,-.1]);e,Q=eigh(K,M);a=Q.T@M@q0;b=(Q.T@M@v0)/np.sqrt(e);times=np.linspace(0,.01,9)
 def rhs(t,y):return np.r_[y[2:],np.linalg.solve(M,-K@y[:2])]
 for c in CONFIGS:
  s=integrate(rhs,np.r_[q0,v0],[0,.01],times,c,np.full(4,1e-12));assert s.success
  exact=np.array([Q@(a*np.cos(np.sqrt(e)*t)+b*np.sin(np.sqrt(e)*t)) for t in times]);np.testing.assert_allclose(s.y[:2].T,exact,atol=1e-13,rtol=1e-11)
  part=accepted_partition(s);assert len(part)>len(times) and part[0]==0 and part[-1]==.01

def test_freefall_and_true_auxiliary_force_momentum():
 mass=.96;g=-9.81;t0=1/12;end=.1;y0=np.array([.5*g*t0*t0,g*t0,0.]);times=np.linspace(t0,end,9);seen=[]
 def rhs(t,y):return np.array([y[1],g,mass*g])
 def event(t,y):seen.append(t);return 1.
 s=integrate(rhs,y0,[t0,end],times,CONFIGS[1],np.full(3,1e-12),event)
 np.testing.assert_allclose(s.y[0],.5*g*times**2,atol=1e-14)
 np.testing.assert_allclose(s.y[1],g*times,atol=1e-14)
 np.testing.assert_allclose(mass*(s.y[1]-y0[1])-s.y[2],0,atol=1e-14)
 np.testing.assert_array_equal(accepted_partition(s),seen)
