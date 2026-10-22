import numpy as np
from experiments.campaign_newton_contact_controls_v04 import oracle,metrics,JOBS

def test_oracle_piecewise_stationarity():
 for z,v in [(.1,-1),(.001,-1),(-.001,.2)]:
  h=.001;m=.01;k=10000.;r=.001
  q,w=oracle(z,v,h,m,k,r)
  np.testing.assert_allclose(m*(q-z-h*v-h*h*(-9.81))/h**2,k*max(r-q,0),atol=1e-11)
  assert w==(q-z)/h

def test_equilibrium_and_unilateral_release():
 r=.001;m=.01;k=10000.
 q,v=oracle(r-m*9.81/k,0,.001)
 np.testing.assert_allclose([q,v],[r-m*9.81/k,0],atol=1e-14)
 q,v=oracle(.001,1,.001,gravity=0)
 np.testing.assert_allclose([q,v],[.002,1])

def test_discrete_energy_and_impulse():
 h=1/960;m=.01;r=.001;k=10000.;q=[[0,0,.051]];v=[[0,0,0]]
 for _ in range(192):
  z,w=oracle(q[-1][2],v[-1][2],h);q.append([0,0,z]);v.append([0,0,w])
 met=metrics(q,v,np.array(q),np.array(v),m,k,r,h,True)
 assert met['impulse_balance_error_Ns']<1e-13
 assert met['maximum_energy_increase_J']<1e-13
 assert met['maximum_radius_corrected_penetration_m']>0

def test_scope_exact():
 assert [r['name'] for r in JOBS]==['equilibrium960','area_equivalent960']
