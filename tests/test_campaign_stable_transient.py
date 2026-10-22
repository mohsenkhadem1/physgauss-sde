import numpy as np
from scipy import sparse
from experiments.campaign_stable_transient import TransientState,advance,StableBlockTransient


def toy(k):
 def elastic(q,tangent=False):
  g=np.zeros_like(q);g[0,0]=k*q[0,0]
  return .5*k*q[0,0]**2,g,sparse.diags([k,0.,0.]) if tangent else None,np.ones(1)
 return elastic


def test_BE_mass_force_impulse_position_and_discrete_work_identity():
 m=np.array([2.]);F=np.array([[.3,0,0]]);mask=np.array([[False,True,True]]);s=TransientState(np.zeros((1,3)),np.zeros((1,3)));h=.02
 for n in range(1,11):
  old=s;s,d=advance(toy(0.),m,F,mask,s,h)
  np.testing.assert_allclose(s.v[0,0],n*h*.3/2,atol=2e-15)
  np.testing.assert_allclose(s.q[0,0],.3/2*h*h*n*(n+1)/2,atol=2e-15)
  kinetic_change=.5*np.sum(m[:,None]*(s.v*s.v-old.v*old.v));numerical=.5*np.sum(m[:,None]*(s.v-old.v)**2);work=np.sum(F*(s.q-old.q))
  assert abs(kinetic_change+numerical-work)<1e-16
  assert d['physical_original_free_R_N']<1e-12


def test_BDF2_independent_first_order_2x2_oscillator_discrete_oracle():
 mass=np.array([2.]);F=np.array([[.1,0,0]]);mask=np.array([[False,True,True]]);h=.03;k=7.;A=np.array([[0.,1.],[-k/2,0.]]);b=np.array([0.,.1/2]);s=TransientState(np.array([[.03,0,0]]),np.array([[-.02,0,0]]));prev=None;Y=np.array([.03,-.02]);Yprev=None
 for n in range(20):
  expected=np.linalg.solve(np.eye(2)-h*A,Y+h*b) if n==0 else np.linalg.solve(3*np.eye(2)-2*h*A,4*Y-Yprev+2*h*b)
  new,d=advance(toy(k),mass,F,mask,s,h,'BDF2',prev)
  np.testing.assert_allclose([new.q[0,0],new.v[0,0]],expected,atol=3e-14,rtol=0)
  assert d['physical_original_free_R_N']<1e-11
  prev,s=s,new;Yprev,Y=Y,expected


def test_physical_stable_block_original_BE_BDF2_residual_and_exact_constraints():
 model=StableBlockTransient();s=model.initial();prev=None
 assert abs(model.mass.sum()-.96)<1e-14
 for n in range(3):
  new,d=model.step(s,1/240,'BDF2',prev)
  assert d['minimum_J']>.1 and d['constraint_defect_m']==0
  assert d['objective_free_R_N']<=1e-10 and d['physical_original_free_R_N']<=1e-9
  np.testing.assert_array_equal(new.q[model.mask],model.model.rest[model.mask]);np.testing.assert_array_equal(new.v[model.mask],0.)
  if n:
   original=(3*new.v-4*s.v+prev.v)/(2/240)*model.mass[:,None]+model.model.elastic(new.q)[1]-model.force
   np.testing.assert_allclose(original[~model.mask],0,atol=1e-9)
  prev,s=s,new
