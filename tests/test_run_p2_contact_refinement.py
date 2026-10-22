import numpy as np
from experiments.run_p2_contact_refinement import initial_vectors,can_stop,T0,GRAVITY,RATES

def test_exact_analytic_initializer_and_BDF2_kinematics():
 X=np.array([[0,0,.05],[.24,.04,.13]])
 for hz in RATES:
  h=1/hz;u,v,old,oldv=initial_vectors(X,h)
  np.testing.assert_allclose((3*u-4*old+.5*(T0-2*h)**2*GRAVITY)/(2*h),v,atol=1e-12)
  np.testing.assert_allclose((v-oldv)/h, np.broadcast_to(GRAVITY,v.shape),atol=1e-12)
  np.testing.assert_allclose((X+u).min(0)[2],.0159375,atol=1e-15)
  assert hz//15%(hz//120)==0

def test_earlystop_needs_threshold_and_contraction_without_tuning():
 assert not can_stop([1e-6])
 assert not can_stop([1e-6,2e-6])
 assert not can_stop([40e-6,21e-6])
 assert can_stop([40e-6,20e-6])
 assert can_stop([3e-6,2e-6])
