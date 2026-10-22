import numpy as np
from experiments.campaign_p2_contact_root_fixed import integrate_face_fixed
from experiments.campaign_p2_contact_root_integration import integrate_face
N=np.array([[0.,0.],[1,0],[0,1],[.5,0],[0,.5],[.5,.5]])

def test_fixedouter_mapping_metric_constant_affine_quadratic_and_degeneracies():
 for fun in [lambda r,s:0*r,lambda r,s:0*r+.1,lambda r,s:0*r-.001,lambda r,s:r-.31+1e-15*s*s,lambda r,s:(r-.3)**2,lambda r,s:(r-.3)**2+(s-.3)**2-.04,lambda r,s:s-.5*r]:
  h=np.asarray(fun(N[:,0],N[:,1]),dtype=np.float64);ref=integrate_face(h,.03,kappa_A=1500);a=integrate_face_fixed(h,.03,kappa_A=1500,outer_order=16);b=integrate_face_fixed(h,.03,kappa_A=1500,outer_order=32)
  assert abs(b['energy_J']-ref['energy_J'])<1e-12
  np.testing.assert_allclose(b['gradient_N'],ref['gradient_N'],atol=1e-10,rtol=1e-12)
  np.testing.assert_allclose(b['hessian_N_m'],ref['hessian_N_m'],atol=1e-7,rtol=1e-10)
  assert abs(a['energy_J']-b['energy_J'])<1e-12
  np.testing.assert_allclose(a['gradient_N'],b['gradient_N'],atol=1e-10,rtol=1e-12)
  assert np.linalg.eigvalsh(b['hessian_N_m']).min()>-1e-10
  np.testing.assert_allclose(b['hessian_N_m'],b['hessian_N_m'].T,atol=1e-13,rtol=0)
