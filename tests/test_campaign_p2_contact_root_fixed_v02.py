import numpy as np
from experiments.campaign_p2_contact_root_fixed import integrate_face_fixed
from experiments.campaign_p2_contact_root_integration import integrate_face
N=np.array([[0.,0.],[1,0],[0,1],[.5,0],[0,.5],[.5,.5]])

def test_original_reference_and_fixed32_64_pathological_same_strict_gates():
 functions=[lambda r,s:0*r,lambda r,s:0*r+.1,lambda r,s:0*r-.001,lambda r,s:r-.31+1e-15*s*s,lambda r,s:(r-.3)**2,lambda r,s:(r-.3)**2+(s-.3)**2-.04,lambda r,s:s-.5*r,lambda r,s:(r-.05)**2+(s-.15)**2-.0001]
 for f in functions:
  h=np.asarray(f(N[:,0],N[:,1]),dtype=np.float64);a=integrate_face(h,.03,kappa_A=1500);b=integrate_face_fixed(h,.03,kappa_A=1500,outer_order=32);c=integrate_face_fixed(h,.03,kappa_A=1500,outer_order=64)
  for x,y in [(a,b),(a,c),(b,c)]:
   assert abs(x['energy_J']-y['energy_J'])<=1e-12
   assert np.linalg.norm(x['gradient_N']-y['gradient_N'])<=1e-10
   assert abs(x['hessian_N_m']-y['hessian_N_m']).max()<=1e-7
  assert np.linalg.eigvalsh(c['hessian_N_m']).min()>-1e-10
