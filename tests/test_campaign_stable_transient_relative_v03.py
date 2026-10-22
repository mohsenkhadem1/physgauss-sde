import numpy as np
from experiments.campaign_stable_transient_relative_v03 import RelativeStableTransientV03

def test_moderate_u_same_energy_force_full_tangent_and_FD():
 m=RelativeStableTransientV03();rng=np.random.default_rng(81);u=m.model.rest@np.diag([.04,-.01,.005])+rng.normal(size=m.model.rest.shape)*1e-5;v=rng.normal(size=u.shape);v/=np.linalg.norm(v)
 E,g,H,J=m.elastic_u(u,True);old=m.model.elastic(m.model.rest+u,True)
 np.testing.assert_allclose(E,old[0],rtol=1e-12,atol=1e-14);np.testing.assert_allclose(g,old[1],atol=1e-12);np.testing.assert_allclose(H.toarray(),old[2].toarray(),atol=2e-11)
 h=1e-7;plus=m.elastic_u(u+h*v);minus=m.elastic_u(u-h*v)
 np.testing.assert_allclose((plus[0]-minus[0])/(2*h),np.sum(g*v),rtol=2e-7,atol=2e-9)
 np.testing.assert_allclose((plus[1]-minus[1]).ravel()/(2*h),H@v.ravel(),rtol=2e-7,atol=2e-7)

def test_fine_step_original_residual_tolerance_and_unchanged_roundoff_scale():
 m=RelativeStableTransientV03();s=m.initial();prev=None
 for n in range(12):
  new,d=m.step(s,1/7680,'BDF2',prev)
  assert d['objective_free_R_N']<=1e-10 and d['physical_original_free_R_N']<=1e-9
  assert d['Armijo_roundoff_scale_J']==m.model.young*m.model.volumes.sum()
  assert d['constraint_defect_m']==0 and d['minimum_J']>.1
  prev,s=s,new
