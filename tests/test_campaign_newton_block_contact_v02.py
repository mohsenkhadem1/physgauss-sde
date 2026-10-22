import numpy as np
from experiments.campaign_newton_block_contact_v02 import geometry,probes,nonrigid,JOBS

def test_physical_geometry_surface_mass_and_nested_probes():
 for cells in [(4,2,2),(8,4,4)]:
  X,T,F,m,A=geometry(cells,.05)
  np.testing.assert_allclose(X.min(0),[0,-.04,.05],atol=1e-16)
  np.testing.assert_allclose(X.max(0),[.24,.04,.13],atol=1e-16)
  np.testing.assert_allclose(m.sum(),.96);np.testing.assert_allclose(A.sum(),.0896)
  assert np.linalg.det(np.swapaxes(X[T[:,1:]]-X[T[:,:1]],-1,-2)).min()>0
  assert np.all(A[np.unique(F)]>0)
 assert probes().shape==(81,3)

def test_rigid_removal_and_scope():
 X,T,F,m,A=geometry((4,2,2),.05)
 Q=np.array([[0,-1,0],[1,0,0],[0,0,1]])
 assert nonrigid(X,X@Q.T+1,m)<1e-15
 assert len(JOBS)==2 and all(j['iterations']==128 for j in JOBS)
 assert set(j['E'] for j in JOBS)=={300000.}
 assert set(j['kappa'] for j in JOBS)=={1e9,1e10}
