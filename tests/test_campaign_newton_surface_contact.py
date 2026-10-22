import numpy as np
from experiments.campaign_newton_surface_contact import surface_areas,penalty

def test_surface_refinement_and_affine_integrals():
 x=np.array([[0,0,0],[1,0,0],[0,1,0]],float);f=np.array([[0,1,2]])
 a=surface_areas(x,f)
 np.testing.assert_allclose(a.sum(),.5)
 np.testing.assert_allclose(a@x,[1/6,1/6,0])
 y=np.concatenate([x,[(x[0]+x[1])/2,(x[1]+x[2])/2,(x[0]+x[2])/2]])
 g=np.array([[0,3,5],[3,1,4],[5,4,2],[3,4,5]])
 b=surface_areas(y,g)
 for q in [a,b]:assert np.all(q>0)
 np.testing.assert_allclose(b.sum(),a.sum())
 np.testing.assert_allclose(b@y,a@x)
 for pts,w in [(x,a),(y,b)]:
  U,F=penalty(np.full(len(pts),-.002),w,1e8)
  np.testing.assert_allclose(F.sum(),1e5)
  np.testing.assert_allclose(U,100.)

def test_rigid_frame_area_and_energy_gradient():
 x=np.array([[0,0,0],[1,0,0],[0,1,0]],float);f=np.array([[0,1,2]])
 Q=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
 np.testing.assert_allclose(surface_areas(x@Q.T+7,f),surface_areas(x,f))
 a=surface_areas(x,f);d=np.array([-.002,.001,-.004]);U,F=penalty(d,a,1e6)
 for j in range(3):
  e=np.eye(3)[j]*1e-8
  grad=(penalty(d+e,a,1e6)[0]-penalty(d-e,a,1e6)[0])/2e-8
  np.testing.assert_allclose(grad,-F[j],atol=1e-7)
 assert U>=0 and np.all(F>=0) and F[1]==0
