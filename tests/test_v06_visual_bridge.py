import numpy as np
from scripts.phystwin.v06_visual import camera,project,backproject,render,track
from scripts.phystwin.v06_physbridge import forces,masses
from scripts.phystwin.sequential_fixture import SequentialFixture,points,ACTIONS

def test_camera_roundtrip_and_marker_quantization():
 x=points();cam=camera();uv,z=project(x,cam);assert np.allclose(backproject(uv,z,cam),x,atol=1e-14)
 ids=[36,37,38];rgb,depth,cam=render(x,0,ids)
 class NoNoise:
  def normal(self,*args):return 0.
 measured=track(rgb,depth,cam,ids,NoNoise());assert np.sqrt(np.mean((measured-x[ids])**2))<.0004

def test_phystwin_force_projection_preserves_total_force_and_moment():
 m=SequentialFixture(2,3000,200,1500,'crossed',device='cpu')
 assert np.isclose(masses().sum(),.96)
 for action in ACTIONS:
  m.set_action(action);f=forces(action)
  assert np.allclose(f.sum(0),m.load.sum(0),atol=1e-13)
  assert np.allclose(np.cross(points(),f).sum(0),np.cross(m.rest,m.load).sum(0),atol=1e-13)

def test_python310_camera_copy_matches_sealed_renderer():
 from scripts.phystwin.v06_camera import render as compatible
 original=render(points(),-1,[0,1,2]);copied=compatible(points(),-1,[0,1,2])
 assert np.array_equal(original[0],copied[0])
 assert np.array_equal(original[1],copied[1])
