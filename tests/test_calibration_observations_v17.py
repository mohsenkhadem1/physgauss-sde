import numpy as np
from scripts.compositional_dynamics_v17.calibration_observations import queries,observation_visibility
from scripts.physical_refinement_v15.observations import Camera


def test_material_sampling_contract():
    surface,interior=queries()
    assert surface.shape==(98,3) and interior.shape==(27,3)
    assert np.all(np.any((surface==0)|(surface==1),axis=1))
    assert np.all((interior>0)&(interior<1))
    np.testing.assert_array_equal(surface,np.unique(surface,axis=0))


def test_visibility_rejects_occluded_outside_and_behind_points():
    camera=Camera((0,-1,0),(0,0,0),32,24,20.)
    frame=dict(depth=np.ones((24,32)))
    points=np.array([[0,0,0],[0,.03,0],[100,0,0],[0,-2,0]])
    np.testing.assert_array_equal(observation_visibility(points,frame,camera),[True,False,False,False])
