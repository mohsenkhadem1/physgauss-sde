import numpy as np
from scripts.external_benchmark.data import phystwin_position_mask

def raw():
    return dict(object_points=np.zeros((6,3,3)),object_visibilities=np.ones((6,3),bool),
                object_motions_valid=np.ones((6,3),bool))

def test_uses_incoming_transition_and_keeps_initial_visibility():
    r=raw();r['object_motions_valid'][0,0]=False;r['object_motions_valid'][-1]=False
    m=phystwin_position_mask(r,6)
    assert m[0].all() and not m[1,0] and m[-1].all()

def test_no_suffix_mask_or_position_can_change_prefix():
    r=raw();expected=phystwin_position_mask(r,4)
    r['object_motions_valid'][3:]=False
    r['object_visibilities'][4:]=False
    r['object_points'][4:]=np.nan
    np.testing.assert_array_equal(phystwin_position_mask(r,4),expected)

def test_nonfinite_or_invisible_observation_excluded():
    r=raw();r['object_points'][2,1]=np.nan;r['object_visibilities'][3,2]=False
    m=phystwin_position_mask(r,4)
    assert not m[2,1] and not m[3,2]

def test_virtual_camera_projection_uses_shared_pose():
    from scripts.real_transfer_v22.projected_landmarks import project
    K=np.array([[100.,0,256],[0,100,256],[0,0,1]])
    pose=np.eye(4);pose[:3,3]=[1,2,3]
    points=np.array([[1,2,5],[2,2,5.]])
    np.testing.assert_allclose(project(points,pose,K),[[256,256],[306,256]])
