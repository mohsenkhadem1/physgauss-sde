import numpy as np
from scipy.spatial.transform import Rotation
from scripts.coupled_contact_v13.physics import make_block
from scripts.coupled_contact_v13.surface_motion_audit import motion_metrics


def test_proper_rigid_motion_has_zero_deformation_and_physical_com():
    rest,tets=make_block((.08,.06,.05),(3,2,2),(.2,-.3,.1))
    rotation=Rotation.from_rotvec([.3,-.8,.2]).as_matrix()
    center=np.array([.2,-.3,.1]);shift=np.array([.06,-.02,.03])
    moved=(rest-center)@rotation.T+center+shift
    result=motion_metrics(rest,tets,np.stack([rest,moved]),[0.,1.])
    assert result['deformation_coordinate_rms_m']<1e-15
    assert result['world_coordinate_rms_m']>.01
    assert np.allclose(result['per_frame'][0]['physical_com_m'],center+shift,atol=1e-14)
    assert np.isclose(result['per_frame'][0]['proper_rotation_determinant'],1.)


def test_affine_stretch_matches_volume_integral_and_mesh_independence():
    size=np.array([.08,.06,.05]);strain=np.array([.2,-.1,.05])
    expected=np.sqrt(np.sum((strain*size)**2)/36)
    values=[]
    for cells in [(1,1,1),(3,2,4)]:
        rest,tets=make_block(size,cells,(0,0,0));stretched=rest*(1+strain)
        result=motion_metrics(rest,tets,np.stack([rest,stretched]),[0.,1.])
        values.append(result['deformation_coordinate_rms_m'])
        assert np.isclose(values[-1],expected,rtol=1e-12,atol=1e-15)
        assert np.isclose(result['deformation_world_rms_ratio'],1.,rtol=1e-12)
    assert np.isclose(*values,rtol=1e-12)


def test_global_rigid_coordinate_change_preserves_motion_metrics():
    rest,tets=make_block((.08,.06,.05),(2,2,2),(0,0,0));x=rest*[1.2,.9,1.05]+[.01,.02,-.03]
    frames=np.stack([rest,x]);rotation=Rotation.from_rotvec([.8,.1,-.3]).as_matrix();shift=np.array([2.,-1.,.3])
    a=motion_metrics(rest,tets,frames,[0.,1.]);b=motion_metrics(rest@rotation.T+shift,tets,frames@rotation.T+shift,[0.,1.])
    for key in ['world_coordinate_rms_m','deformation_coordinate_rms_m','deformation_world_rms_ratio']:
        assert np.isclose(a[key],b[key],rtol=1e-11,atol=1e-14)


def test_reflection_is_not_removed_as_a_proper_rotation():
    rest,tets=make_block((.08,.06,.05),(2,2,2),(0,0,0));mirrored=rest*[-1,1,1]
    result=motion_metrics(rest,tets,np.stack([rest,mirrored]),[0.,1.])
    assert result['deformation_coordinate_rms_m']>.001
    assert np.isclose(result['per_frame'][0]['proper_rotation_determinant'],1.)
