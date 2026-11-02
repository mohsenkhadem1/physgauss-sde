import numpy as np
import pytest
from scripts.compositional_dynamics_v17.sequence_metrics import matched_sequence_metrics, proper_alignment


def cloud():
    return np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)


def test_rigid_motion_is_not_deformation_and_world_error_is_not_aligned_away():
    x=cloud(); R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    ref=np.array([x,x@R+[2,3,4]]); pred=np.array([x,x]); t=np.array([0.,1.])
    out=matched_sequence_metrics(x,t,ref,t,pred,np.ones(4),characteristic_length_m=1)
    assert out['world_point_rms_m']>1
    assert out['world_coordinate_rms_m']*np.sqrt(3)==pytest.approx(out['world_point_rms_m'])
    assert out['shape_point_rms_m']<1e-14
    assert out['reference_nonrigid_rms_m']<1e-14
    assert out['world_error_over_reference_deformation'] is None


def test_stretch_is_detected_and_reflection_not_allowed():
    x=cloud(); stretch=x*np.array([1.5,1,1]); t=np.array([0.,1.])
    out=matched_sequence_metrics(x,t,np.array([x,stretch]),t,np.array([x,x]),np.ones(4),characteristic_length_m=1)
    assert out['reference_nonrigid_rms_m']>0.1
    aligned,R,_=proper_alignment(x,x*np.array([-1,1,1]),np.ones(4))
    assert np.linalg.det(R)==pytest.approx(1)
    assert np.linalg.norm(aligned-x*np.array([-1,1,1]))>0.1


def test_failed_prefix_and_sample_time_mismatch_are_explicit():
    x=cloud(); t=np.array([0.,1.,2.]); ref=np.array([x,x,x])
    out=matched_sequence_metrics(x,t,ref,t[:1],ref[:1],np.ones(4),characteristic_length_m=1)
    assert not out['complete'] and out['first_missing_time_s']==1
    assert out['frame_coverage']==pytest.approx(1/3)
    with pytest.raises(ValueError,match='sample times'):
        matched_sequence_metrics(x,t,ref,np.array([0.1]),ref[:1],np.ones(4),characteristic_length_m=1)
