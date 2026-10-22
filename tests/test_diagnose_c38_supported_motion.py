import numpy as np
from experiments.diagnose_c38_supported_motion import motion_components


def test_exact_rigid_motion_removed_under_nonuniform_positive_weights():
    X=np.random.default_rng(3).normal(size=(50,3));w=np.arange(1,51,dtype=float);theta=.6;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    out=motion_components(X,X@R+np.array([1.,-2.,.3]),w)
    assert out['total_coordinate_motion_RMS_m']>1 and out['nonrigid_coordinate_RMS_m']<1e-14


def test_symmetric_stretch_is_nonrigid_and_ratio_is_RMS():
    X=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],float);out=motion_components(X,1.1*X,np.ones(6))
    np.testing.assert_allclose(out['total_coordinate_motion_RMS_m'],.1/np.sqrt(3),rtol=1e-14)
    np.testing.assert_allclose(out['nonrigid_to_total_RMS_ratio'],1.,rtol=1e-14)
