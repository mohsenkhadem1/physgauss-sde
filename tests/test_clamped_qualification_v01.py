import numpy as np
from experiments.run_clamped_deformation_v01 import volume_metrics
from experiments.campaign_clamped_deformation_v01 import geometry


def test_exact_volume_rigid_motion_and_uniform_axial_strain():
    X,tets,*_=geometry();angle=.1
    Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    u=X@Q+np.array([.01,-.02,.005])-X
    r=volume_metrics(X,tets,u[None]);assert r['volume_nonrigid_coordinate_RMS_m']<1e-14
    delta=.001;u=np.zeros_like(X);u[:,0]=delta*X[:,0]/.24
    r=volume_metrics(X,tets,u[None]);assert np.isclose(r['volume_nonrigid_coordinate_RMS_m'],delta/6,rtol=1e-11)
    assert np.isclose(r['volume_motion_coordinate_RMS_m'],delta/3,rtol=1e-11)
