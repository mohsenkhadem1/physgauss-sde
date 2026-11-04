import numpy as np
from scripts.gaussian_boundary_flow_v02.prefix_informativeness import measure_prefix


def cloud():return np.array(list(__import__('itertools').product((-.1,.1),repeat=3)),float)


def test_pure_rigid_motion_has_zero_shape_signal():
    x=cloud();R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    points=np.stack([x,x@R+np.array([.2,.1,0.])]);r=measure_prefix(points,np.ones((2,8),bool))
    assert r['eligible'] and r['aggregate_total_motion_RMS_mm']>50
    assert r['aggregate_rigid_removed_shape_RMS_mm']<1e-10


def test_nonrigid_scaling_survives_kabsch():
    x=cloud();points=np.stack([x,x*1.1]);r=measure_prefix(points,np.ones((2,8),bool))
    assert abs(r['aggregate_rigid_removed_shape_RMS_mm']-10)<1e-10
    assert abs(r['aggregate_shape_to_total_RMS_ratio']-1)<1e-12


def test_invalid_and_initial_invisible_tracks_are_excluded():
    x=cloud();y=x+np.array([.2,0.,0.]);y[0]=np.nan;y[1]=100
    mask=np.ones((2,8),bool);mask[0,1]=False
    r=measure_prefix(np.stack([x,y]),mask)
    assert r['frames'][0]['common_valid_points']==6
    assert r['aggregate_rigid_removed_shape_RMS_mm']<1e-9


def test_insufficient_and_collinear_cases_are_ineligible():
    x=cloud()[:5];r=measure_prefix(np.stack([x,x+.1]),np.ones((2,5),bool))
    assert not r['eligible']
    x=np.c_[np.linspace(0,1,8),np.zeros((8,2))]
    r=measure_prefix(np.stack([x,x+.1]),np.ones((2,8),bool));assert not r['eligible']


def test_stationary_prefix_does_not_report_arbitrary_ratio():
    x=cloud();r=measure_prefix(np.stack([x,x]),np.ones((2,8),bool))
    assert r['aggregate_shape_to_total_RMS_ratio'] is None
    assert r['moving_frame_RMS_ratio'] is None


def test_reflection_is_not_removed_as_a_proper_rigid_motion():
    x=cloud();y=x.copy();y[:,0]*=-1
    r=measure_prefix(np.stack([x,y]),np.ones((2,8),bool))
    assert r['aggregate_rigid_removed_shape_RMS_mm']>100
