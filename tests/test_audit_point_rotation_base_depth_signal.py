"""Synthetic censored-depth contrast and exact public-camera units checks."""
import numpy as np
import pytest
from experiments import audit_point_rotation_base_depth_signal as audit


def test_paired_hit_depth_RMS_bias_sigma_and_visibility_counts():
    actual=audit.CensoredRanges([1,1,0,0],[.0001,.0005,np.nan,np.nan])
    rigid=audit.CensoredRanges([1,0,1,0],[.0004,np.nan,.02,np.nan])
    r=audit.paired_depth_metrics(actual,rigid)
    assert r['both_visible_rays']==1 and r['true_visible_rigid_miss_rays']==1 and r['true_miss_rigid_visible_rays']==1 and r['both_background_rays']==1
    assert r['visibility_changed_rays']==2 and r['paired_visible_depth_difference_RMS_m']==pytest.approx(.0003)
    assert r['paired_visible_depth_difference_bias_m']==pytest.approx(-.0003)
    assert r['paired_visible_RMS_over_single_observation_sigma']==pytest.approx(1.5)
    assert r['paired_visible_rays_absolute_contrast_above_sigma']==1


def test_missed_rays_remain_absent_not_zero_depth_observations():
    a=audit.CensoredRanges([0,1],[np.nan,.001]);b=audit.CensoredRanges([1,0],[.9,np.nan])
    r=audit.paired_depth_metrics(a,b)
    assert r['both_visible_rays']==0 and r['paired_visible_depth_difference_RMS_m'] is None
    assert r['paired_visible_RMS_over_single_observation_sigma'] is None and r['paired_depth_squared_difference_sum_m2']==0
    assert r['visibility_changed_rays']==2


def test_identical_surface_depths_zero_contrast_without_touching_inputs():
    a=audit.CensoredRanges([1,0,1],[.002,np.nan,-.001]);visible=a.visible.copy();values=a.residual_m.copy()
    r=audit.paired_depth_metrics(a,a)
    assert r['paired_visible_depth_difference_RMS_m']==0 and r['visibility_changed_rays']==0 and r['both_background_rays']==1
    np.testing.assert_array_equal(a.visible,visible);np.testing.assert_array_equal(a.residual_m,values)


def test_synthetic_predict_ranges_camera_axis_depth_contrast_not_fake_noise():
    rest=np.array([[-1.,-1.,1.],[1.,-1.,1.],[0.,1.,1.]])
    q=np.broadcast_to(rest,(2,61,3,3)).copy();q[1,:,:,2]+=.0003
    pose=np.eye(4);pose[:3,:3]=np.diag([-1.,1.,-1.])
    query=dict(pose=pose,K=np.array([[1.,0.,.5],[0.,1.,.5],[0.,0.,1.]]),xs=[0],ys=[0],rest_depth_m=[1.])
    queries=[dict(query,frame_index=f) for f in [10,20]]
    pred=audit.predict_ranges(q,np.array([[0,1,2]]),queries,device='cpu',particle_chunk=2)
    assert pred.visible.all()
    np.testing.assert_allclose(pred.residual_m,[[0.,0.],[.0003,.0003]],atol=1e-15)
    result=audit.paired_depth_metrics(audit.CensoredRanges(pred.visible[0],pred.residual_m[0]),audit.CensoredRanges(pred.visible[1],pred.residual_m[1]))
    assert result['paired_visible_RMS_over_single_observation_sigma']==pytest.approx(1.5)


def test_all32_object_bootstrap_constant_and_missing_foreground_not_filtered():
    a=audit.CensoredRanges(np.ones(72,bool),np.full(72,.0001));b=audit.CensoredRanges(np.ones(72,bool),np.zeros(72));r=audit.paired_depth_metrics(a,b)
    objects=[dict(object_index=i,**{window:r for window in ['base72','frame10','frame20']}) for i in range(16,48)]
    s=audit.population_summary(objects)['base72']
    np.testing.assert_allclose(s['per_object_paired_RMS_m']['object_bootstrap_mean_95_interval'],[.0001,.0001])
    assert s['total_counts']['both_visible_rays']==32*72 and s['pooled_RMS_over_single_observation_sigma']==pytest.approx(.5)
    bg=audit.paired_depth_metrics(audit.CensoredRanges(np.zeros(72,bool),np.zeros(72)),audit.CensoredRanges(np.zeros(72,bool),np.zeros(72)))
    objects[-1]['base72']=bg;s=audit.population_summary(objects)['base72']
    assert s['objects']==32 and s['objects_without_paired_visible_rays']==1 and s['per_object_paired_RMS_m'] is None
    assert s['total_counts']['both_background_rays']==72
    with pytest.raises(ValueError,match='All32'):audit.population_summary(objects[:-1])


def test_rigid_fit_surface_reconstruction_matches_proper_row_world_convention():
    r=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    a=.7;R=np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    q=(r@R.T+[.3,.2,-.1])[None];fit=audit.geometry.rigid_fit(r,q,np.full(4,.25))
    restored=np.einsum('nc,tdc->tnd',r,fit['rotation_world_from_rest'])+fit['translation_m'][:,None]
    np.testing.assert_allclose(restored,q,atol=1e-15)
