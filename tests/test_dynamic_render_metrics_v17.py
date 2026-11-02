import numpy as np
import pytest
from scripts.compositional_dynamics_v17.render_metrics import compare_renderings


def frame(empty=False):
    mask=np.zeros((40,50),bool)
    if not empty:mask[12:28,17:33]=True
    rgb=np.ones((40,50,3))*.95;rgb[mask]=[.1,.4,.7]
    return dict(time_s=0.,rgb=rgb,body_visible=mask,body_visible_depth=np.where(mask,2.,np.inf),camera={'id':'fixed-calibration'})


def test_identity_has_perfect_visibility_and_finite_psnr_cap():
    x=frame();r=compare_renderings(x,x)
    assert r['ssim_rgb_global']==pytest.approx(1)
    assert r['silhouette_iou']==1 and r['depth_joint_rmse_m']==0
    assert r['psnr_global_db']==120 and r['rgb_global_mse']==0


def test_missing_body_cannot_look_good_from_shared_background_or_depth_intersection():
    r=compare_renderings(frame(True),frame())
    assert r['missing_predicted_pixels']==256
    assert r['silhouette_iou']==0 and r['reference_depth_coverage']==0
    assert r['depth_joint_rmse_m']is None
    assert r['psnr_foreground_union_db']<r['psnr_global_db']


def test_camera_and_depth_conventions_are_not_silently_reconciled():
    a=frame();b=frame();b['camera']={'id':'other'}
    with pytest.raises(ValueError,match='calibrated camera'):compare_renderings(a,b)
    a=frame();a['body_visible_depth'][0,0]=0
    with pytest.raises(ValueError,match='exactly'):compare_renderings(a,frame())


def test_render_time_mismatch_cannot_hide_phase_error():
    a=frame();b=frame();a["time_s"]=.1
    with pytest.raises(ValueError,match="timestamps"):compare_renderings(a,b)
