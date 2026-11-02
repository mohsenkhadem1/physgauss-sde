"""Paired calibrated-frame metrics; geometry, visibility and appearance separate.

SSIM uses scikit-image's documented Wang-compatible Gaussian settings. Full-image
scores are accompanied by visible-body union scores so a large common background
cannot hide missing predicted geometry. Caller fixes exposure and image encoding.
"""
import numpy as np


def _frame(frame):
    rgb=np.asarray(frame['rgb'],dtype=np.float64);mask=np.asarray(frame['body_visible']);depth=np.asarray(frame['body_visible_depth'],dtype=np.float64)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<11 or not np.isfinite(rgb).all() or (rgb<0).any() or (rgb>1).any():
        raise ValueError('Finite RGB in [0,1], at least11x11, required; no automatic exposure fitting')
    if mask.dtype!=bool or mask.shape!=rgb.shape[:2] or depth.shape!=mask.shape:
        raise ValueError('Explicit boolean body visibility and matching optical-axis depth required')
    if np.isnan(depth).any() or np.isneginf(depth).any() or not np.array_equal(np.isfinite(depth),mask) or (depth[mask]<=0).any():
        raise ValueError('Body depth must be positive finite exactly on visible-body mask, infinity outside')
    return rgb,mask,depth


def compare_renderings(prediction,reference):
    from skimage.metrics import structural_similarity
    a,pa,da=_frame(prediction);b,pb,db=_frame(reference)
    times=np.array([prediction['time_s'],reference['time_s']],float)
    if not np.isfinite(times).all() or abs(times[0]-times[1])>1e-12:
        raise ValueError('Matched physical timestamps required; no temporal alignment')
    if a.shape!=b.shape or prediction['camera']!=reference['camera']:
        raise ValueError('Same dimensions and calibrated camera required')
    union=pa|pb;joint=pa&pb;d=(a-b)**2
    global_mse=float(d.mean());union_mse=float(d[union].mean())if union.any()else None
    ssim,ssim_map=structural_similarity(a,b,data_range=1.,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True)
    inner=union[5:-5,5:-5];union_ssim=float(ssim_map[5:-5,5:-5][inner].mean())if inner.any()else None
    psnr=lambda mse:float(-10*np.log10(max(mse,1e-12)))if mse is not None else None
    return dict(time_s=float(times[1]),timestamp_roundoff_difference_s=float(times[0]-times[1]),rgb_global_mse=global_mse,rgb_foreground_union_mse=union_mse,
        psnr_global_db=psnr(global_mse),psnr_foreground_union_db=psnr(union_mse),psnr_upper_cap_db=120.,
        ssim_rgb_global=float(ssim),ssim_rgb_foreground_union=union_ssim,
        ssim_foreground_valid_center_count=int(inner.sum()),
        silhouette_iou=float(joint.sum()/union.sum())if union.any()else 1.,
        reference_body_pixels=int(pb.sum()),predicted_body_pixels=int(pa.sum()),union_body_pixels=int(union.sum()),joint_depth_pixels=int(joint.sum()),
        missing_predicted_pixels=int((pb&~pa).sum()),extra_predicted_pixels=int((pa&~pb).sum()),
        reference_depth_coverage=float(joint.sum()/pb.sum())if pb.any()else None,
        depth_joint_rmse_m=float(np.sqrt(np.mean((da[joint]-db[joint])**2)))if joint.any()else None,
        scope='Matched calibrated frames, no alignment/time shift/exposure fit. Depth is optical-axis metres. Foreground SSIM averages valid-window centers on body-mask union, not a separately cropped/resized image.',
        ssim_settings=dict(data_range=1.,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,channel_axis=2,valid_border_pixels=5))
