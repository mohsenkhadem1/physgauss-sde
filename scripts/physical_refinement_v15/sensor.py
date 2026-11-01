"""Rendered-image feature sensor with declared Gaussian FEATURE measurement noise.
Noise is applied after actual RGB-D rendering/projection. It is not a raw-pixel
camera emulator, and its covariance is not calibrated to hardware. Clean rendered
images are diagnostic/internal prediction artifacts, not extra inference inputs.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.linalg import block_diag
from scripts.physical_refinement_v15.feature_likelihood import features,covariance,GRID,FAR_CLIP

@dataclass(frozen=True)
class SensorSettings:
    depth_sd_m: float=.0003
    rgb_sd: float=.005
    force_sd_N: float=.01
    torque_sd_Nm: float=.001
    # These document fixed feature/covariance implementation, not tunable options.
    tile_grid: int=GRID
    no_return_depth_m: float=FAR_CLIP
    spatial_correlation_pixels: float=64.
    registration_sd_pixels: float=.5
    def validate(self):
        if min(self.depth_sd_m,self.rgb_sd,self.force_sd_N,self.torque_sd_Nm)<=0:raise ValueError('Positive sensor scales required')
        if not np.isfinite([self.depth_sd_m,self.rgb_sd,self.force_sd_N,self.torque_sd_Nm]).all():raise ValueError('Finite sensor scales required')
        if (self.tile_grid,self.no_return_depth_m,self.spatial_correlation_pixels,self.registration_sd_pixels)!=(GRID,FAR_CLIP,64.,.5):raise ValueError('Featuregeometry/noise-correlation settings fixed for this development sensor')


def distribution(image,settings=SensorSettings(),force=None,visual_covariance=None,wrench_origin_world_m=None):
    """Prediction mean and covariance for SAME rendered mechanical hypothesis.

    For a fixed-noise acquisition experiment, pass visual_covariance constructed
    once from a public nominal reference image, shared by all hypotheses and the
    actual sensor. Do not choose it using hidden material/support labels.
    """
    settings.validate();mean=features(image['rgb'],image['depth'])
    C=covariance(image['rgb'],image['depth'],settings.depth_sd_m,settings.rgb_sd)if visual_covariance is None else np.asarray(visual_covariance,float).copy()
    if C.shape!=(len(mean),len(mean)) or not np.isfinite(C).all()or not np.allclose(C,C.T):raise ValueError('Invalid fixed visual covariance')
    np.linalg.cholesky(C)
    visual_indices=np.arange(len(mean),dtype=int);force_indices=np.array([],dtype=int);torque_indices=np.array([],dtype=int)
    units=['normalized_RGB','normalized_RGB','normalized_RGB','m']*GRID**2
    modality='rendered_RGBD_features';origin=None
    if force is not None:
        force=np.asarray(force,float)
        if force.shape not in ((3,),(6,)) or not np.isfinite(force).all():raise ValueError('Reaction must be a finite vector3 force or vector6 [Fx,Fy,Fz,Tx,Ty,Tz] wrench')
        begin=len(mean);force_indices=np.arange(begin,begin+3,dtype=int);sd=[settings.force_sd_N]*3;units+=['N']*3
        modality+='+reaction_force'
        if len(force)==6:
            origin=np.asarray(wrench_origin_world_m,float)
            if origin.shape!=(3,)or not np.isfinite(origin).all():raise ValueError('World wrench requires declared finite torque origin3')
            origin=origin.tolist();torque_indices=np.arange(begin+3,begin+6,dtype=int);sd+=[settings.torque_sd_Nm]*3;units+=['N m']*3
            modality='rendered_RGBD_features+reaction_wrench'
        mean=np.r_[mean,force];C=block_diag(C,np.diag(np.square(sd)))
    return dict(mean=mean,covariance=C,visual_indices=visual_indices,force_indices=force_indices,torque_indices=torque_indices,reaction_indices=np.r_[force_indices,torque_indices],channel_units=units,reaction_frame='world'if force is not None else None,wrench_origin_world_m=origin,settings=asdict(settings),modality=modality)



def measure(image,rng,settings=SensorSettings(),force=None,visual_covariance=None,wrench_origin_world_m=None):
    """Observable packet only: no clean mean, hidden label or mechanical state."""
    result=distribution(image,settings,force,visual_covariance,wrench_origin_world_m);mean=result.pop('mean')
    result['values']=mean+np.linalg.cholesky(result['covariance'])@rng.normal(size=len(mean))
    result['camera']=image['camera']
    result['noise_interpretation']='Gaussian feature noise after actual RGBD projection; fixed correlated nuisance approximation, not rawpixelnoise/jitter simulation or hardware calibration'
    return result
