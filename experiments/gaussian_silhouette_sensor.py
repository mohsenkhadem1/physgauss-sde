"""RGB-mask support functions for acquisition, without depth or mesh truth.

Support functions depend on the convex hull of a silhouette, so projecting all
mesh vertices predicts them even when the silhouette itself is nonconvex. This
does not predict interior concavities or appearance. Masks are supplied inputs.
Native pixel quantization and correlated angular errors belong in the noise.
"""
import numpy as np


def query(reference,pose,k,angles=16):
    center=np.asarray(reference['rest_points_m']).mean(0)
    z=-((center-pose[3,:3])@pose[:3,:3].T)[2]
    if z<=0:raise ValueError('Object center must lie in front of camera')
    theta=np.arange(angles)*2*np.pi/angles
    return {'pose':np.asarray(pose),'K':np.asarray(k),'reference_axis_depth_m':float(z),
            'directions':np.stack([np.cos(theta),np.sin(theta)],-1)}


def predict_support(positions,sensor,calibration_pixel_offset=None):
    positions=np.asarray(positions);pose=sensor['pose'];k=sensor['K']
    camera=(positions-pose[3,:3])@pose[:3,:3].T;depth=-camera[...,2]
    if (depth<=0).any():raise ValueError('Predicted object crossed camera plane')
    # Use common pixel units (equal fx/fy in this dataset). Image v is downward.
    uv=np.stack([k[0,0]*camera[...,0]/depth,-k[1,1]*camera[...,1]/depth],-1)
    support=np.einsum('...nc,dc->...nd',uv,sensor['directions']).max(-2)
    if calibration_pixel_offset is not None:support=support+np.asarray(calibration_pixel_offset)
    return support


def measure_support(mask,sensor):
    mask=np.asarray(mask)
    if mask.ndim!=2:raise ValueError('Expected a single segmentation image')
    ys,xs=np.nonzero(mask>.5)
    if len(xs)<10:raise ValueError('Object mask is empty or too small')
    k=sensor['K'];uv=np.stack([xs+.5-k[0,2],ys+.5-k[1,2]],-1)
    return (uv@sensor['directions'].T).max(0)


def covariance(angles=16,independent_std_pixels=.75,common_std_pixels=.75):
    if min(independent_std_pixels,common_std_pixels)<=0:raise ValueError('Positive noise required')
    theta=np.arange(angles)*2*np.pi/angles
    # Independent raster error, global silhouette expansion and shared centroid
    # shifts. Positive definite even when angular features are redundant.
    basis=np.stack([np.ones(angles),np.cos(theta),np.sin(theta)],-1)
    return np.eye(angles)*independent_std_pixels**2+common_std_pixels**2*(basis@basis.T)
