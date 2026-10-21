"""Sparse range queries for depth-assisted acquisition with public geometry.

Pixels are chosen on the REST surface, before viewing any deformed observation.
The same mesh-ray observation operator predicts particle and measured features.
This is an explicit RGB-D experiment; it does not demonstrate RGB-only sensing.
"""
from __future__ import annotations
import numpy as np
try:
    from validate_isaac_dataset import ray_depth
except ImportError:
    from experiments.validate_isaac_dataset import ray_depth


def public_ray_queries(reference, pose, k, count=24, seed=31415, image_size=(640,480)):
    points=np.asarray(reference['rest_points_m']);faces=reference['surface_faces']
    camera=(points-pose[3,:3])@pose[:3,:3].T
    depth=-camera[:,2]
    if (depth<=0).any():raise ValueError('Reference must lie in front of camera')
    u=k[0,0]*camera[:,0]/depth+k[0,2];v=k[1,2]-k[1,1]*camera[:,1]/depth
    lower=np.maximum(np.floor([u.min(),v.min()]).astype(int),0)
    upper=np.minimum(np.ceil([u.max(),v.max()]).astype(int),np.asarray(image_size)-1)
    xx,yy=np.meshgrid(np.arange(lower[0],upper[0]+1,4),np.arange(lower[1],upper[1]+1,4))
    xs,ys=xx.ravel(),yy.ravel()
    rest=ray_depth(points,faces,pose,k,xs,ys)
    interior=np.isfinite(rest)
    # Restrict queries to pixels at least eight native pixels inside the rest
    # silhouette. The criterion uses no deformed state or ground-truth visibility.
    for dx,dy in [(8,0),(-8,0),(0,8),(0,-8)]:
        interior&=np.isfinite(ray_depth(points,faces,pose,k,xs+dx,ys+dy))
    eligible=np.flatnonzero(interior)
    if len(eligible)<count:raise ValueError('Too few public reference-interior range queries')
    chosen=np.random.default_rng(seed).choice(eligible,count,replace=False)
    return {'xs':xs[chosen],'ys':ys[chosen],'rest_depth_m':rest[chosen],
            'pose':np.asarray(pose),'K':np.asarray(k),
            'selection_rule':'Seeded sample of rest silhouette eroded by8native pixels'}


def range_features(depth, rest, maximum_residual_m=.05):
    """Censored range residuals; an absent return has the positive clip value.

    The Gaussian likelihood on these features is a declared working likelihood,
    including discrepancy. Its confidence must be checked empirically.
    """
    depth=np.asarray(depth);rest=np.asarray(rest)
    valid=np.isfinite(depth)&(depth>0)
    return np.where(valid,np.clip(depth-rest,-maximum_residual_m,maximum_residual_m),maximum_residual_m)


def predict_ranges(positions,reference,query):
    positions=np.asarray(positions)
    if positions.ndim!=3 or positions.shape[1:]!=reference['rest_points_m'].shape:
        raise ValueError('Expected [particles,nodes,3] positions matching reference')
    return np.stack([range_features(ray_depth(points,reference['surface_faces'],query['pose'],query['K'],query['xs'],query['ys']),
                                    query['rest_depth_m']) for points in positions])


def measure_ranges(depth_image,query):
    depth_image=np.asarray(depth_image)
    if depth_image.ndim!=2:raise ValueError('Expected a depth image')
    return range_features(depth_image[query['ys'],query['xs']],query['rest_depth_m'])


def discrepancy_covariance(count,independent_std_m=.001,common_std_m=.001):
    """A common offset prevents treating many correlated rays as independent."""
    if min(independent_std_m,common_std_m)<=0:raise ValueError('Positive discrepancy scales required')
    return np.eye(count)*independent_std_m**2+np.ones((count,count))*common_std_m**2
