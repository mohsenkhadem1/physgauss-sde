"""Public-rest camera queries and analytic sparse depth likelihood inputs.

This is an oracle depth-ray observation adapter, not an RGB encoder or Gaussian
rasterizer. It observes the compatible surface shared with material Gaussians.
Queries depend on known rest geometry/cameras only, never deformed truth or a
material/support label. Missing rays remain censored, not depth measurements.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from render_independent_trajectory import rest_camera_plan, camera_pose, camera_intrinsics, project_points, ray_surface
    from campaign_ray_sensor import pixel_rays, intersect_mesh_rays
    from censored_range_observation import concatenate_observations
except ImportError:
    from experiments.render_independent_trajectory import rest_camera_plan, camera_pose, camera_intrinsics, project_points, ray_surface
    from experiments.campaign_ray_sensor import pixel_rays, intersect_mesh_rays
    from experiments.censored_range_observation import concatenate_observations


def make_queries(rest, faces, *, frame_indices=(10,20), offsets_deg=((0.,0.),),
                 resolution=(192,160), grid_size=8):
    """A fixed grid in each public-rest projected box on the 20Hz clock.

    Background rest rays use public center depth solely as a residual origin;
    their absent range is never scored as an observation. Keeping those rays
    permits future visibility changes without truth-dependent pixel selection.
    """
    rest = np.asarray(rest,dtype=np.float64)
    faces = np.asarray(faces,dtype=np.int64)
    indices = list(frame_indices)
    if (not indices or len(set(indices))!=len(indices)
            or any(type(i) is not int or not 0<=i<=60 for i in indices)
            or type(grid_size) is not int or grid_size<2 or not offsets_deg):
        raise ValueError('Explicit unique frames0..60, cameras and grid size>=2 required')
    width,height = resolution
    k,_ = camera_intrinsics(width,height)
    plan = rest_camera_plan(rest,candidate_offsets=offsets_deg)
    queries = []
    for view,(azimuth,elevation) in enumerate(offsets_deg):
        for frame in indices:
            time_s = frame/20
            pose = camera_pose(plan,time_s,(azimuth,elevation))
            pixels,depth = project_points(rest,pose,k)
            if not np.isfinite(pixels).all() or np.any(depth<=0):
                raise ValueError('Public rest object must be in front of the camera')
            low = np.maximum(np.floor(pixels.min(0)).astype(int),[0,0])
            high = np.minimum(np.ceil(pixels.max(0)).astype(int),[width-1,height-1])
            if np.any(high-low<grid_size-1):
                raise ValueError('Grid is too dense for the projected public-rest box')
            xs,ys = np.meshgrid(np.linspace(low[0],high[0],grid_size).round().astype(int),
                                np.linspace(low[1],high[1],grid_size).round().astype(int))
            xs,ys = xs.flatten(),ys.flatten()
            depths,_,_ = ray_surface(rest,faces,pose,k,np.stack([xs+.5,ys+.5],axis=-1))
            hit = np.isfinite(depths)&(depths>0)
            # The independent NumPy audit has a smaller determinant cutoff.
            # Reject ambiguous rest queries rather than changing hit masks.
            origins,directions = pixel_rays(pose,k,xs,ys)
            checked = intersect_mesh_rays(torch.as_tensor(rest),faces,origins,directions)
            if (not np.array_equal(hit,checked.visible.numpy())
                    or not np.allclose(depths[hit],checked.depth_m.numpy()[hit],atol=1e-12,rtol=0)):
                raise ValueError('Public-rest ray query is numerically ambiguous')
            center_depth = float(((rest.mean(0)-pose[3,:3])@pose[:3,:3].T)[2]*-1)
            reference = np.where(hit,depths,center_depth)
            queries.append(dict(view_id=view,frame_index=frame,time_s=time_s,pose=pose,K=k.copy(),
                xs=xs,ys=ys,rest_depth_m=reference,rest_visible=hit,
                pixel_selection='Uniform grid of public-rest projected bounding box; no truth mask',
                depth_semantics='Camera-axis depth in metres, pixel centers x+.5/y+.5'))
    return queries


def predict_ranges(positions, faces, queries, *, device='cpu', particle_chunk=8):
    """[particles,61,nodes,3] compatible surfaces to censored depth residuals."""
    q = torch.as_tensor(positions,dtype=torch.float64,device=device)
    if q.ndim!=4 or q.shape[1]!=61 or q.shape[-1]!=3 or not bool(torch.isfinite(q).all()):
        raise ValueError('Finite full61-frame position hypotheses required')
    observations = []
    for query in queries:
        origins,directions = pixel_rays(query['pose'],query['K'],query['xs'],query['ys'],device=device)
        result = intersect_mesh_rays(q[:,query['frame_index']],faces,origins,directions,
                                    particle_chunk=particle_chunk)
        observations.append(result.censored(query['rest_depth_m']))
    if not observations:
        raise ValueError('At least one explicit public query required')
    return concatenate_observations(observations)
