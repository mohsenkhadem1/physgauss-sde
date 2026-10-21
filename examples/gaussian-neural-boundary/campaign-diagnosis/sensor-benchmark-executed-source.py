"""Chunked Torch mesh-ray sensor with the Isaac/NumPy audit conventions.

USD poses store translation in row 3. Pixel (x,y) means its centre (x+.5,y+.5).
Directions are NOT normalized: their camera z is -1, so intersection t is
camera-axis/image-plane depth in metres, rather than Euclidean range.

Float64 is the default to match validate_isaac_dataset.ray_depth, including its
absolute 1e-10 determinant cutoff and inclusive triangle-edge inequalities.
Exactly tangent/edge rays are inherently ill conditioned; no enlarged triangles,
hit-mask repair, depth clipping or visibility ground truth is used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class RayDepths:
    depth_m: torch.Tensor
    visible: torch.Tensor

    def censored(self, rest_depth_m, *, flatten_frames=False):
        """Transfer to the existing mixed visibility/range likelihood contract.

        With flatten_frames, [...,frames,rays] becomes [...,frames*rays].
        Missing residuals are harmless zeros and are NEVER Gaussian observations.
        """
        try:
            from censored_range_observation import CensoredRanges
        except ImportError:
            from experiments.censored_range_observation import CensoredRanges
        rest = torch.as_tensor(rest_depth_m, dtype=self.depth_m.dtype,
                               device=self.depth_m.device)
        residual = torch.where(self.visible, self.depth_m-rest, 0.)
        visible = self.visible
        if flatten_frames:
            if residual.ndim < 2:
                raise ValueError('Frame flattening requires at least [frames,rays]')
            residual = residual.flatten(-2)
            visible = visible.flatten(-2)
        if residual.ndim not in (1, 2):
            raise ValueError('CensoredRanges requires [D] or [particles,D]; flatten frames explicitly')
        return CensoredRanges(visible.detach().cpu().numpy(),
                              residual.detach().cpu().numpy())


def pixel_rays(pose, k, xs, ys, *, device=None, dtype=torch.float64):
    """Public cameras/pixels -> (origins[...,3], directions[...,rays,3]).

    Camera, intrinsic and pixel leading axes broadcast; no pose inversion or
    half-pixel rounding beyond the reference's explicit +.5 is performed.
    """
    pose = torch.as_tensor(pose, device=device, dtype=dtype)
    k = torch.as_tensor(k, device=device, dtype=dtype)
    xs = torch.as_tensor(xs, device=device, dtype=dtype)
    ys = torch.as_tensor(ys, device=device, dtype=dtype)
    if pose.shape[-2:] != (4, 4) or k.shape[-2:] != (3, 3):
        raise ValueError('Expected USD row poses[...,4,4] and K[...,3,3]')
    if xs.ndim < 1 or ys.ndim < 1 or xs.shape[-1] != ys.shape[-1]:
        raise ValueError('Equal nonempty ray axes required for xs and ys')
    if xs.shape[-1] == 0:
        raise ValueError('At least one ray required')
    if not bool(torch.isfinite(pose).all() and torch.isfinite(k).all()
                and torch.isfinite(xs).all() and torch.isfinite(ys).all()):
        raise ValueError('Cameras/pixels must be finite')
    if bool((k[..., 0, 0] <= 0).any() or (k[..., 1, 1] <= 0).any()):
        raise ValueError('Positive focal lengths required')
    lead = torch.broadcast_shapes(pose.shape[:-2], k.shape[:-2],
                                  xs.shape[:-1], ys.shape[:-1])
    pose = pose.expand(*lead, 4, 4)
    k = k.expand(*lead, 3, 3)
    xs, ys = xs.expand(*lead, xs.shape[-1]), ys.expand(*lead, ys.shape[-1])
    camera = torch.stack(((xs+.5-k[..., 0, 2, None])/k[..., 0, 0, None],
                          -(ys+.5-k[..., 1, 2, None])/k[..., 1, 1, None],
                          -torch.ones_like(xs)), dim=-1)
    return pose[..., 3, :3], camera @ pose[..., :3, :3]


def intersect_mesh_rays(positions, faces, origins, directions, *,
                        particle_chunk=16, ray_chunk=64, triangle_chunk=128,
                        dtype=torch.float64):
    """Batched two-sided Moller–Trumbore, with bounded working allocations.

    positions[...,N,3], origins[...,3], directions[...,R,3] broadcast on
    leading axes. Flattened [B,N,3] and [particles,times,N,3] both work.
    The largest intersection tensor is [particle_chunk,ray_chunk,triangle_chunk]
    (or that shape plus xyz). All triangles are tested, so this is O(B*R*F),
    not a spatial acceleration structure. Backpropagation through nearest-hit
    depths is piecewise differentiable; visibility/triangle choice is discrete.
    """
    if not all(isinstance(x, int) and x > 0 for x in
               (particle_chunk, ray_chunk, triangle_chunk)):
        raise ValueError('Chunk sizes must be positive integers')
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('Use float32 or float64 geometry, preferably float64')
    device = positions.device if isinstance(positions, torch.Tensor) else None
    q = torch.as_tensor(positions, device=device, dtype=dtype)
    faces = torch.as_tensor(faces, device=q.device, dtype=torch.long)
    origins = torch.as_tensor(origins, device=q.device, dtype=dtype)
    directions = torch.as_tensor(directions, device=q.device, dtype=dtype)
    if q.ndim < 2 or q.shape[-1] != 3 or faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError('Expected positions[...,nodes,3] and faces[triangles,3]')
    if origins.shape[-1:] != (3,) or directions.ndim < 2 or directions.shape[-1] != 3:
        raise ValueError('Expected origins[...,3] and directions[...,rays,3]')
    if directions.shape[-2] == 0:
        raise ValueError('At least one ray required')
    if not bool(torch.isfinite(q).all() and torch.isfinite(origins).all()
                and torch.isfinite(directions).all()):
        raise ValueError('Mesh and rays must be finite')
    if faces.numel() and bool((faces < 0).any() or (faces >= q.shape[-2]).any()):
        raise ValueError('Face index outside mesh')
    lead = torch.broadcast_shapes(q.shape[:-2], origins.shape[:-1], directions.shape[:-2])
    n, r, batch = q.shape[-2], directions.shape[-2], math.prod(lead)
    q = q.expand(*lead, n, 3).reshape(batch, n, 3)
    origins = origins.expand(*lead, 3).reshape(batch, 3)
    directions = directions.expand(*lead, r, 3).reshape(batch, r, 3)
    blocks = []
    for b in range(0, batch, particle_chunk):
        points, origin, rays = q[b:b+particle_chunk], origins[b:b+particle_chunk], directions[b:b+particle_chunk]
        ray_blocks = []
        for start in range(0, r, ray_chunk):
            ray = rays[:, start:start+ray_chunk]
            best = torch.full(ray.shape[:-1], torch.inf, dtype=dtype, device=q.device)
            for f in range(0, len(faces), triangle_chunk):
                tri = points[:, faces[f:f+triangle_chunk]]
                e1, e2 = tri[:, :, 1]-tri[:, :, 0], tri[:, :, 2]-tri[:, :, 0]
                # Scalar-triple-product rearrangement of the reference's h=
                # cross(ray,e2). Matrix products avoid a [B,R,F,3] temporary.
                # Algebra is identical; floating point summation may differ at
                # exactly grazing/triangle-edge queries, as documented above.
                det = torch.bmm(ray, torch.linalg.cross(e2, e1, dim=-1).transpose(1, 2))
                valid_det = det.abs() > 1e-10
                inv = torch.where(valid_det, det, 1.).reciprocal()
                s = origin[:, None]-tri[:, :, 0]
                u = inv*torch.bmm(ray, torch.linalg.cross(e2, s, dim=-1).transpose(1, 2))
                cross = torch.linalg.cross(s, e1, dim=-1)
                v = inv*torch.bmm(ray, cross.transpose(1, 2))
                t = inv*(e2*cross).sum(-1)[:, None]
                hit = valid_det & (u >= 0) & (v >= 0) & (u+v <= 1) & (t > 0)
                candidate = torch.where(hit, t, torch.inf).amin(-1)
                best = torch.minimum(best, candidate)
            ray_blocks.append(best)
        blocks.append(torch.cat(ray_blocks, dim=-1))
    if not blocks:
        depth = torch.empty((*lead, r), dtype=dtype, device=q.device)
    else:
        depth = torch.cat(blocks).reshape(*lead, r)
    return RayDepths(depth, torch.isfinite(depth))


def batched_ray_depth(positions, faces, pose, k, xs, ys, **kwargs):
    """Convenience API matching reference ray_depth arguments, with batching."""
    device = positions.device if isinstance(positions, torch.Tensor) else None
    origin, direction = pixel_rays(pose, k, xs, ys, device=device,
                                   dtype=kwargs.get('dtype', torch.float64))
    return intersect_mesh_rays(positions, faces, origin, direction, **kwargs)


class TorchMeshRaySensor:
    """Cache fixed public rays/faces on CPU or GPU for repeated forward queries."""
    def __init__(self, faces, query, *, device='cpu', dtype=torch.float64, **chunks):
        self.faces = torch.as_tensor(faces, dtype=torch.long, device=device)
        self.origins, self.directions = pixel_rays(query['pose'], query['K'],
            query['xs'], query['ys'], device=device, dtype=dtype)
        self.rest_depth_m = torch.as_tensor(query['rest_depth_m'], dtype=dtype, device=device)
        self.dtype, self.chunks = dtype, chunks

    def __call__(self, positions):
        positions = torch.as_tensor(positions, device=self.faces.device, dtype=self.dtype)
        return intersect_mesh_rays(positions, self.faces, self.origins,
                                   self.directions, dtype=self.dtype, **self.chunks)

    def predict_censored(self, positions, frame_ids=None):
        """Existing [K,T,N,3] -> CensoredRanges[K,len(frame_ids)*rays].

        Default excludes rest frame0 as in predict_censored_ranges. Shared rays
        work directly; per-time rays are indexed using the same frame IDs.
        """
        q = torch.as_tensor(positions, dtype=self.dtype, device=self.faces.device)
        if q.ndim != 4:
            raise ValueError('Expected positions[particles,times,nodes,3]')
        ids = torch.arange(1, q.shape[1], device=q.device) if frame_ids is None else torch.as_tensor(frame_ids, dtype=torch.long, device=q.device)
        if ids.ndim != 1 or not ids.numel() or bool((ids < 0).any() or (ids >= q.shape[1]).any()):
            raise ValueError('Nonempty frame indices inside the trajectory required')
        origin, direction, rest = self.origins, self.directions, self.rest_depth_m
        if direction.ndim == 3:
            if direction.shape[0] != q.shape[1]:
                raise ValueError('Per-time query cameras must cover every input time')
            origin, direction = origin[ids], direction[ids]
        elif direction.ndim != 2:
            raise ValueError('Censored wrapper supports shared or per-time camera queries')
        if rest.ndim == 2:
            rest = rest[ids]
        result = intersect_mesh_rays(q[:, ids], self.faces, origin, direction,
                                     dtype=self.dtype, **self.chunks)
        return result.censored(rest, flatten_frames=True)


def predict_censored_ranges(positions, reference, query, frame_ids=None, **kwargs):
    """Drop-in fast counterpart of censored_range_observation's prediction API."""
    return TorchMeshRaySensor(reference['surface_faces'], query,
        device=positions.device if isinstance(positions, torch.Tensor) else 'cpu',
        **kwargs).predict_censored(positions, frame_ids)
