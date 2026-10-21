"""Candidate per-particle point-rotation geometry correction before observation.

Wraps an unchanged packed linear-modal forward. Corrected positions/times are
the ONLY physical outputs: original infinitesimal handle coordinates/velocities
are not relabelled as corrected dynamics. Public rest and candidate pin IDs are
explicit. No observations, truth labels or parameter fitting enter this class.
"""
from __future__ import annotations
import torch
try:
    from campaign_batched_point_rotation import retract_positions
except ImportError:
    from experiments.campaign_batched_point_rotation import retract_positions


class PointRotationForward:
    def __init__(self,packed,public_rest):
        X=torch.as_tensor(public_rest)
        if X.dtype!=torch.float64 or X.ndim!=2 or X.shape[1]!=3 or not bool(torch.isfinite(X).all()) or not callable(getattr(packed,'predict',None)):
            raise ValueError('Unchanged packed forward and explicit finite float64 public rest[N,3] required')
        self.packed=packed;self.rest=X.detach().clone()

    def predict(self,clock,geometry_ids,physical_coordinates,device='cpu',*,return_uncorrected=False):
        raw=self.packed.predict(clock,geometry_ids,physical_coordinates,device=device)
        q=raw['positions_m'];times=raw['times_s']
        if q.ndim!=4 or q.shape[1]!=61 or times.shape!=(61,):raise ValueError('Complete61-frame packed output required')
        correction=retract_positions(self.rest.to(q.device),q,geometry_ids.to(q.device))
        output=dict(positions_m=correction['positions_m'],times_s=times)
        if return_uncorrected:output['uncorrected_positions_m']=q
        return output


def signed_tet_diagnostics(positions,rest,tetrahedra):
    """Per-particle all61-frame extrema/counts; finite inversions are retained.

    One frame at a time bounds scratch; no Gaussian shape projection or tet
    reordering is applied to candidate outputs. This is a diagnostic only.
    """
    if positions.dtype!=torch.float64 or positions.ndim!=4 or positions.shape[-2:]!=rest.shape or positions.device!=rest.device:
        raise ValueError('Matching float64 positions[B,T,N,3]/rest[N,3] required')
    ids=torch.as_tensor(tetrahedra,dtype=torch.int64,device=positions.device)
    if ids.ndim!=2 or ids.shape[1]!=4 or not len(ids):raise ValueError('Explicit nonempty tetrahedra required')
    corners=rest[ids];edges=corners[:,1:]-corners[:,:1]
    rest_det=(edges[:,0]*torch.linalg.cross(edges[:,1],edges[:,2],dim=-1)).sum(-1)
    if not bool(torch.isfinite(rest_det).all()) or bool((rest_det==0).any()):raise ValueError('Nondegenerate reference tetrahedra required')
    minimum=torch.full((len(positions),),torch.inf,dtype=torch.float64,device=positions.device)
    count=torch.zeros(len(positions),dtype=torch.int64,device=positions.device)
    for frame in range(positions.shape[1]):
        p=positions[:,frame,ids];e=p[:,:,1:]-p[:,:,:1]
        J=(e[:,:,0]*torch.linalg.cross(e[:,:,1],e[:,:,2],dim=-1)).sum(-1)/rest_det
        if not bool(torch.isfinite(J).all()):raise ValueError('Nonfinite candidate tet Jacobian; abort/preserve, never filter')
        minimum=torch.minimum(minimum,J.min(-1).values);count+=(J<=0).sum(-1)
    return dict(minimum_J=minimum,nonpositive_tet_frame_count=count)
