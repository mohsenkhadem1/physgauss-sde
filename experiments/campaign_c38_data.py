"""Separate public inputs and admitted private trajectory targets for C38.

No loader infers qualification from exit status, scores or a single E/BC case.
Public RGBD files contain ONLY acquired prefix frames, not full trajectories.
This is an explicit data/API boundary, not an OS security sandbox.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
from experiments.campaign_c38_study import verify_admission


def _checked(root,record):
    root=Path(root).resolve();p=(root/record['path']).resolve()
    if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Changed/outside data record')
    return p


def _load(root,record,keys,*,exact=True):
    p=_checked(root,record)
    with np.load(p,allow_pickle=False) as z:
        if (exact and set(z.files)!=set(keys)) or not set(keys)<=set(z.files):raise ValueError('Exact member allowlist required; public inputs cannot contain private labels')
        return {k:z[k].copy() for k in keys}


def _geometry(data):
    X=data['rest_m'];e=data['elements'];t=data['tets'];f=data['faces']
    if X.ndim!=2 or X.shape[1]!=3 or not np.isfinite(X).all():raise ValueError('Finite public rest nodes required')
    for name,a,width in (('elements',e,10),('tets',t,4),('faces',f,None)):
        if a.ndim!=2 or a.dtype.kind not in 'iu' or (width is not None and a.shape[1]!=width) or (width is None and a.shape[1] not in (3,6)) or (a<0).any() or (a>=len(X)).any():raise ValueError('Valid P2 reference connectivity required: '+name)
    if len(e)!=len(t) or not np.array_equal(e[:,:4],t):raise ValueError('P2 and corner tetrahedra must be identically ordered')
    for value in data.values():value.flags.writeable=False
    return data


def load_public_geometry(root,record):
    """Rest geometry only, independent of physical labels/responses."""
    return _geometry(_load(root,record,('rest_m','elements','tets','faces')))


def load_public_rgbd_prefix(root,record,*,cutoff):
    """Reject future clocks before opening any image members."""
    p=_checked(root,record);keys=('times_s','rgb','depth_m','visible','camera_poses','intrinsics')
    with np.load(p,allow_pickle=False) as z:
        if set(z.files)!=set(keys):raise ValueError('Public RGBD member allowlist violated')
        t=z['times_s'].copy()
        if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or (np.diff(t)<=0).any() or not np.isfinite(cutoff) or np.any(t>cutoff+1e-12):raise ValueError('File must contain only acquired prefix times; future images are not loaded')
        data={k:z[k].copy() for k in keys if k!='times_s'};data['times_s']=t
    d=data['depth_m'];v=data['visible'];r=data['rgb'];poses=data['camera_poses'];k=data['intrinsics']
    if d.ndim!=4 or d.shape[0]!=len(t) or v.dtype!=np.bool_ or v.shape!=d.shape or r.shape!=(*d.shape,3) or poses.shape!=(*d.shape[:2],4,4) or k.shape!=(*d.shape[:2],3,3) or not all(np.isfinite(x).all() for x in (d,r,poses,k)) or np.any(d[v]<=0) or np.any(d[~v]!=0):raise ValueError('Clocked RGBD/cameras with explicit zero missing-depth convention required')
    return dict(information_role='public_acquired_RGBD_prefix_B',**data)


def load_known_controls(root,record):
    d=_load(root,record,('force_times_s','force_coefficients','load_basis_N'));t=d['force_times_s'];c=d['force_coefficients'];b=d['load_basis_N']
    if t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<=0) or c.ndim!=2 or c.shape[0]!=len(t)-1 or b.ndim!=3 or b.shape[0]!=c.shape[1] or b.shape[-1]!=3 or not all(np.isfinite(a).all() for a in (t,c,b)):raise ValueError('Literal known external-force intervals/basis required')
    return dict(information_role='known_external_reference_force_only',**d)


@dataclass(frozen=True)
class QualifiedTrajectory:
    geometry:dict
    times_s:np.ndarray
    positions_m:np.ndarray
    velocities_m_s:np.ndarray
    constraint_mask:np.ndarray
    information_role:str='PRIVATE_FULL_STATE_TARGET_OR_SUPPLIED_PHYSICS_ORACLE_A'


def load_qualified_targets(root,manifest,record):
    """Admission precedes ANY private NPZ read; never feed this object to B."""
    gate=verify_admission(manifest,root=root)
    admitted=gate.get('admitted_trajectories')
    if not isinstance(admitted,list) or not admitted:raise ValueError('Explicit admitted trajectory records required')
    matches=[r for r in admitted if isinstance(r,dict) and (r.get('path'),r.get('sha256'))==(record.get('path'),record.get('sha256'))]
    if len(matches)!=1:raise ValueError('Requested trajectory is not uniquely bound to the qualified domain')
    bound=matches[0]
    for key in ('case_id','episode_id'):
        values=[x[key] for x in (record,bound,gate) if key in x]
        if values and (key not in record or key not in bound or any(v!=values[0] for v in values)):raise ValueError('Admitted case identity differs')
    for item in (record,bound):
        if 'domain_sha256' in item and item['domain_sha256']!=manifest['domain_sha256']:raise ValueError('Admitted physical domain differs')
    keys=('rest_m','elements','tets','faces','times_s','positions_m','velocities_m_s','constraint_mask');d=_load(root,record,keys,exact=False);g=_geometry({k:d[k] for k in ('rest_m','elements','tets','faces')});t=d['times_s'];q=d['positions_m'];v=d['velocities_m_s'];mask=d['constraint_mask']
    if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t)<=0) or q.shape!=(len(t),len(g['rest_m']),3) or v.shape!=q.shape or mask.shape!=(len(g['rest_m']),3) or mask.dtype!=np.bool_ or not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('Complete matched private trajectory and exact constrained DOFs required')
    for a in (t,q,v,mask):a.flags.writeable=False
    return QualifiedTrajectory(g,t,q,v,mask)
