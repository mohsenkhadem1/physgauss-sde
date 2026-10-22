"""Public equilibrium acquisitions, separate from timed trajectory APIs.

Source-only interface: no cohort, images, admission or physics is created.
Acquisition order is an integer ledger, not invented simulation time. Rest
geometry uses the existing exact allowlist; no physical labels enter B inputs.
"""
from __future__ import annotations
import numpy as np
from experiments.campaign_c38_data import _checked, _load, load_public_geometry


def _indices(value, name, *, unique=True):
    a=np.asarray(value)
    if a.ndim!=1 or not len(a) or a.dtype.kind not in 'iu' or np.any(a<0):
        raise ValueError('Nonnegative integer '+name+' required')
    if unique and len(np.unique(a))!=len(a):
        raise ValueError('Unique '+name+' required')
    return a


def load_public_static_controls(root, record, *, number_of_nodes):
    """Known load-state table only; loads must be compiled without E/support."""
    d=_load(root,record,('load_state_indices','load_vectors_N'))
    ids=_indices(d['load_state_indices'],'load state indices');f=d['load_vectors_N']
    if type(number_of_nodes) is not int or number_of_nodes<1 or f.shape!=(len(ids),number_of_nodes,3) or not np.isfinite(f).all():
        raise ValueError('Finite known force per load state and reference node required')
    return dict(information_role='public_known_static_external_forces',**d)


def load_public_static_rgbd_prefix(root, record, *, last_acquisition, known_load_state_indices):
    """Reject an unacquired frame before loading any image/camera members.

    Each acquisition may contain multiple calibrated views of one equilibrium.
    A repeated load state is allowed; acquisition ledger entries remain unique.
    This file contains no clocks, velocities, stiffness, masks or target shapes.
    """
    if type(last_acquisition) is not int or last_acquisition<0:
        raise ValueError('Nonnegative integer acquisition cutoff required')
    known=_indices(known_load_state_indices,'known load state indices')
    keys=('acquisition_indices','load_state_indices','rgb','depth_m','visible','camera_poses','intrinsics')
    with np.load(_checked(root,record),allow_pickle=False) as z:
        if set(z.files)!=set(keys):raise ValueError('Exact public equilibrium member allowlist required')
        a=_indices(z['acquisition_indices'],'acquisition indices')
        if np.any(a[1:]<=a[:-1]) or np.any(a>last_acquisition):
            raise ValueError('Only acquired ordered equilibrium frames may be loaded')
        states=_indices(z['load_state_indices'],'acquired load state indices',unique=False)
        if states.shape!=a.shape or not np.isin(states,known).all():
            raise ValueError('Acquisitions must bind to declared public load states')
        data={k:z[k].copy() for k in keys}
    d=data['depth_m'];v=data['visible'];r=data['rgb'];poses=data['camera_poses'];intrinsics=data['intrinsics']
    if d.ndim!=4 or d.shape[0]!=len(a) or min(d.shape[1:])<1 or v.dtype!=np.bool_ or v.shape!=d.shape or r.shape!=(*d.shape,3) or poses.shape!=(*d.shape[:2],4,4) or intrinsics.shape!=(*d.shape[:2],3,3) or not all(np.isfinite(x).all() for x in (d,r,poses,intrinsics)) or np.any(d[v]<=0) or np.any(d[~v]!=0):
        raise ValueError('Finite equilibrium RGBD/calibrated cameras and zero missing-depth convention required')
    return dict(information_role='public_acquired_static_RGBD_B',**data)
