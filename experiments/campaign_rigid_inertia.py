"""Exact point-mass realization of physical COM inertia for a reference solver.

Auxiliary masses describe rigid kinetic energy only. They are not attachment
springs, collision geometry, extra hardware mass, or a native-engine change.
All points must share one rigid transform; independent motion is invalid.
"""
from __future__ import annotations
import numpy as np


def auxiliary_mass_points(mass, inertia):
    """Return six COM-centred points with the specified total mass and inertia.

    S = sum m_i r_i r_i.T = trace(I)/2 * identity - I.
    For each eigenpair (s_i,e_i) of S, put masses m/6 at
    +/-sqrt(3*s_i/m)*e_i. Positive inertias violating the rigid-body triangle
    inequalities are rejected. A planar distribution (one zero s_i) is valid.
    """
    mass=float(mass);I=np.asarray(inertia,dtype=float)
    if I.shape==(3,):I=np.diag(I)
    if not np.isfinite(mass) or mass<=0 or I.shape!=(3,3) or not np.isfinite(I).all():
        raise ValueError('Positive mass and finite 3x3 COM inertia required')
    scale=float(np.max(np.abs(I)))
    tolerance=64*np.finfo(float).eps*scale
    if scale==0 or np.max(np.abs(I-I.T))>tolerance:
        raise ValueError('Symmetric positive COM inertia required')
    I=.5*(I+I.T)
    if np.linalg.eigvalsh(I)[0]<=0:raise ValueError('Strictly positive COM inertia required')
    S=.5*np.trace(I)*np.eye(3)-I
    values,vectors=np.linalg.eigh(S)
    if values[0]<-tolerance:raise ValueError('Inertia violates physical triangle inequalities')
    # Only eigenvalues within the declared roundoff window can reach this clamp.
    # No physical negative second moment or zero rotational inertia is repaired.
    values=np.maximum(values,0.)
    axes=(vectors*np.sqrt(3*values/mass)[None,:]).T
    offsets=np.stack([axes,-axes],axis=1).reshape(6,3)
    masses=np.full(6,mass/6)
    return offsets,masses


def mass_properties(offsets,masses):
    points=np.asarray(offsets,dtype=float);m=np.asarray(masses,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or m.shape!=(len(points),) or np.any(m<=0):
        raise ValueError('Aligned positive point masses required')
    if not np.isfinite(points).all() or not np.isfinite(m).all():raise ValueError('Finite points/masses required')
    total=float(m.sum());com=(points*m[:,None]).sum(0)/total
    r=points-com
    S=np.einsum('n,ni,nj->ij',m,r,r)
    return total,com,np.trace(S)*np.eye(3)-S


def rigid_point_velocities(offsets,rotation,linear_velocity,angular_velocity):
    """Finite world pose; angular velocity is a WORLD vector in rad/s."""
    offsets=np.asarray(offsets,dtype=float);R=np.asarray(rotation,dtype=float)
    v=np.asarray(linear_velocity,dtype=float);omega=np.asarray(angular_velocity,dtype=float)
    if offsets.ndim!=2 or offsets.shape[1]!=3 or R.shape!=(3,3) or v.shape!=(3,) or omega.shape!=(3,):
        raise ValueError('Finite rigid kinematic shapes required')
    if not all(np.isfinite(x).all() for x in [offsets,R,v,omega]):raise ValueError('Finite rigid state required')
    if not np.allclose(R.T@R,np.eye(3),atol=1e-12,rtol=0) or not np.isclose(np.linalg.det(R),1,atol=1e-12,rtol=0):
        raise ValueError('Proper SO(3) rotation required')
    return v+np.cross(omega,offsets@R.T)
