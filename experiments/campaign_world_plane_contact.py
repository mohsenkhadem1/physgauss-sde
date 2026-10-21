"""Unilateral world-plane penalty contact, independent of material fixtures.

This is a local constitutive primitive, not an exact nonpenetration solver or
an Isaac contact replacement. Vertex-lumped surface areas turn pressure into
nodal force. Only approach is damped; separation never creates attraction.
Tangential friction is a smooth, dissipative Coulomb approximation.
"""
from __future__ import annotations
import numpy as np


def plane_contact(*, positions_m, velocities_m_s, surface_areas_m2,
                  plane_origin_m, plane_normal, plane_velocity_m_s,
                  stiffness_Pa_per_m, damping_Pa_s_per_m,
                  friction_coefficient, slip_regularization_m_s):
    x=np.asarray(positions_m,dtype=float);v=np.asarray(velocities_m_s,dtype=float)
    area=np.asarray(surface_areas_m2,dtype=float)
    origin=np.asarray(plane_origin_m,dtype=float);normal=np.asarray(plane_normal,dtype=float)
    velocity=np.asarray(plane_velocity_m_s,dtype=float)
    if (x.ndim!=2 or x.shape[1]!=3 or v.shape!=x.shape or area.shape!=(len(x),)
        or origin.shape!=(3,) or normal.shape!=(3,) or velocity.shape!=(3,)):
        raise ValueError('Explicit nodal positions/velocities/areas and world plane vectors required')
    if not all(np.isfinite(a).all() for a in [x,v,area,origin,normal,velocity]) or np.any(area<0):
        raise ValueError('Finite inputs and nonnegative surface areas required')
    if not np.isclose(normal@normal,1.,rtol=0,atol=1e-12):
        raise ValueError('Plane normal must already be unit length')
    k,d,mu,epsilon=map(float,[stiffness_Pa_per_m,damping_Pa_s_per_m,
                             friction_coefficient,slip_regularization_m_s])
    if not np.isfinite([k,d,mu,epsilon]).all() or min(k,d,mu)<0 or epsilon<=0:
        raise ValueError('Nonnegative constitutive coefficients and positive slip scale required')
    gap=(x-origin)@normal;relative=v-velocity;vn=relative@normal
    penetration=np.maximum(-gap,0.)
    # Open nodes have zero force, including those approaching the plane.
    active=gap<0
    elastic_scalar=area*k*penetration
    damping_scalar=area*d*np.maximum(-vn,0.)*active
    normal_scalar=elastic_scalar+damping_scalar
    elastic=elastic_scalar[:,None]*normal;damping=damping_scalar[:,None]*normal
    tangent=relative-vn[:,None]*normal
    friction=-mu*normal_scalar[:,None]*tangent/np.sqrt(np.sum(tangent*tangent,axis=1)+epsilon**2)[:,None]
    force=elastic+damping+friction
    return dict(kind='unilateral_world_plane_finite_penalty_regularized_Coulomb',
        gap_m=gap,active=active,forces_world_N=force,
        elastic_forces_world_N=elastic,damping_forces_world_N=damping,
        friction_forces_world_N=friction,normal_force_N=normal_scalar,
        elastic_energy_J=float(.5*np.sum(area*k*penetration**2)),
        damping_dissipation_W=float(-np.sum(damping*relative)),
        friction_dissipation_W=float(-np.sum(friction*relative)),
        plane_actuator_power_W=float(np.sum(force@velocity)),
        scope='Local penalty law only; finite penetration, no sticking state, no contact inference or integration guarantee')
