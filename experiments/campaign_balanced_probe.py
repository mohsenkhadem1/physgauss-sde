"""Known reference-central force pair for an elastic-excitation pilot.

Uses only public hardware. Net force and moment vanish at REFERENCE handle
origins; deformed origins can have a nonzero moment, which must be measured.
This is an open-loop prescribed world force, not an unknown inferred load or
a follower force. No observation, material or boundary truth is an input.
"""
from __future__ import annotations

import hashlib
import numpy as np

try:
    from campaign_fixed_actuator_clock import FixedActuatorClock, KNOWN_ROLE
except ImportError:
    from experiments.campaign_fixed_actuator_clock import FixedActuatorClock, KNOWN_ROLE


def balanced_probe_clock(handle_origins_world_m, *, amplitude_N=.2):
    origins = np.asarray(handle_origins_world_m, dtype=np.float64)
    if origins.shape != (2, 3) or not np.isfinite(origins).all():
        raise ValueError('Two finite public handle origins required')
    if not np.isfinite(amplitude_N) or amplitude_N <= 0:
        raise ValueError('Positive finite amplitude required')
    separation = origins[1]-origins[0]
    length = np.linalg.norm(separation)
    if length <= 0:
        raise ValueError('Distinct public handle origins required')
    endpoints = np.arange(361, dtype=np.float64)/120
    midpoint = (np.arange(360, dtype=np.float64)+.5)/120
    phase = np.clip((midpoint-.15)/2.4, 0., 1.)
    magnitude = float(amplitude_N)*np.sin(np.pi*phase)**2
    magnitude[(midpoint <= .15) | (midpoint >= 2.55)] = 0.
    positive = magnitude[:, None]*(separation/length)
    force = np.stack([-positive, positive], axis=1)
    return FixedActuatorClock(actuator_hz=120, endpoint_times_s=endpoints,
        output_times_s=endpoints[::6], handle_forces_world_N=force,
        handle_torques_world_Nm=np.zeros((360, 2, 3)),
        nodal_forces_world_N=np.zeros((360, 0, 3)),
        body_acceleration_world_m_s2=np.zeros((360, 3)), information_role=KNOWN_ROLE)


def wrench_diagnostics(clock, handle_origins_world_m):
    """Reference [2,3] or interval-specific origins[360,2,3]."""
    x = np.asarray(handle_origins_world_m, dtype=np.float64)
    if x.shape not in ((2, 3), (360, 2, 3)) or not np.isfinite(x).all():
        raise ValueError('Explicit finite reference or per-interval handle origins required')
    f = clock.handle_forces_world_N
    return dict(maximum_resultant_N=float(np.linalg.norm(f.sum(1), axis=-1).max()),
        maximum_moment_about_world_origin_Nm=float(np.linalg.norm(np.cross(x, f).sum(1), axis=-1).max()),
        maximum_per_handle_force_N=float(np.linalg.norm(f, axis=-1).max()),
        force_values_sha256=hashlib.sha256(f.tobytes()).hexdigest(),
        meaning='Zero reference moment is not a guarantee of zero moment after deformation')
