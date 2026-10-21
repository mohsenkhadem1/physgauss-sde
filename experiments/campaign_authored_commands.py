"""Compile explicit authored TRAIN force interventions without loading responses.

The force compiler preserves ``isaac_dataset.forces_at`` float32 quantization,
then promotes the held commands to float64 for the independent solver. This
preserves authored commands, not native clock, engine or trajectory equivalence.
Caller roles are declarations: a recipe force is not automatically observable.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

try:
    from isaac_dataset import LOADS, forces_at
except ImportError:
    # The frozen original source uses sibling imports; do not modify it merely
    # to import this compiler as an experiments namespace package.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from isaac_dataset import LOADS, forces_at
    finally:
        sys.path.pop(0)


INFORMATION_ROLES = ('known_actuator_command', 'candidate_force_hypothesis')


def _validate_spec(spec):
    seconds = float(spec['seconds'])
    peak = float(spec['peak_force_N'])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Finite positive duration required')
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError('Exact positive authored peak required; no implicit scaling')
    for key in ['normal_direction_world', 'shear_direction_world', 'couple_direction_world']:
        value = np.asarray(spec[key], dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all() or not np.isclose(np.linalg.norm(value), 1., atol=1e-12, rtol=0):
            raise ValueError('Finite unit world direction required: '+key)
    return seconds, peak


def mathematical_peak_vectors(spec, force_family):
    """Unquantized vectors, solely for independent command diagnostics."""
    _, peak = _validate_spec(spec)
    if force_family not in LOADS:
        raise ValueError('Explicit authored force family required')
    result = np.zeros((2, 3), dtype=np.float64)
    key = {'push_pulse': 'normal_direction_world', 'pull_hold': 'normal_direction_world',
           'shear_oscillation': 'shear_direction_world', 'twist_couple': 'couple_direction_world'}[force_family]
    result[0] = peak*np.asarray(spec[key])*(-1 if force_family == 'push_pulse' else 1)
    if force_family == 'twist_couple':
        result[1] = -result[0]
    return result


def mathematical_total_impulse(spec, force_family):
    """Closed-form integral of the continuous, unquantized authored force law."""
    seconds, _ = _validate_spec(spec)
    integral_fraction = {'push_pulse': .45/2, 'twist_couple': .45/2,
                         'pull_hold': .2/2 + (.65-.3) + .25/2,
                         'shear_oscillation': 0.}
    if force_family not in integral_fraction:
        raise ValueError('Explicit authored force family required')
    return mathematical_peak_vectors(spec, force_family)*(seconds*integral_fraction[force_family])


def mathematical_impulse_diagnostics(spec, force_family):
    """Non-cancelling diagnostics also detect a missing oscillatory command."""
    seconds, _ = _validate_spec(spec)
    peak = mathematical_peak_vectors(spec, force_family)
    if force_family == 'shear_oscillation':
        def primitive(u):
            return -np.cos(4*np.pi*u)/(8*np.pi)+np.cos(6*np.pi*u)/(24*np.pi)+np.cos(2*np.pi*u)/(8*np.pi)
        absolute_area = .8*seconds*np.abs(np.diff(primitive(np.arange(5)/4))).sum()
        first_moment = (.8*seconds)**2/(24*np.pi)
    elif force_family in ['push_pulse', 'twist_couple']:
        absolute_area = .225*seconds
        first_moment = .225*.325*seconds**2
    elif force_family == 'pull_hold':
        absolute_area = .575*seconds
        first_moment = seconds**2*(.1*.2/2 + .2**2*(.25+1/np.pi**2)
            + (.65**2-.3**2)/2 + .65*.25/2 + .25**2*(.25-1/np.pi**2))
    else:
        raise ValueError('Explicit authored force family required')
    return {'continuous_componentwise_absolute_impulse_Ns': np.abs(peak)*absolute_area,
            'continuous_first_time_moment_Ns2': peak*first_moment}


def compile_authored_commands(spec, *, force_family, physics_hz, output_hz,
                              information_role):
    """Compile midpoint-held forces on an integer grid, with no dynamics call.

    Intervening on a force family does not alter layout, peak or directions.
    Compiled interval widths are endpoint differences; both those and nominal
    1/hz widths are retained to avoid silently conflating command clock rules.
    Outputs are exact members of the command endpoint grid.
    """
    seconds, _ = _validate_spec(spec)
    if force_family not in LOADS or information_role not in INFORMATION_ROLES:
        raise ValueError('Explicit force family and information role required')
    for name, value in [('physics_hz', physics_hz), ('output_hz', output_hz)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError('Positive integer '+name+' required')
    if physics_hz % output_hz or not (seconds*physics_hz).is_integer() or not (seconds*output_hz).is_integer():
        raise ValueError('Complete duration and common exact output grid required')
    count = int(seconds*physics_hz)
    endpoints = np.arange(count+1, dtype=np.float64)/physics_hz
    intervals = np.stack([endpoints[:-1], endpoints[1:]], axis=1)
    midpoints = (np.arange(count, dtype=np.float64)+.5)/physics_hz
    intervention = deepcopy(spec)
    intervention['force_type'] = force_family
    commands32 = np.stack([forces_at(intervention, t) for t in midpoints])
    if commands32.dtype != np.float32 or commands32.shape != (count, 2, 3) or not np.isfinite(commands32).all():
        raise ValueError('Authored compiler changed its finite two-handle float32 contract')
    commands = commands32.astype(np.float64)
    actual_impulse = np.einsum('t,thc->hc', np.diff(endpoints), commands)
    nominal_impulse = commands.sum(0)/physics_hz
    analytic = mathematical_total_impulse(spec, force_family)
    noncancelling = mathematical_impulse_diagnostics(spec, force_family)
    source_path = Path(forces_at.__code__.co_filename)
    return {
        'command_boundary_times_s': endpoints,
        'force_interval_times_s': intervals,
        'force_midpoints_s': midpoints,
        'nominal_step_durations_s': np.full(count, 1./physics_hz),
        'held_handle_forces_world_N': commands,
        'requested_output_times_s': endpoints[::physics_hz//output_hz],
        'metadata': {
            'information_role': information_role,
            'original_recipe_force_family': spec['force_type'],
            'requested_force_family': force_family,
            'force_family_intervention': force_family != spec['force_type'],
            'exact_authored_peak_force_N': float(spec['peak_force_N']),
            'force_precision': 'Original forces_at float32 result promoted exactly to float64; solver precision unchanged.',
            'force_application': 'World force at each rigid handle origin; zero pure torque, no external nodal force.',
            'midpoint_clock': '(integer_step_index+0.5)/physics_hz; held over [index/hz,(index+1)/hz].',
            'clock_evidence': 'Compiled integer grid only; a dynamics runner must separately record and reconcile measured solver-state clocks.',
            'command_source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'command_source_file': source_path.name,
            'compiled_endpoint_interval_impulse_Ns': actual_impulse.tolist(),
            'nominal_interval_impulse_Ns': nominal_impulse.tolist(),
            'continuous_unquantized_impulse_Ns': analytic.tolist(),
            **{key: value.tolist() for key, value in noncancelling.items()},
            'compiled_componentwise_absolute_impulse_Ns': np.einsum('t,thc->hc', np.diff(endpoints), np.abs(commands)).tolist(),
            'compiled_first_time_moment_Ns2': np.einsum('t,thc->hc', np.diff(endpoints)*midpoints, commands).tolist(),
            'maximum_compiled_vs_continuous_impulse_error_Ns': float(np.abs(actual_impulse-analytic).max()),
            'maximum_compiled_vs_nominal_clock_impulse_error_Ns': float(np.abs(actual_impulse-nominal_impulse).max()),
            'native_engine_or_trajectory_equivalence': False,
            'qualification': 'Command compilation only; no trajectory, numerical convergence or predictive qualification.',
        },
    }
