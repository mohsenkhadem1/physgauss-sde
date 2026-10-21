"""Fixed known-actuator clock, separate from private integrator refinement.

Inputs are explicit endpoint times and held world commands only. There is no
waveform, material, boundary, recipe, response, support law or physics backend.
Refinement repeats each command; it does not reevaluate or interpolate it.
This component alone qualifies no dataset, simulator or inference procedure.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

KNOWN_ROLE = 'known_actuator_command'
VERSION = 'fixed-known-actuator-clock-v0.1'
LIMITATION = ('Command-grid compilation only; actual solver clocks must be measured separately. '
              'No support-law interpolation or execution is implemented. '
              'A caller must freeze one public actuator clock independently of latent labels. '
              'Production dataset and physical convergence are not qualified.')


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(name+' must be an explicit positive integer')
    if value > 2**53:
        raise ValueError(name+' exceeds exact float64 integer clock representation')
    return int(value)


def _immutable(value, shape, name, *, dtype=np.float64):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise ValueError(name+' must contain finite real numeric values')
    result = np.array(raw, dtype=dtype, copy=True, order='C')
    if result.ndim != len(shape) or any(want is not None and want != got for want, got in zip(shape, result.shape)):
        raise ValueError(name+' has invalid dimensions')
    # Reject a lossy wider-float/integer input conversion. float32 promotion is exact.
    with np.errstate(invalid='ignore', over='ignore'):
        restored = result.astype(raw.dtype)
    if not np.isfinite(result).all() or not np.array_equal(restored, raw):
        raise ValueError(name+' cannot be represented exactly at the declared precision')
    # Immutable bytes backing also prevents callers re-enabling write access.
    return np.frombuffer(result.tobytes(), dtype=np.dtype(dtype)).reshape(result.shape)


def _description(value):
    return {'shape':list(value.shape), 'dtype':str(value.dtype),
            'sha256':hashlib.sha256(value.tobytes(order='C')).hexdigest()}


@dataclass(frozen=True, kw_only=True)
class FixedActuatorClock:
    """Public known commands on an explicit uniform integer-Hz clock from t=0.

    Endpoints must be exactly arange(T+1)/actuator_hz in float64. Noncanonical
    clocks are rejected, never rounded or silently changed. Requested outputs
    are an explicit nonempty increasing subset of those public endpoints.
    Forces act at handle origins; additional pure torques remain explicit.
    Acceleration is not converted into density-dependent nodal forces.
    """
    actuator_hz: int
    endpoint_times_s: np.ndarray
    output_times_s: np.ndarray
    handle_forces_world_N: np.ndarray
    handle_torques_world_Nm: np.ndarray
    nodal_forces_world_N: np.ndarray
    body_acceleration_world_m_s2: np.ndarray
    information_role: str

    def __post_init__(self):
        hz = _integer(self.actuator_hz, 'actuator_hz')
        if self.information_role != KNOWN_ROLE:
            raise ValueError('Only explicitly known actuator commands are accepted')
        times = _immutable(self.endpoint_times_s, (None,), 'endpoint_times_s')
        if len(times) < 2 or not np.array_equal(times, np.arange(len(times), dtype=np.float64)/hz):
            raise ValueError('Exact increasing integer actuator clock from zero required')
        T = len(times)-1
        forces = _immutable(self.handle_forces_world_N, (T, None, 3), 'handle forces')
        H = forces.shape[1]
        values = {'actuator_hz':hz, 'endpoint_times_s':times, 'handle_forces_world_N':forces,
            'handle_torques_world_Nm':_immutable(self.handle_torques_world_Nm, (T,H,3), 'handle pure torques'),
            'nodal_forces_world_N':_immutable(self.nodal_forces_world_N, (T,None,3), 'nodal forces'),
            'body_acceleration_world_m_s2':_immutable(self.body_acceleration_world_m_s2, (T,3), 'body acceleration')}
        outputs = _immutable(self.output_times_s, (None,), 'output_times_s')
        if (not len(outputs) or np.any(np.diff(outputs) <= 0) or not np.isin(outputs, times).all()):
            raise ValueError('Outputs must be a nonempty increasing exact subset of public command endpoints')
        values['output_times_s'] = outputs
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_force_program(cls, *, actuator_hz, endpoint_times_s, output_times_s, force_program):
        """Read only the typed known role and four command arrays, no provenance labels."""
        try:
            from campaign_full_hardware_context import ForceProgram
        except ImportError:
            from experiments.campaign_full_hardware_context import ForceProgram
        if type(force_program) is not ForceProgram:
            raise TypeError('An exact typed ForceProgram is required')
        return cls(actuator_hz=actuator_hz, endpoint_times_s=endpoint_times_s, output_times_s=output_times_s,
            handle_forces_world_N=force_program.forces, handle_torques_world_Nm=force_program.torques,
            nodal_forces_world_N=force_program.nodal, body_acceleration_world_m_s2=force_program.body_acceleration,
            information_role=force_program.information_role)

    def public_features(self):
        """Identical keys/bytes for every private solver rate; no private settings."""
        return {'command_boundary_times_s':self.endpoint_times_s,
            'requested_output_times_s':self.output_times_s,
            'held_handle_forces_world_N':self.handle_forces_world_N,
            'held_handle_torques_world_Nm':self.handle_torques_world_Nm,
            'held_nodal_forces_world_N':self.nodal_forces_world_N,
            'held_body_acceleration_world_m_s2':self.body_acceleration_world_m_s2}

    def public_manifest(self):
        result = {'schema':VERSION, 'information_role':self.information_role,
            'actuator_hz':self.actuator_hz,
            'arrays':{key:_description(value) for key,value in self.public_features().items()},
            'qualification':LIMITATION, 'production_dataset_qualified':False}
        result['manifest_sha256'] = hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return result


@dataclass(frozen=True, kw_only=True)
class RefinedActuatorClock:
    """Private execution schedule; never a replacement public input clock."""
    public_clock: FixedActuatorClock
    integration_hz: int
    substeps_per_public_interval: int
    endpoint_times_s: np.ndarray
    parent_command_indices: np.ndarray
    output_step_indices: np.ndarray
    handle_forces_world_N: np.ndarray
    handle_torques_world_Nm: np.ndarray
    nodal_forces_world_N: np.ndarray
    body_acceleration_world_m_s2: np.ndarray

    @property
    def information_role(self):
        return self.public_clock.information_role

    def public_features(self):
        return self.public_clock.public_features()

    def public_manifest(self):
        return self.public_clock.public_manifest()

    def private_features(self):
        return {'private_step_boundary_times_s':self.endpoint_times_s,
            'private_step_parent_command_indices':self.parent_command_indices,
            'requested_output_private_step_indices':self.output_step_indices,
            'held_handle_forces_world_N':self.handle_forces_world_N,
            'held_handle_torques_world_Nm':self.handle_torques_world_Nm,
            'held_nodal_forces_world_N':self.nodal_forces_world_N,
            'held_body_acceleration_world_m_s2':self.body_acceleration_world_m_s2}


def repeat_for_integration(clock, *, integration_hz):
    """Repeat exact held arrays over integer substeps, with no waveform access.

    Coarser or nonaligned integrator clocks are rejected. Output times remain
    members of the public grid; support endpoint transforms are intentionally
    absent because a finer support law cannot be inferred from held forces.
    Returned times are compiled requests, not measured accumulated state times.
    """
    if type(clock) is not FixedActuatorClock:
        raise TypeError('Explicit FixedActuatorClock required')
    hz = _integer(integration_hz, 'integration_hz')
    if hz < clock.actuator_hz or hz % clock.actuator_hz:
        raise ValueError('Integration Hz must be an integer multiple of the fixed actuator Hz')
    factor = hz//clock.actuator_hz
    count = (len(clock.endpoint_times_s)-1)*factor
    if count >= 2**53 or count > np.iinfo(np.int64).max-1:
        raise ValueError('Private integer grid exceeds exact float64/int64 indexing')
    endpoints = np.arange(count+1,dtype=np.float64)/hz
    # No drift correction: public boundaries must already be exact grid members.
    if not np.array_equal(endpoints[::factor],clock.endpoint_times_s):
        raise ValueError('Private grid does not preserve exact public endpoints')
    parent = np.repeat(np.arange(len(clock.endpoint_times_s)-1,dtype=np.int64),factor)
    output_indices = np.searchsorted(clock.endpoint_times_s,clock.output_times_s)*factor
    if not np.array_equal(endpoints[output_indices],clock.output_times_s):
        raise ValueError('Requested output times do not align with private endpoints')
    repeated = {key:_immutable(np.repeat(getattr(clock,key),factor,axis=0), (count,*getattr(clock,key).shape[1:]), key)
        for key in ['handle_forces_world_N','handle_torques_world_Nm','nodal_forces_world_N','body_acceleration_world_m_s2']}
    # Make all clock/index arrays immutable too; no input/public array is changed.
    freeze = lambda value:np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)
    return RefinedActuatorClock(public_clock=clock, integration_hz=hz, substeps_per_public_interval=factor,
        endpoint_times_s=freeze(endpoints), parent_command_indices=freeze(parent),
        output_step_indices=freeze(output_indices), **repeated)


def integrated_commands(endpoint_times_s, *, handle_forces_world_N, handle_torques_world_Nm,
                        nodal_forces_world_N, body_acceleration_world_m_s2):
    """Independent rectangular integration; body acceleration integrates to m/s.

    This is a command-clock diagnostic, not measured solver impulse/work. No
    material density or response trajectory is accepted or inferred.
    """
    times = _immutable(endpoint_times_s,(None,), 'integration endpoint times')
    if len(times)<2 or np.any(np.diff(times)<=0):
        raise ValueError('Strictly increasing finite command intervals required')
    T = len(times)-1
    forces = _immutable(handle_forces_world_N,(T,None,3),'handle forces')
    arrays = {'handle_force_impulse_Ns':forces,
        'handle_pure_torque_impulse_Nms':_immutable(handle_torques_world_Nm,forces.shape,'handle torques'),
        'nodal_force_impulse_Ns':_immutable(nodal_forces_world_N,(T,None,3),'nodal forces'),
        'body_acceleration_integral_m_s':_immutable(body_acceleration_world_m_s2,(T,3),'body acceleration')}
    return {key:np.tensordot(np.diff(times),value,axes=(0,0)) for key,value in arrays.items()}
