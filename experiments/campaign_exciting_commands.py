"""Two prospective, truth-independent rich known world-force interventions.

No material, support, recipe, geometry, response or hardware state is accepted.
The exact float64 120 Hz held values are the public intervention. Private
refinement must repeat those values through repeat_for_integration, never
reevaluate the carrier at a finer time. Command diagnostics qualify no physics.
"""
from __future__ import annotations

import hashlib
from types import MappingProxyType
import numpy as np

try:
    from campaign_fixed_actuator_clock import FixedActuatorClock,KNOWN_ROLE,repeat_for_integration
except ImportError:
    from experiments.campaign_fixed_actuator_clock import FixedActuatorClock,KNOWN_ROLE,repeat_for_integration


PROGRAM_IDS=('exciting_world_a_v01','exciting_world_b_v01')
PROGRAM_SEEDS=MappingProxyType(dict(zip(PROGRAM_IDS,(2026102101,2026102102))))
PUBLIC_PROGRAM_FORCE_SHA256=MappingProxyType({
    'exciting_world_a_v01':'8c2ac76b3b5dac14fd19c4ae46b1b9f66c7f757ad9ee31736ad27df027d4281a',
    'exciting_world_b_v01':'d0360cfbc4d3c81ff18050668b88123f61726c6835391127d2981400807a02e2'})
ACTUATOR_HZ=120
SECONDS=3
OUTPUT_HZ=20
ACTIVE_INTERVAL_S=(.15,2.55)
CARRIER_FREQUENCIES_HZ=(.30,.61,.92,1.23,1.54,1.85)
CHANNEL_SIGNS=(1.,-1.,1.,-1.,1.,-1.)
PUBLIC_ANALYTIC_FORCE_NORM_BOUND_N=.02
COMPONENT_AMPLITUDE_N=PUBLIC_ANALYTIC_FORCE_NORM_BOUND_N/np.sqrt(3.)
MAXIMUM_GRAM_CONDITION=100.
MINIMUM_GRAM_EIGENVALUE_RATIO=.01


def _parameters(program_id):
    if program_id not in PROGRAM_IDS:
        raise ValueError('One of the two frozen rich world-force program IDs is required')
    seed=PROGRAM_SEEDS[program_id]
    phases=np.random.Generator(np.random.PCG64(seed)).uniform(-np.pi,np.pi,6)
    return seed,phases


def _held_forces(program_id,midpoint_times_s):
    """Pure absolute-time carrier evaluation, used ONLY on the public grid."""
    _,phases=_parameters(program_id)
    t=np.asarray(midpoint_times_s,dtype=np.float64)
    lo,hi=ACTIVE_INTERVAL_S
    u=np.clip((t-lo)/(hi-lo),0.,1.)
    window=np.where((t>lo)&(t<hi),np.sin(np.pi*u)**2,0.)
    signal=np.sin(2*np.pi*t[:,None]*np.asarray(CARRIER_FREQUENCIES_HZ)+phases)
    signal*=np.asarray(CHANNEL_SIGNS)
    return np.ascontiguousarray((COMPONENT_AMPLITUDE_N*window[:,None]*signal).reshape(-1,2,3))


def make_exciting_commands(program_id,*,force_capacity=2):
    """Return the same immutable public clock for every physical hypothesis.

    Capacity is the sole hardware-shaped argument and must be exactly two.
    Empty nodal-force capacity declares no external nodal loading; a dynamics
    adapter can allocate its own all-zero nodal array without altering commands.
    """
    if isinstance(force_capacity,(bool,np.bool_)) or not isinstance(force_capacity,(int,np.integer)) or force_capacity!=2:
        raise ValueError('Exactly two explicit force slots are required')
    _parameters(program_id)
    count=ACTUATOR_HZ*SECONDS
    endpoints=np.arange(count+1,dtype=np.float64)/ACTUATOR_HZ
    midpoint=(np.arange(count,dtype=np.float64)+.5)/ACTUATOR_HZ
    return FixedActuatorClock(actuator_hz=ACTUATOR_HZ,endpoint_times_s=endpoints,
        output_times_s=endpoints[::ACTUATOR_HZ//OUTPUT_HZ],
        handle_forces_world_N=_held_forces(program_id,midpoint),
        handle_torques_world_Nm=np.zeros((count,2,3),dtype=np.float64),
        nodal_forces_world_N=np.zeros((count,0,3),dtype=np.float64),
        body_acceleration_world_m_s2=np.zeros((count,3),dtype=np.float64),
        information_role=KNOWN_ROLE)


def refine_exciting_commands(clock,*,integration_hz):
    """Only the predeclared 120/240/480 Hz exact held-value repeats."""
    if isinstance(integration_hz,(bool,np.bool_)) or not isinstance(integration_hz,(int,np.integer)) or integration_hz not in (120,240,480):
        raise ValueError('Only 120, 240 and 480 Hz integration schedules are declared')
    if type(clock) is not FixedActuatorClock:
        raise TypeError('An already-compiled FixedActuatorClock is required')
    if clock.actuator_hz!=ACTUATOR_HZ:
        raise ValueError('The public intervention must remain at 120 Hz')
    return repeat_for_integration(clock,integration_hz=integration_hz)


def describe_exciting_commands(program_id,*,force_capacity=2):
    """Compact physical force-Gram and exact held-clock integral diagnostics.

    Gram is integral F_flat(t) F_flat(t)^T dt, units N^2 s, in channel order
    handle0 xyz then handle1 xyz. Its conditioning concerns command excitation,
    not observability or physical parameter identifiability after deformation.
    """
    clock=make_exciting_commands(program_id,force_capacity=force_capacity)
    seed,phases=_parameters(program_id)
    f=clock.handle_forces_world_N
    widths=np.diff(clock.endpoint_times_s)
    midpoints=(clock.endpoint_times_s[:-1]+clock.endpoint_times_s[1:])/2
    flattened=f.reshape(len(f),6)
    gram=flattened.T@(widths[:,None]*flattened)
    eigenvalues=np.linalg.eigvalsh(gram)
    condition=float(eigenvalues[-1]/eigenvalues[0])
    ratio=float(eigenvalues[0]/eigenvalues[-1])
    return {'program_id':program_id,'seed':seed,'phase_generator':'NumPy PCG64 uniform[-pi,pi)',
        'phases_rad':phases.tolist(),'carrier_frequencies_Hz':list(CARRIER_FREQUENCIES_HZ),
        'channel_signs':list(CHANNEL_SIGNS),'channel_order':'handle0 xyz, handle1 xyz; known world forces at handle origins',
        'active_interval_s':list(ACTIVE_INTERVAL_S),'window':'sin^2(pi*(t-.15)/2.4) inside active interval; zero outside; C1 at endpoints',
        'spectral_scope':'Carrier frequencies are <=2 Hz; compact window adds spectral sidebands, so no strict band-limit claim.',
        'component_amplitude_N':float(COMPONENT_AMPLITUDE_N),
        'analytic_per_handle_force_norm_bound_N':PUBLIC_ANALYTIC_FORCE_NORM_BOUND_N,
        'maximum_held_force_norm_per_handle_N':np.linalg.norm(f,axis=-1).max(0).tolist(),
        'physical_force_gram_N2s':gram.tolist(),'physical_force_gram_eigenvalues_N2s':eigenvalues.tolist(),
        'force_gram_rank':int(np.linalg.matrix_rank(gram)),'force_gram_condition':condition,
        'force_gram_minimum_eigenvalue_ratio':ratio,
        'prospective_command_condition_gate':{'maximum_condition':MAXIMUM_GRAM_CONDITION,
            'minimum_eigenvalue_ratio':MINIMUM_GRAM_EIGENVALUE_RATIO,
            'passed':bool(condition<=MAXIMUM_GRAM_CONDITION and ratio>=MINIMUM_GRAM_EIGENVALUE_RATIO)},
        'held_force_impulse_Ns':np.einsum('t,thc->hc',widths,f).tolist(),
        'held_force_absolute_component_impulse_Ns':np.einsum('t,thc->hc',widths,np.abs(f)).tolist(),
        'held_force_first_time_moment_Ns2':np.einsum('t,thc->hc',widths*midpoints,f).tolist(),
        'held_force_values_sha256':hashlib.sha256(f.tobytes()).hexdigest(),
        'force_precision':'float64 public midpoint-held values; no float32 requantization',
        'clock':clock.public_manifest(),'same_program_across_all_materials_BCs_geometries_hardware':True,
        'response_or_latent_label_inputs':False,'physical_execution_performed':False,
        'causality':'Predetermined time-only known intervention; no response feedback or latent-condition branch.',
        'selection':'Amplitude, carriers, phases, window and condition criterion fixed before physical execution; no response-based selection.',
        'physical_convergence_or_dataset_admission_established':False}
