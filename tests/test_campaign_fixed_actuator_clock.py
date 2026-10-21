"""Known-command clock properties only; no recipe/response/simulator access."""
from dataclasses import FrozenInstanceError
import inspect

import numpy as np
import pytest

from experiments.campaign_fixed_actuator_clock import (
    FixedActuatorClock, integrated_commands, repeat_for_integration, KNOWN_ROLE)
from experiments.campaign_full_hardware_context import ForceProgram


def inputs(hz=20, intervals=20, seed=104):
    rng=np.random.default_rng(seed)
    return dict(actuator_hz=hz,endpoint_times_s=np.arange(intervals+1,dtype=float)/hz,
        output_times_s=np.arange(intervals+1,dtype=float)[::2]/hz,
        handle_forces_world_N=rng.normal(size=(intervals,2,3)),
        handle_torques_world_Nm=rng.normal(size=(intervals,2,3)),
        nodal_forces_world_N=rng.normal(size=(intervals,3,3)),
        body_acceleration_world_m_s2=rng.normal(size=(intervals,3)),information_role=KNOWN_ROLE)


def commands(value):
    return {key:getattr(value,key) for key in ['handle_forces_world_N','handle_torques_world_Nm',
        'nodal_forces_world_N','body_acceleration_world_m_s2']}


@pytest.mark.parametrize('hz,count',[(5,10),(20,40),(960,192)])
@pytest.mark.parametrize('factor',[1,2,3,7,16])
def test_random_held_commands_roles_public_bytes_and_impulses_survive_private_refinement(hz,count,factor):
    payload=inputs(hz,count,seed=hz+factor);clock=FixedActuatorClock(**payload)
    public_before=clock.public_manifest();schedule=repeat_for_integration(clock,integration_hz=hz*factor)
    assert schedule.information_role==KNOWN_ROLE and schedule.public_manifest()==public_before
    for name,value in clock.public_features().items():
        assert schedule.public_features()[name].dtype==value.dtype
        assert schedule.public_features()[name].tobytes()==value.tobytes()
    np.testing.assert_array_equal(schedule.endpoint_times_s[::factor],clock.endpoint_times_s)
    np.testing.assert_array_equal(schedule.endpoint_times_s[schedule.output_step_indices],clock.output_times_s)
    # Check command selection at every new interval, including both sides of jumps.
    for step,parent in enumerate(schedule.parent_command_indices):
        assert parent==step//factor
        for name,array in commands(schedule).items():np.testing.assert_array_equal(array[step],getattr(clock,name)[parent])
    public_integrals=integrated_commands(clock.endpoint_times_s,**commands(clock))
    private_integrals=integrated_commands(schedule.endpoint_times_s,**commands(schedule))
    eps=np.finfo(float).eps;n=len(schedule.parent_command_indices);gamma=(n+2)*eps/(1-(n+2)*eps)
    # Independent longdouble rectangular integration, including cancellation.
    for (output_name,actual),(input_name,source) in zip(private_integrals.items(),commands(clock).items()):
        dt=np.diff(clock.endpoint_times_s).astype(np.longdouble)
        expected=np.tensordot(dt,source.astype(np.longdouble),axes=(0,0))
        absolute=np.tensordot(dt,np.abs(source).astype(np.longdouble),axes=(0,0))
        bound=8*gamma*np.maximum(absolute,np.finfo(float).tiny)
        assert np.all(np.abs(actual.astype(np.longdouble)-expected)<=bound)
        assert np.all(np.abs(public_integrals[output_name].astype(np.longdouble)-expected)<=bound)
    # The first temporal moment of a held command must also remain unchanged.
    def moment(times,values):
        t=times.astype(np.longdouble);return np.tensordot(np.diff(t)*(t[1:]+t[:-1])/2,values.astype(np.longdouble),axes=(0,0))
    np.testing.assert_allclose(moment(schedule.endpoint_times_s,schedule.handle_forces_world_N),
        moment(clock.endpoint_times_s,clock.handle_forces_world_N),atol=3e-16,rtol=3e-15)


def test_rectangular_jump_and_couple_are_held_without_waveform_or_torque_conversion():
    payload=inputs(hz=4,intervals=4)
    payload['handle_forces_world_N'][:]=0
    payload['handle_forces_world_N'][1,0]=[1,-2,4]
    payload['handle_forces_world_N'][1,1]=[-1,2,-4]
    payload['handle_torques_world_Nm'][:]=[.1,.2,.3]
    payload['nodal_forces_world_N'][:]=0
    payload['body_acceleration_world_m_s2'][:]=[0,0,2]
    clock=FixedActuatorClock(**payload);private=repeat_for_integration(clock,integration_hz=12)
    assert not private.handle_forces_world_N[:3].any() and not private.handle_forces_world_N[6:].any()
    np.testing.assert_array_equal(private.handle_forces_world_N.sum(1),0)
    np.testing.assert_array_equal(private.handle_forces_world_N[3:6,0],np.tile([1,-2,4],(3,1)))
    result=integrated_commands(private.endpoint_times_s,**commands(private))
    np.testing.assert_allclose(result['handle_force_impulse_Ns'][0],[.25,-.5,1],atol=1e-16,rtol=0)
    np.testing.assert_allclose(result['handle_pure_torque_impulse_Nms'],np.tile([.1,.2,.3],(2,1)),atol=1e-16,rtol=0)
    np.testing.assert_allclose(result['body_acceleration_integral_m_s'],[0,0,2],atol=1e-16,rtol=0)
    assert not result['nodal_force_impulse_Ns'].any()


def test_latent_relabelling_can_change_private_hz_but_cannot_enter_public_api():
    payload=inputs(hz=960,intervals=96);clock=FixedActuatorClock(**payload)
    signatures=[inspect.signature(FixedActuatorClock),inspect.signature(FixedActuatorClock.from_force_program),inspect.signature(repeat_for_integration)]
    assert all(not any(p.kind==inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()) for signature in signatures)
    labels=[{'boundary':'pin','E':1e3},{'boundary':'patch','E':1e6},{'boundary':'moving','E':5}]
    public=[]
    for dummy,private_hz in zip(labels,[960,1920,6720]):
        public.append(repeat_for_integration(clock,integration_hz=private_hz).public_manifest())
        with pytest.raises(TypeError):FixedActuatorClock(**payload,material=dummy)
        with pytest.raises(TypeError):repeat_for_integration(clock,integration_hz=private_hz,boundary=dummy)
    assert public[0]==public[1]==public[2]
    assert not any(word in repr(public[0]) for word in ['boundary_type','youngs_modulus','recipe_sha256'])


def test_typed_known_force_program_copies_values_and_role_but_no_hidden_source_labels():
    payload=inputs();program=ForceProgram(held_handle_forces_world_N=payload['handle_forces_world_N'],
        held_handle_torques_world_Nm=payload['handle_torques_world_Nm'],held_nodal_forces_world_N=payload['nodal_forces_world_N'],
        held_body_acceleration_world_m_s2=payload['body_acceleration_world_m_s2'],
        information_role=KNOWN_ROLE,source_sha256={'dummy_material_label':'a'*64})
    clock=FixedActuatorClock.from_force_program(actuator_hz=20,endpoint_times_s=payload['endpoint_times_s'],
        output_times_s=payload['output_times_s'],force_program=program)
    assert clock.information_role==program.information_role
    for name,value in commands(clock).items():np.testing.assert_array_equal(value,payload[name])
    assert 'dummy_material_label' not in repr(clock.public_manifest())
    program.information_role='candidate_force_hypothesis'
    with pytest.raises(ValueError):FixedActuatorClock.from_force_program(actuator_hz=20,endpoint_times_s=payload['endpoint_times_s'],
        output_times_s=payload['output_times_s'],force_program=program)
    with pytest.raises(TypeError):FixedActuatorClock.from_force_program(actuator_hz=20,endpoint_times_s=payload['endpoint_times_s'],
        output_times_s=payload['output_times_s'],force_program={})


@pytest.mark.parametrize('private_hz',[10,30,39,0,-20,40.,True])
def test_coarser_fractional_or_noninteger_private_rates_fail(private_hz):
    with pytest.raises(ValueError):repeat_for_integration(FixedActuatorClock(**inputs()),integration_hz=private_hz)


@pytest.mark.parametrize('change',[
    {'actuator_hz':True},{'actuator_hz':20.},{'actuator_hz':0},{'actuator_hz':10**100},
    {'information_role':'candidate_force_hypothesis'},
    {'endpoint_times_s':np.array([0.,.05,.11])},
    {'endpoint_times_s':np.arange(21)/20+.1},
    {'endpoint_times_s':np.array([0.,np.nan])},
    {'endpoint_times_s':np.zeros(21)},
    {'output_times_s':np.array([0.,.025])}, # finer solver time is not a public endpoint
    {'output_times_s':np.array([0.,1.1])},
    {'output_times_s':np.array([.1,.1])},
    {'handle_forces_world_N':np.ones((19,2,3))},
    {'handle_torques_world_Nm':np.ones((20,1,3))},
    {'body_acceleration_world_m_s2':np.full((20,3),np.inf)},
    {'nodal_forces_world_N':np.ones((20,3,2))},
])
def test_ambiguous_clocks_roles_or_command_dimensions_fail(change):
    with pytest.raises(ValueError):FixedActuatorClock(**(inputs()|change))


def test_boundary_last_bits_are_rejected_without_snapping_and_inputs_are_immutable():
    payload=inputs();changed=payload['endpoint_times_s'].copy();changed[3]=np.nextafter(changed[3],np.inf)
    with pytest.raises(ValueError):FixedActuatorClock(**(payload|{'endpoint_times_s':changed}))
    clock=FixedActuatorClock(**payload);private=repeat_for_integration(clock,integration_hz=60)
    original=clock.handle_forces_world_N.copy();payload['handle_forces_world_N'][:]=0
    np.testing.assert_array_equal(clock.handle_forces_world_N,original)
    for value in list(clock.public_features().values())+list(private.private_features().values()):
        assert not value.flags.writeable
        with pytest.raises(ValueError):value.flags.writeable=True
    with pytest.raises(FrozenInstanceError):clock.actuator_hz=1


def test_float32_commands_promote_exactly_and_lossy_integers_are_rejected():
    payload=inputs();source=payload['handle_forces_world_N'].astype(np.float32)
    clock=FixedActuatorClock(**(payload|{'handle_forces_world_N':source}))
    np.testing.assert_array_equal(clock.handle_forces_world_N,source.astype(np.float64))
    lossy=np.full(source.shape,2**53+1,dtype=np.int64)
    with pytest.raises(ValueError,match='exactly'):FixedActuatorClock(**(payload|{'handle_forces_world_N':lossy}))


def test_free_body_no_handles_or_nodal_commands_still_has_explicit_acceleration():
    payload=inputs(hz=10,intervals=5);payload['handle_forces_world_N']=np.zeros((5,0,3))
    payload['handle_torques_world_Nm']=np.zeros((5,0,3));payload['nodal_forces_world_N']=np.zeros((5,0,3))
    private=repeat_for_integration(FixedActuatorClock(**payload),integration_hz=70)
    assert private.handle_forces_world_N.shape==(35,0,3) and private.body_acceleration_world_m_s2.shape==(35,3)
