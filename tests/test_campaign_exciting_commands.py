"""Prospective force-only checks; no response files or physical execution."""
import hashlib
import inspect
import numpy as np
import pytest
from experiments.campaign_exciting_commands import (
    PROGRAM_IDS,PROGRAM_SEEDS,PUBLIC_PROGRAM_FORCE_SHA256,
    PUBLIC_ANALYTIC_FORCE_NORM_BOUND_N,COMPONENT_AMPLITUDE_N,
    CARRIER_FREQUENCIES_HZ,ACTIVE_INTERVAL_S,make_exciting_commands,
    describe_exciting_commands,refine_exciting_commands,_held_forces)


@pytest.mark.parametrize('program_id',PROGRAM_IDS)
def test_frozen_force_capacity_clock_roles_and_zero_other_channels(program_id):
    clock=make_exciting_commands(program_id)
    assert clock.actuator_hz==120 and clock.information_role=='known_actuator_command'
    np.testing.assert_array_equal(clock.endpoint_times_s,np.arange(361,dtype=np.float64)/120)
    np.testing.assert_array_equal(clock.output_times_s,clock.endpoint_times_s[::6])
    assert len(clock.output_times_s)==61 and clock.output_times_s[-1]==3
    for value,shape in [(clock.handle_forces_world_N,(360,2,3)),(clock.handle_torques_world_Nm,(360,2,3)),
                        (clock.nodal_forces_world_N,(360,0,3)),(clock.body_acceleration_world_m_s2,(360,3))]:
        assert value.shape==shape and value.dtype==np.float64 and not value.flags.writeable
    assert not np.any(clock.handle_torques_world_Nm) and not np.any(clock.nodal_forces_world_N) and not np.any(clock.body_acceleration_world_m_s2)
    assert hashlib.sha256(clock.handle_forces_world_N.tobytes()).hexdigest()==PUBLIC_PROGRAM_FORCE_SHA256[program_id]
    midpoint=(np.arange(360)+.5)/120
    outside=(midpoint<=ACTIVE_INTERVAL_S[0])|(midpoint>=ACTIVE_INTERVAL_S[1])
    assert not np.any(clock.handle_forces_world_N[outside])
    assert np.all(np.linalg.norm(clock.handle_forces_world_N,axis=-1)<=PUBLIC_ANALYTIC_FORCE_NORM_BOUND_N)
    assert max(CARRIER_FREQUENCIES_HZ)<=2 and len(set(CARRIER_FREQUENCIES_HZ))==6
    np.testing.assert_allclose(np.sqrt(3)*COMPONENT_AMPLITUDE_N,PUBLIC_ANALYTIC_FORCE_NORM_BOUND_N,atol=1e-17,rtol=0)


@pytest.mark.parametrize('program_id',PROGRAM_IDS)
def test_independent_physical_gram_integrals_and_condition_gate(program_id):
    clock=make_exciting_commands(program_id);report=describe_exciting_commands(program_id)
    gram=np.zeros((6,6));impulse=np.zeros((2,3));first=np.zeros((2,3))
    for i,force in enumerate(clock.handle_forces_world_N):
        lo,hi=clock.endpoint_times_s[i:i+2];width=hi-lo
        gram+=np.outer(force.ravel(),force.ravel())*width
        impulse+=width*force;first+=(hi*hi-lo*lo)/2*force
    np.testing.assert_allclose(report['physical_force_gram_N2s'],gram,atol=2e-19,rtol=2e-14)
    eigen=np.linalg.eigvalsh(gram)
    assert len(eigen)==6 and eigen[0]/eigen[-1]>=.01 and eigen[-1]/eigen[0]<=100
    assert report['force_gram_rank']==6 and report['prospective_command_condition_gate']['passed']
    np.testing.assert_allclose(report['held_force_impulse_Ns'],impulse,atol=2e-17,rtol=0)
    np.testing.assert_allclose(report['held_force_first_time_moment_Ns2'],first,atol=2e-17,rtol=0)
    assert report['seed']==PROGRAM_SEEDS[program_id]
    assert report['held_force_values_sha256']==PUBLIC_PROGRAM_FORCE_SHA256[program_id]
    assert report['same_program_across_all_materials_BCs_geometries_hardware']
    assert not report['response_or_latent_label_inputs'] and not report['physical_execution_performed']


@pytest.mark.parametrize('program_id',PROGRAM_IDS)
@pytest.mark.parametrize('hz',[120,240,480])
def test_private_refinement_repeats_exact_values_without_requantization(program_id,hz):
    clock=make_exciting_commands(program_id);private=refine_exciting_commands(clock,integration_hz=hz)
    factor=hz//120
    np.testing.assert_array_equal(private.handle_forces_world_N,np.repeat(clock.handle_forces_world_N,factor,axis=0))
    np.testing.assert_array_equal(private.endpoint_times_s[::factor],clock.endpoint_times_s)
    np.testing.assert_array_equal(private.endpoint_times_s[private.output_step_indices],clock.output_times_s)
    assert private.public_manifest()==clock.public_manifest()
    dt=np.diff(private.endpoint_times_s)
    middle=(private.endpoint_times_s[1:]+private.endpoint_times_s[:-1])/2
    report=describe_exciting_commands(program_id)
    impulse=(dt[:,None,None]*private.handle_forces_world_N).sum(0)
    first=(dt[:,None,None]*middle[:,None,None]*private.handle_forces_world_N).sum(0)
    np.testing.assert_allclose(impulse,report['held_force_impulse_Ns'],atol=2e-17,rtol=0)
    # Repeated floating-point sums have different lengths; held values and
    # endpoint membership above remain bit-exact, integrals agree to roundoff.
    np.testing.assert_allclose(first,report['held_force_first_time_moment_Ns2'],atol=1e-16,rtol=0)


def test_public_invariance_no_latent_inputs_rng_mutation_or_future_dependence():
    assert tuple(inspect.signature(make_exciting_commands).parameters)==('program_id','force_capacity')
    np.random.seed(42);before=np.random.get_state()
    a=make_exciting_commands(PROGRAM_IDS[0]);b=make_exciting_commands(PROGRAM_IDS[0])
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    assert a.public_manifest()==b.public_manifest()
    assert not np.array_equal(a.handle_forces_world_N,make_exciting_commands(PROGRAM_IDS[1]).handle_forces_world_N)
    midpoint=(np.arange(360)+.5)/120
    np.testing.assert_array_equal(_held_forces(PROGRAM_IDS[0],midpoint[:117]),a.handle_forces_world_N[:117])
    # Both value and first derivative of the continuous window vanish at the
    # edges; the actual executed command remains the separately frozen holds.
    edges=np.array([.15,2.55]);assert not np.any(_held_forces(PROGRAM_IDS[0],edges))
    near=_held_forces(PROGRAM_IDS[0],np.array([.15+1e-6,2.55-1e-6]))
    assert np.max(np.abs(near))<1e-12
    with pytest.raises(TypeError):make_exciting_commands(PROGRAM_IDS[0],material={'E':1000})
    with pytest.raises(ValueError):make_exciting_commands('legacy_push')
    for bad in [True,1,3,2.,'2']:
        with pytest.raises(ValueError):make_exciting_commands(PROGRAM_IDS[0],force_capacity=bad)
    for bad in [True,60,180,960]:
        with pytest.raises(ValueError):refine_exciting_commands(a,integration_hz=bad)
