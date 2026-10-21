"""Offline adapter tests: no physical solves, GPU, or response reads."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest

from experiments import campaign_hypothesis_dynamics as runner
from experiments.campaign_fixed_actuator_clock import FixedActuatorClock,KNOWN_ROLE


def command_clock(*,nodes=0):
    rng=np.random.default_rng(541)
    forces=rng.normal(size=(360,2,3))*.01
    forces[239]=0;forces[240]=[[-.01,.002,.004],[.008,-.004,.003]]
    return FixedActuatorClock(actuator_hz=120,endpoint_times_s=np.arange(361)/120,
        output_times_s=np.arange(61)/20,handle_forces_world_N=forces,
        handle_torques_world_Nm=np.zeros((360,2,3)),
        nodal_forces_world_N=rng.normal(size=(360,nodes,3))*.001,
        body_acceleration_world_m_s2=np.zeros((360,3)),information_role=KNOWN_ROLE)


class CoreCallSpy:
    def __init__(self):self.calls=[];self.points=np.zeros((4,3))
    def step(self,state,h,**kwargs):
        self.calls.append((state.time_s,h,kwargs))
        # Model the core's actual accumulated state clock, never replace it by nominal.
        return SimpleNamespace(time_s=state.time_s+h),{'converged':True}


@pytest.mark.parametrize('rate',[120,240,480])
@pytest.mark.parametrize('nodes',[0,4])
def test_actual_core_calls_repeat_exact_held_commands_and_measure_state_clock(rate,nodes):
    clock=command_clock(nodes=nodes);public=clock.public_manifest()
    refined=runner.validate_clock(clock,rate,4,2)
    spy=CoreCallSpy();state=SimpleNamespace(time_s=0.)
    for step in range(3*rate):state,_=runner.advance(spy,state,refined,step)
    factor=rate//120
    assert refined.public_manifest()==public
    assert abs(state.time_s-3)<1e-12
    assert len(spy.calls)==3*rate
    for i,(time,h,args) in enumerate(spy.calls):
        parent=i//factor
        assert h==1/rate and abs(time-i/rate)<1e-12
        assert args['handle_forces'].tobytes()==clock.handle_forces_world_N[parent].tobytes()
        expected=np.zeros((4,3)) if nodes==0 else clock.nodal_forces_world_N[parent]
        assert args['nodal_forces'].tobytes()==expected.tobytes()
        assert args['maximum_iterations']==250
        assert args['relative_tolerance']==1e-7 and args['absolute_tolerance_N']==1e-10
    public_impulse=np.sum(clock.handle_forces_world_N.astype(np.longdouble)/120,axis=0)
    actual_impulse=sum((args['handle_forces'].astype(np.longdouble)*np.longdouble(h)
        for _,h,args in spy.calls),start=np.zeros((2,3),dtype=np.longdouble))
    np.testing.assert_allclose(actual_impulse,public_impulse,atol=2e-18,rtol=3e-15)


@pytest.mark.parametrize('kind',['torque','acceleration','nodes','output','rate','role'])
def test_reject_unsupported_inputs_instead_of_dropping_or_resampling(kind):
    original=command_clock();args={key:getattr(original,key) for key in original.__dataclass_fields__}
    rate=120
    if kind=='torque':args['handle_torques_world_Nm']=np.ones((360,2,3))*.001
    elif kind=='acceleration':args['body_acceleration_world_m_s2']=np.ones((360,3))*.001
    elif kind=='nodes':args['nodal_forces_world_N']=np.zeros((360,3,3))
    elif kind=='output':args['output_times_s']=np.arange(31)/10
    elif kind=='rate':rate=180
    elif kind=='role':args['information_role']='candidate_force_hypothesis'
    with pytest.raises(ValueError):
        clock=FixedActuatorClock(**args);runner.validate_clock(clock,rate,4,2)


def test_bad_measured_clock_fails_before_or_after_core_call():
    refined=runner.validate_clock(command_clock(),120,4,2)
    spy=CoreCallSpy()
    with pytest.raises(ValueError,match='Measured solver time'):
        runner.advance(spy,SimpleNamespace(time_s=.01),refined,0)
    assert not spy.calls
    spy.step=lambda state,h,**kwargs:(SimpleNamespace(time_s=h+.01),{})
    with pytest.raises(ValueError,match='Measured endpoint'):
        runner.advance(spy,SimpleNamespace(time_s=0.),refined,0)


@pytest.mark.parametrize('metric_scale',[1.25,2.])
def test_metric_override_is_private_and_does_not_mutate_cached_factor(metric_scale):
    class Original:
        def __init__(self):self.factor=runner.torch.eye(2,dtype=runner.torch.float64)
        def preconditioner(self,state,h):return self.factor
    a,b=Original(),Original();before=a.factor.clone()
    runner.install_metric(a,metric_scale)
    assert 'preconditioner' in a.__dict__ and 'preconditioner' not in b.__dict__
    np.testing.assert_array_equal(a.factor,before)
    np.testing.assert_allclose(a.preconditioner(None,1/120),before*np.sqrt(metric_scale),rtol=0,atol=0)
    np.testing.assert_array_equal(b.preconditioner(None,1/120),before)


@pytest.mark.parametrize('boundary',[0,1,2,3])
def test_realized_prior_support_and_complete_inertia_reach_constructor_without_physical_solve(monkeypatch,boundary):
    from experiments import campaign_hypothesis_inputs as inputs
    public=copy.deepcopy(inputs.public_hardware_contexts()[0])
    # Exercise full non-diagonal inertia and nonzero COM, even though sourceH is diagonal.
    R=np.array([[np.cos(.31),-np.sin(.31),0],[np.sin(.31),np.cos(.31),0],[0,0,1.]])
    I=np.asarray(public['handles'][0]['com_inertia_body_kg_m2'])
    public['handles'][0]['com_inertia_body_kg_m2']=(R@I@R.T).tolist()
    public['handles'][0]['com_offset_body_m']=[.001,-.002,.001]
    public['public_hardware_sha256']=inputs.digest({k:v for k,v in public.items() if k!='public_hardware_sha256'})
    hardware=inputs.known_hardware_from_public(public);prior=inputs.make_prior(public)
    state,_=inputs.conditioned_BC_sample(prior,boundary,np.random.default_rng(140+boundary))
    spec=inputs.realize_hypothesis(public,state,prior)
    called={}
    def constructor(*args,**kwargs):
        called.update(args=args,kwargs=kwargs);return SimpleNamespace(points=hardware.points)
    monkeypatch.setattr(runner,'NonlinearDynamics',constructor)
    _,validated=runner.build_model(hardware,spec)
    assert validated==spec
    params=called['kwargs']
    np.testing.assert_array_equal(params['fixture_nodes'],spec['fixture_nodes'])
    assert len(params['fixture_nodes'])==[1,16,18,16][boundary]
    assert params['boundary_kind']==spec['boundary_kind']
    for j,h in enumerate(hardware.handles):
        np.testing.assert_array_equal(params['load_patches'][j],h.ordered_patch_nodes)
        np.testing.assert_array_equal(params['handle_inertias'][j],h.com_inertia_body_kg_m2)
        np.testing.assert_array_equal(params['handle_com_offsets'][j],h.com_offset_body_m)
        assert params['handle_masses'][j]==h.mass_kg
    material=spec['material']
    assert called['args'][2:]==(material['youngs_modulus_Pa'],material['poissons_ratio'],material['density_kg_m3'])
    assert params['drag']==material['body_linear_drag_s_inverse']
    if boundary==3:
        nodes=np.array(spec['fixture_nodes'])
        for rate in [120,240,480]:
            accumulated=0.
            for step in range(3*rate):
                accumulated+=1/rate
                expected=hardware.points[nodes]+inputs.support_translation(spec,accumulated)
                np.testing.assert_array_equal(params['fixture_target'](accumulated),expected)
            assert abs(accumulated-3)<1e-12
        np.testing.assert_array_equal(params['fixture_velocity'](0),np.zeros((16,3)))
        # Analytic private endpoint law differs from holding the public endpoint target.
        assert np.max(abs(params['fixture_target'](1/480)-params['fixture_target'](1/120)))>0
    else:
        assert params['fixture_target'] is None and params['fixture_velocity'] is None


def test_realized_material_or_support_corruption_is_rejected_before_constructor(monkeypatch):
    from experiments import campaign_hypothesis_inputs as inputs
    public=inputs.public_hardware_contexts()[0];prior=inputs.make_prior(public)
    state,_=inputs.conditioned_BC_sample(prior,1,np.random.default_rng(763))
    spec=inputs.realize_hypothesis(public,state,prior)
    called=[];monkeypatch.setattr(runner,'NonlinearDynamics',lambda *a,**k:called.append(True))
    for corrupt in ['material','fixture','capacity']:
        bad=copy.deepcopy(spec)
        if corrupt=='material':bad['material']['youngs_modulus_Pa']*=1.01
        if corrupt=='fixture':bad['fixture_nodes']=bad['fixture_nodes'][:-1]
        if corrupt=='capacity':bad['capacities']['S']=1
        with pytest.raises(ValueError):runner.build_model(inputs.known_hardware_from_public(public),bad)
    assert not called
