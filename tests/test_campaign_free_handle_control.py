import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'experiments/campaign_free_handle_control.py'
spec=importlib.util.spec_from_file_location('free_control',path);control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)


def test_exact_held_force_and_kick_drift_are_distinct_known_conventions():
    forces=np.tile([2.,0,0],(4,1));dt=np.array([.1,.2,.1,.3]);mass=2.
    result=control.free_held_reference([.3,.4,.5],[.2,0,0],mass,forces,dt)
    times=np.r_[0.,np.cumsum(dt)]
    np.testing.assert_allclose(result['positions_exact_piecewise_constant_m'][:,0],.3+.2*times+.5*times**2,atol=1e-15)
    np.testing.assert_allclose(result['velocities_m_s'][:,0],.2+times,atol=1e-15)
    assert result['positions_kick_drift_m'][-1,0]-result['positions_exact_piecewise_constant_m'][-1,0]==pytest.approx(.5*np.sum(dt**2))
    np.testing.assert_allclose(result['impulse_Ns'],[1.4,0,0])


def test_analytic_continuous_pulse_impulse_and_post_force_free_flight():
    mass=.001;result=control.continuous_pulse_reference(np.array([0.,.3,1.65,2.,3.]),[0,0,0],mass)
    np.testing.assert_array_equal(result['positions_m'][:2],np.zeros((2,3)))
    np.testing.assert_allclose(result['impulse_Ns'],[.00675,.00675,.003375],atol=1e-16)
    np.testing.assert_allclose(result['velocities_m_s'][-3:],np.tile(np.array([6.75,6.75,3.375]),(3,1)),atol=1e-14)
    np.testing.assert_allclose(result['positions_m'][-1]-result['positions_m'][-2],[6.75,6.75,3.375],atol=1e-14)


def test_float32_commands_and_actual_clock_impulse_are_distinct_from_nominal():
    for hz in [120,960]:
        midpoints,commands=control.command_schedule(hz)
        assert commands.shape==(3*hz,3) and commands.dtype==np.float32
        np.testing.assert_array_equal(commands[:,0],commands[:,1])
        np.testing.assert_array_equal(commands[:,2],commands[:,0]*np.float32(.5))
        np.testing.assert_allclose(commands.astype(float).sum(0)/hz,[.00675,.00675,.003375],atol=2e-10)
        assert np.all(commands[(midpoints<.3)|(midpoints>1.65)]==0)
        dt=float(np.float32(1/hz))
        result=control.free_held_reference([0,0,0],[0,0,0],.001,commands,dt)
        np.testing.assert_allclose(result['velocities_m_s'][-1]*.001,result['impulse_Ns'],atol=1e-15)


def test_invalid_reference_mass_or_clock_rejected():
    with pytest.raises(ValueError):control.free_held_reference([0,0,0],[0,0,0],0,[[1,0,0]],.1)
    with pytest.raises(ValueError):control.free_held_reference([0,0,0],[0,0,0],1,[[1,0,0]],-.1)
    with pytest.raises(ValueError):control.command_schedule(0)
