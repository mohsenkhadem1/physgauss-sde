import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.object_dynamics import DynamicObjectBoundaryBackend,falling_cube_payload


def test_unsupported_body_falls_without_gauge_and_step_refines():
    d=falling_cube_payload(duration=.1,step=.01);errors=[]
    for sub in (2,4):
        b=DynamicObjectBoundaryBackend(d,substeps=sub,floor_mode='off')
        x,_=b.rollout(np.zeros(b.parameter_dim),np.arange(len(d['times'])))
        center=x.mean(1);exact=d['nodes'].mean(0)[2]-.5*9.81*d['times']**2
        errors.append(abs(center[-1,2]-exact[-1]))
        assert np.max(abs((x-x.mean(1)[:,None])-(d['nodes']-d['nodes'].mean(0))))<1e-9
        assert np.max(abs(center[:,:2]))<1e-10
        assert b.last_path_diagnostics['max_residual']<1e-7
    assert .45<errors[1]/errors[0]<.55


def test_requested_sampling_does_not_change_dynamic_path():
    d=falling_cube_payload(duration=.04);b=DynamicObjectBoundaryBackend(d,substeps=2,floor_mode='off');z=np.zeros(b.parameter_dim)
    full,_=b.rollout(z,np.arange(5));sparse,_=b.rollout(z,np.array([0,2,4]))
    np.testing.assert_array_equal(sparse,full[[0,2,4]])
    with pytest.raises(ValueError):b.rollout(z,np.array([2,1]))


def test_initial_state_is_not_equilibrium_shifted():
    d=falling_cube_payload(duration=.01);b=DynamicObjectBoundaryBackend(d,floor_mode='off')
    x,diag=b.rollout(np.zeros(b.parameter_dim),np.array([0]))
    np.testing.assert_allclose(x[0],d['nodes'],atol=1e-15)
    assert diag[0]['initial_state']
