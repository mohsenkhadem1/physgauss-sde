import numpy as np
import pytest
from .scene import PhysicalHypothesis,Intervention,Camera,build

def test_force_conservation_and_material_fixed():
    h=PhysicalHypothesis(12000);a=Intervention('force_pulse',gravity=(0,0,0))
    sim,s=build(h,a);f,_,_,_,_=sim.evaluate(s.u,s.rigid,.05)
    np.testing.assert_allclose(f[0].sum(0),a.force_N,atol=1e-10)
    assert h.young_pa==12000

def test_camera_is_projection_only():
    sim,s=build(PhysicalHypothesis(12000),Intervention('drop'))
    before=sim.momentum(s).copy();C=np.eye(4);C[2,3]=-1
    p=sim.models[0].rest+s.u[0];a=Camera(C,np.eye(3)).project(p)
    C[0,3]=.1;b=Camera(C,np.eye(3)).project(p)
    assert not np.allclose(a,b)
    np.testing.assert_array_equal(before,sim.momentum(s))

def test_force_work_matches_time_derivative_of_potential():
    sim,s=build(PhysicalHypothesis(12000),Intervention('force_pulse',gravity=(0,0,0)))
    t=.035;eps=1e-7
    derivative=(sim.evaluate(s.u,s.rigid,t+eps)[2]-sim.evaluate(s.u,s.rigid,t-eps)[2])/(2*eps)
    np.testing.assert_allclose(derivative,sim.evaluate(s.u,s.rigid,t)[3],rtol=1e-6)

def test_release_removes_attachments():
    sim,s=build(PhysicalHypothesis(12000),Intervention('lift_release'))
    sim.controls(0,.15)
    assert all(np.abs(A.data).sum()==0 for A in sim.models[0].anchor_matrices)

def test_invalid_queries_rejected():
    with pytest.raises(ValueError):Intervention('invalid')
    with pytest.raises(ValueError):PhysicalHypothesis(-1)
