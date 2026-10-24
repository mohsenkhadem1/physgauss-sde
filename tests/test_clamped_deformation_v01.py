import numpy as np
import pytest
import torch
from experiments import campaign_clamped_deformation_v01 as s

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2);yield;torch.set_num_threads(old)


def test_prospective_design_geometry_clamp_and_loads():
    X,tets,faces,fixed,free=s.geometry()
    assert X.shape==(20,3) and tets.shape==(24,4) and len(fixed)==4 and len(free)==48
    assert not fixed.flags.writeable and not free.flags.writeable
    determinants=np.linalg.det((X[tets[:,1:]]-X[tets[:,:1]]).transpose(0,2,1))
    assert np.all(determinants>0)
    assert abs((determinants/6).sum()-.24*.08*.08)<1e-16
    # Nodesare notcospherical: overdeterminedcenter equations inconsistent.
    center=np.linalg.lstsq(2*(X[1:]-X[0]),(X[1:]**2).sum(1)-(X[0]**2).sum(),rcond=None)[0]
    assert np.ptp(((X-center)**2).sum(1))>1e-3
    p=s.programs(1/40)
    np.testing.assert_allclose(p['qualification']['bending'][-1].sum(0),[0,0,.03])
    np.testing.assert_allclose(p['qualification']['stretching'][-1].sum(0),[.24,0,0])
    assert s.design()['deformation_gates']['minimum_nonrigid_RMS_m']==100e-6
    np.testing.assert_allclose(s.physical(np.array([[0,0],[1,1]])),[[2000,5],[8000,30]])


def test_independent_numpy_assembly_and_backward_euler_lowload():
    forward=s.ClampedForward();ku,m,w=s.independent_numpy_assembly()
    np.testing.assert_allclose(forward.Kunit,ku,rtol=2e-14,atol=1e-16)
    np.testing.assert_allclose(forward.mass,m,rtol=2e-14,atol=1e-16)
    np.testing.assert_allclose(forward.diagonal,w,rtol=2e-14,atol=1e-16)
    h=1/40;forces=np.stack(list(s.programs(h)['qualification'].values()))*1e-3
    theta=np.array([[3700.,11.],[6200.,21.]])
    actual=forward.trajectory(theta,forces,h)
    assert actual.shape==(2,2,81,20,3)
    assert np.count_nonzero(actual[:,:,:,forward.fixed])==0
    for i,th in enumerate(theta):
        for j,force in enumerate(forces):
            np.testing.assert_allclose(actual[i,j],s.independent_numpy_trajectory(th,force,h),rtol=5e-10,atol=2e-14)


def test_prospective_causality_and_exact_prefix_truncation_lowload():
    forward=s.ClampedForward();h=1/40;p=s.programs(h);mask=s.time_grid(h)<=s.PREFIX_END
    for force in p['future'].values():assert np.array_equal(force[mask],p['prefix'][mask])
    theta=np.array([[4700.,14.]])
    forces=np.stack(list(p['future'].values()))*1e-3
    allstates=forward.trajectory(theta,forces,h);end=round(s.PREFIX_END/h)+1
    assert np.array_equal(allstates[:,0,:end],allstates[:,1,:end])
    assert np.array_equal(allstates[:,0,:end],allstates[:,2,:end])
    truncated=forward.trajectory(theta,forces,h,stop_time=s.PREFIX_END)
    np.testing.assert_array_equal(truncated[:,0],allstates[:,0,:end])
    assert truncated.shape[2]==33


def test_gradient_and_infinitesimal_metrics_analytical_fields():
    forward=s.ClampedForward();X=forward.X
    A=np.array([[.01,.002,0],[.002,-.003,.001],[0,.001,.004]])
    u=(X@A.T)[None];g=s.displacement_gradient(forward,u)
    np.testing.assert_allclose(g,np.broadcast_to(A,g.shape),atol=2e-17)
    metrics=s.deformation_metrics(forward,u)
    np.testing.assert_allclose(metrics['maximum_linear_strain'],.01,atol=1e-17)
    np.testing.assert_allclose(metrics['minimum_J'],np.linalg.det(np.eye(3)+A),atol=1e-15)
    omega=np.array([[0,-.01,.02],[.01,0,-.03],[-.02,.03,0]])
    rigid=(X@omega.T+[.001,.002,.003])[None]
    assert s.deformation_metrics(forward,rigid)['nonrigid_RMS_m'][0]<1e-17


def test_invalid_inputs_fail_closed_without_physics():
    with pytest.raises(ValueError):s.physical(np.zeros((1,3)))
    with pytest.raises(ValueError):s.time_grid(.031)
    with pytest.raises(ValueError):s.physical(np.array([[1.1,.5]]))
