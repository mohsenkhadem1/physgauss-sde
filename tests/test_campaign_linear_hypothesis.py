"""Small independent linear controls, without captured trajectories or labels."""
import numpy as np
import pytest
import torch
from experiments.campaign_nonlinear_dynamics import NonlinearDynamics
from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
from test_campaign_nonlinear_dynamics import unit_tet,centered_cube


@pytest.mark.parametrize('kind,nodes',[('free',[]),('point_pin',[0]),('fixed_patch',[0,1,2]),('two_patches',[0,1])])
def test_zero_rest_including_singular_stiffness_and_one_cached_factor(kind,nodes,monkeypatch):
    points,tets=unit_tet()
    physical=NonlinearDynamics(points,tets,1200.,.3,600.,fixture_nodes=nodes,boundary_kind=kind,drag=.3)
    # Any accidental delegation to the finite-strain time step must fail.
    monkeypatch.setattr(physical,'step',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Nonlinear step forbidden')))
    model=LinearHypothesisDynamics(physical);state=model.initial_state()
    for _ in range(3):state=model.step(state,.02)
    np.testing.assert_array_equal(state.positions,points)
    np.testing.assert_array_equal(state.velocities,np.zeros_like(points))
    assert model.factorizations==1
    assert torch.all(torch.diag(model.factor(.02))>0)
    if kind in ['free','point_pin']:
        assert float(torch.linalg.eigvalsh(model.Kr).abs().min())<1e-10


def test_implicit_euler_matches_independent_three_dof_linear_system_and_superposition():
    points,tets=unit_tet();physical=NonlinearDynamics(points,tets,1200.,.3,600.,fixture_nodes=[0,1,2],boundary_kind='fixed_patch',drag=.4)
    model=LinearHypothesisDynamics(physical);h=.025
    M=np.eye(3)*float(physical.body_mass[3]);C=.4*M;K=physical.K.numpy()[9:12,9:12]
    lhs=M/h**2+C/h+K;u=np.zeros(3);v=np.zeros(3)
    states=[model.initial_state() for _ in range(3)]
    fa=np.array([.03,-.02,.01]);fb=np.array([-.01,.005,.02])
    for _ in range(5):
        new=np.linalg.solve(lhs,fa+M@(u/h**2+v/h)+C@u/h);v=(new-u)/h;u=new
        for j,force in enumerate([fa,fb,fa+fb]):
            f=np.zeros_like(points);f[3]=force;states[j]=model.step(states[j],h,nodal_forces=f)
        np.testing.assert_allclose(states[0].coordinates.numpy(),u,atol=1e-18,rtol=1e-11)
        np.testing.assert_allclose(states[0].positions[3].numpy(),points[3]+u,atol=2e-16,rtol=0)
        np.testing.assert_allclose(states[0].velocities[3],v,atol=5e-16,rtol=1e-11)
        torch.testing.assert_close(states[2].coordinates,states[0].coordinates+states[1].coordinates,atol=1e-17,rtol=1e-12)
    assert model.factorizations==1


def test_exact_moving_dirichlet_lift_and_independent_rhs():
    points,tets=unit_tet();nodes=[0,1,2];direction=np.array([.002,-.001,.003]);h=.02;gamma=.6
    target=lambda t:points[nodes]+np.sin(np.pi*t/3)**2*direction
    physical=NonlinearDynamics(points,tets,1200.,.3,600.,fixture_nodes=nodes,boundary_kind='moving_patch',
        fixture_target=target,fixture_velocity=lambda t:np.tile(np.pi/3*np.sin(2*np.pi*t/3)*direction,(3,1)),drag=gamma)
    model=LinearHypothesisDynamics(physical);state=model.initial_state()
    M=np.eye(3)*float(physical.body_mass[3]);K=physical.K.numpy()[9:12,9:12];Kfc=physical.K.numpy()[9:12,:9]
    C=gamma*M;lhs=M/h**2+C/h+K;u=np.zeros(3);v=np.zeros(3)
    for i in range(1,5):
        time=i*h;b=(target(time)-points[nodes]).ravel()
        new=np.linalg.solve(lhs,M@(u/h**2+v/h)+C@u/h-Kfc@b);v=(new-u)/h;u=new
        old=state;state=model.step(state,h)
        np.testing.assert_allclose(state.positions[nodes],target(state.time_s),atol=0,rtol=0)
        np.testing.assert_allclose(state.coordinates.numpy(),u,atol=1e-18,rtol=1e-11)
        np.testing.assert_allclose(state.positions[3].numpy(),points[3]+u,atol=2e-16,rtol=0)
        np.testing.assert_allclose(state.velocities[nodes],(target(state.time_s)-target(old.time_s))/h,atol=2e-15,rtol=1e-11)


def test_full_offdiagonal_com_inertia_and_offset_mass_metric_are_retained():
    points,tets=centered_cube();mass=.7;offset=np.array([.02,-.03,.01])
    inertia=np.array([[.002,.0003,.0002],[.0003,.003,-.0001],[.0002,-.0001,.004]])
    physical=NonlinearDynamics(points,tets,1500.,.3,20.,load_patches=[list(range(9))],load_nodes=[0],
        handle_masses=[mass],handle_inertias=[inertia],handle_com_offsets=[offset],drag=.2)
    model=LinearHypothesisDynamics(physical);dofs=physical.handle_dofs[0];aux=model.Maux.numpy()[np.ix_(dofs,dofs)]
    x,y,z=offset;S=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    expected=np.block([[mass*np.eye(3),-mass*S],[mass*S,inertia+mass*((offset@offset)*np.eye(3)-np.outer(offset,offset))]])
    np.testing.assert_allclose(aux,expected,atol=2e-17,rtol=1e-12)
    assert abs(aux[3,4])>0 and np.linalg.norm(aux[:3,3:])>0
    # Rigid body strain modes are zero; full mass still makes dynamics SPD.
    state=model.step(model.initial_state(),.02,handle_forces=[[.02,-.01,.03]])
    assert torch.isfinite(state.coordinates).all() and state.handle_rotation_coordinates.abs().sum()>0
    assert state.handle_rotation_coordinates.shape==(1,3)
