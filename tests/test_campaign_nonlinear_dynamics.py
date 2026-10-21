"""Mechanics controls for an explicitly independent nonlinear dynamic model."""
import json
import numpy as np
import pytest
import torch

from experiments.campaign_nonlinear_dynamics import NonlinearDynamics, InvalidConfiguration
from experiments.campaign_nonlinear_fem import rotation_matrix


def unit_tet():
    return np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),np.array([[0,1,2,3]])


def centered_cube():
    # One central node and eight corners; 12 positively oriented tetrahedra.
    points=np.array([[0.,0.,0.],[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                     [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=float)*.1
    faces=[[1,2,3,4],[5,6,7,8],[1,2,6,5],[2,3,7,6],[3,4,8,7],[4,1,5,8]]
    tets=[]
    for a,b,c,d in faces:
        for tri in [[a,b,c],[a,c,d]]:
            tet=[0,*tri]
            if np.linalg.det((points[tet[1:]]-points[tet[0]]).T)<0:
                tet[2],tet[3]=tet[3],tet[2]
            tets.append(tet)
    return points,np.array(tets)


def rigid_model(*,rotation=None,translation=None):
    points,tets=centered_cube()
    I=np.diag([.002,.003,.004])
    if rotation is not None:
        points=points@rotation.T;I=rotation@I@rotation.T
    if translation is not None:points=points+translation
    return NonlinearDynamics(points,tets,1500.,.3,20.,load_patches=[list(range(9))],
        load_nodes=[0],handle_masses=[.7],handle_inertias=[I])


def integrate(model,state,h,seconds,**kw):
    ledger=[]
    for _ in range(round(seconds/h)):
        state,row=model.step(state,h,**kw);ledger.append(row)
    return state,ledger


def test_zero_rest_and_all_exact_static_fixture_families_allow_rigid_null_modes():
    x,tets=unit_tet()
    for kind,nodes in [('free',[]),('point_pin',[0]),('fixed_patch',[0,1,2]),('two_patches',[0,1])]:
        m=NonlinearDynamics(x,tets,1000.,.3,600.,boundary_kind=kind,fixture_nodes=nodes)
        s=m.initial_state();out,row=m.step(s,.01)
        np.testing.assert_array_equal(out.positions,x)
        np.testing.assert_array_equal(out.velocities,0)
        assert row['nonlinear_history'][-1]['scaled_generalized_residual_norm_N']==0
        assert row['converged']
        # K is singular for a free body/pin; the physical dynamic mass metric is not.
        factor=m.preconditioner(s,.01)
        assert factor is not None and torch.diagonal(factor).min()>0


def test_free_rigid_translation_held_force_impulse_and_implicit_position_convention():
    m=rigid_model();v=np.array([.02,-.03,.01]);F=np.array([[.015,-.01,.02]])
    s=m.initial_state(velocities=np.tile(v,(9,1)),handle_linear_velocities=[v])
    h=.025;steps=8;x0=s.positions.numpy().copy();mass=m.total_mass
    for n in range(1,steps+1):
        s,row=m.step(s,h,handle_forces=F,relative_tolerance=1e-10,absolute_tolerance_N=1e-11)
        expected_v=v+n*h*F[0]/mass
        expected_d=n*h*v+(h*h*n*(n+1)/2)*F[0]/mass
        np.testing.assert_allclose(s.velocities,np.tile(expected_v,(9,1)),atol=2e-11,rtol=0)
        np.testing.assert_allclose(s.positions,x0+expected_d,atol=3e-12,rtol=0)
        np.testing.assert_allclose(row['linear_momentum_balance_error_Ns'],0,atol=1e-11)
        assert abs(row['energy_algebra_identity_residual_J'])<1e-14
    # All nine nodes share exact finite rigid distances, not penalty attachments.
    np.testing.assert_allclose(torch.cdist(s.positions,s.positions),torch.cdist(m.points,m.points),atol=2e-15,rtol=0)


def test_body_free_uniform_acceleration_uses_declared_nodal_mass_without_rigid_hardware():
    x,tets=unit_tet();m=NonlinearDynamics(x,tets,1200.,.32,600.)
    s=m.initial_state();g=np.array([.03,-.02,.01]);h=.02
    f=m.body_mass.numpy()[:,None]*g
    for n in range(1,5):
        s,row=m.step(s,h,nodal_forces=f,absolute_tolerance_N=1e-10)
        np.testing.assert_allclose(s.positions,x+h*h*n*(n+1)/2*g,atol=1e-12,rtol=0)
        np.testing.assert_allclose(s.velocities,np.tile(n*h*g,(4,1)),atol=2e-11,rtol=0)
        np.testing.assert_allclose(row['linear_momentum_balance_error_Ns'],0,atol=2e-11)


def test_finite_rotation_objectivity_with_translated_rotated_full_inertia():
    Q=rotation_matrix(torch.tensor([.7,-.4,.3],dtype=torch.float64)).numpy();offset=np.array([.3,-.5,.7])
    a=rigid_model();b=rigid_model(rotation=Q,translation=offset)
    omega=np.array([.4,.7,-.9]);v=np.array([.03,.02,-.01])
    va=v+np.cross(omega,a.points.numpy())
    vb=va@Q.T
    sa=a.initial_state(velocities=va,handle_linear_velocities=[v],handle_angular_velocities=[omega])
    sb=b.initial_state(velocities=vb,handle_linear_velocities=[v@Q.T],handle_angular_velocities=[omega@Q.T])
    F=np.array([[.003,-.002,.001]])
    for _ in range(4):
        sa,ra=a.step(sa,.01,handle_forces=F,relative_tolerance=1e-10,absolute_tolerance_N=1e-10)
        sb,rb=b.step(sb,.01,handle_forces=F@Q.T,relative_tolerance=1e-10,absolute_tolerance_N=1e-10)
    np.testing.assert_allclose(sb.positions,sa.positions.numpy()@Q.T+offset,atol=3e-10,rtol=0)
    np.testing.assert_allclose(sb.velocities,sa.velocities.numpy()@Q.T,atol=1e-8,rtol=0)
    assert rb['new_invariants']['chord_kinetic_energy_J']==pytest.approx(ra['new_invariants']['chord_kinetic_energy_J'],rel=1e-8)


def test_small_amplitude_limit_against_independent_linear_implicit_euler_recurrence():
    x,tets=unit_tet();m=NonlinearDynamics(x,tets,1200.,.3,600.,boundary_kind='fixed_patch',fixture_nodes=[0,1,2],drag=.2)
    s=m.initial_state();h=.02;node_force=np.zeros_like(x);node_force[3]=[2e-5,-1e-5,1e-5]
    free=[9,10,11];K=m.K.numpy()[np.ix_(free,free)]
    M=np.diag(np.repeat(float(m.body_mass[3]),3));u=np.zeros(3);v=np.zeros(3)
    system=K+M/(h*h)+.2*M/h
    for _ in range(10):
        rhs=M@(u+h*v)/(h*h)+.2*M@u/h+node_force[3]
        new_u=np.linalg.solve(system,rhs);v=(new_u-u)/h;u=new_u
        s,row=m.step(s,h,nodal_forces=node_force,relative_tolerance=1e-9,absolute_tolerance_N=1e-12)
    np.testing.assert_allclose(s.positions[3]-m.points[3],u,rtol=3e-5,atol=1e-12)
    np.testing.assert_array_equal(s.positions[:3],x[:3])


def test_moving_fixture_reaction_and_work_include_inertia_and_body_drag():
    x,tets=unit_tet();speed=np.array([.02,-.01,.03]);gamma=.7
    # All nodes prescribed: independent exact kinetic/work identity, no elasticity.
    target=lambda t:x+t*speed
    m=NonlinearDynamics(x,tets,1000.,.3,600.,fixture_nodes=list(range(4)),boundary_kind='moving_patch',
        fixture_target=target,fixture_velocity=lambda t:np.tile(speed,(4,1)),drag=gamma)
    s=m.initial_state();out,row=m.step(s,.02)
    np.testing.assert_allclose(out.positions,target(.02),atol=0,rtol=0)
    reaction=np.array(row['fixture_reactions_inertia_elastic_drag_minus_external_N'])
    np.testing.assert_allclose(reaction,gamma*m.body_mass.numpy()[:,None]*speed,atol=1e-9,rtol=0)
    assert row['support_work_J']==pytest.approx(.02*gamma*float(m.body_mass.sum())*(speed@speed),rel=1e-10)
    assert row['body_viscous_dissipation_J']==pytest.approx(row['support_work_J'],rel=1e-10)
    np.testing.assert_allclose(row['linear_momentum_balance_error_Ns'],0,atol=1e-10)
    np.testing.assert_allclose(row['endpoint_angular_momentum_discretization_defect_Nms'],0,atol=1e-10)
    assert abs(row['energy_algebra_identity_residual_J'])<1e-14
    # Accelerating boundary: verify the additional mass acceleration explicitly.
    acceleration=np.array([.2,-.1,.15]);h=.025
    target2=lambda t:x+.5*acceleration*t*t
    n=NonlinearDynamics(x,tets,1000.,.3,600.,fixture_nodes=list(range(4)),boundary_kind='moving_patch',
        fixture_target=target2,drag=gamma)
    sn=n.initial_state();nn,rr=n.step(sn,h)
    expected=n.body_mass.numpy()[:,None]*(.5*acceleration+gamma*.5*acceleration*h)
    np.testing.assert_allclose(rr['fixture_reactions_inertia_elastic_drag_minus_external_N'],expected,atol=1e-9,rtol=0)
    assert rr['support_work_J']>rr['body_viscous_dissipation_J']
    np.testing.assert_allclose(rr['linear_momentum_balance_error_Ns'],0,atol=1e-10)
    assert abs(rr['energy_algebra_identity_residual_J'])<1e-14


def test_free_damped_translation_matches_independent_momentum_sink_and_energy_loss():
    x,tets=unit_tet();gamma=.6;h=.025;v=np.array([.02,-.01,.03])
    m=NonlinearDynamics(x,tets,1200.,.3,600.,drag=gamma)
    s=m.initial_state(velocities=np.tile(v,(4,1)));mass=float(m.body_mass.sum())
    for _ in range(5):
        initial=m.invariants(s);v=v/(1+gamma*h)
        s,row=m.step(s,h,relative_tolerance=1e-10,absolute_tolerance_N=1e-10)
        np.testing.assert_allclose(s.velocities,np.tile(v,(4,1)),atol=2e-12,rtol=0)
        np.testing.assert_allclose(row['linear_momentum_balance_error_Ns'],0,atol=1e-10)
        np.testing.assert_allclose(row['body_drag_impulse_Ns'],h*gamma*mass*v,atol=1e-13,rtol=0)
        loss=row['new_invariants']['energy_J']-initial['energy_J']
        assert loss==pytest.approx(-row['body_viscous_dissipation_J']-row['implicit_velocity_increment_loss_J'],abs=1e-13)


def test_nonzero_COM_offset_continuous_kinetics_and_free_discrete_momentum():
    x,tets=centered_cube();mass=.7;I=np.diag([.002,.003,.004]);com=np.array([.03,.01,-.02])
    m=NonlinearDynamics(x,tets,1500.,.3,20.,load_patches=[list(range(9))],load_nodes=[0],
        handle_masses=[mass],handle_inertias=[I],handle_com_offsets=[com])
    v=np.array([.03,-.02,.01]);omega=np.array([.7,-.4,.5]);body_v=v+np.cross(omega,x)
    s=m.initial_state(velocities=body_v,handle_linear_velocities=[v],handle_angular_velocities=[omega])
    expected_T=.5*np.sum(m.body_mass.numpy()[:,None]*body_v**2)
    vcom=v+np.cross(omega,com)
    expected_T+=.5*mass*(vcom@vcom)+.5*omega@I@omega
    expected_P=(m.body_mass.numpy()[:,None]*body_v).sum(0)+mass*vcom
    initial=m.invariants(s)
    assert initial['chord_kinetic_energy_J']==pytest.approx(expected_T,rel=1e-13)
    np.testing.assert_allclose(initial['point_linear_momentum_Ns'],expected_P,atol=1e-15,rtol=0)
    for _ in range(3):
        s,row=m.step(s,.01,relative_tolerance=1e-10,absolute_tolerance_N=1e-10)
        np.testing.assert_allclose(row['new_invariants']['point_linear_momentum_Ns'],expected_P,atol=1e-10,rtol=0)


def test_free_asymmetric_rigid_spin_energy_and_angular_momentum_errors_refine(record_property):
    errors=[]
    for h in [.04,.02,.01]:
        m=rigid_model();omega=np.array([.7,1.1,-.8]);vel=np.cross(omega,m.points.numpy())
        s=m.initial_state(velocities=vel,handle_angular_velocities=[omega])
        initial=m.invariants(s)
        final,rows=integrate(m,s,h,.4,relative_tolerance=1e-10,absolute_tolerance_N=1e-10)
        last=m.invariants(final)
        energy_error=abs(last['energy_J']-initial['energy_J'])/initial['energy_J']
        L0=np.array(initial['chord_angular_momentum_about_world_origin_Nms']);L1=np.array(last['chord_angular_momentum_about_world_origin_Nms'])
        angular_error=np.linalg.norm(L1-L0)/np.linalg.norm(L0)
        errors.append((energy_error,angular_error))
        np.testing.assert_allclose(last['point_linear_momentum_Ns'],0,atol=2e-9)
        assert all(abs(r['energy_algebra_identity_residual_J'])<1e-13 for r in rows)
    record_property('free_spin_refinement',json.dumps({'steps_s':[.04,.02,.01],
        'duration_s':.4,'relative_energy_and_chord_angular_momentum_errors':errors}))
    for coarse,fine in zip(errors[:-1],errors[1:]):
        assert fine[0]<.75*coarse[0] and fine[1]<.75*coarse[1],errors
    # Nonzero finite-step errors are disclosed; this is not an exact conservation test.
    assert errors[-1][0]>1e-7 and errors[-1][1]>1e-7


def test_invalid_J_hardware_kinematics_and_iteration_caps_fail_without_repair():
    x,tets=unit_tet();m=NonlinearDynamics(x,tets,1000.,.3,600.)
    bad=x.copy();bad[[1,2]]=bad[[2,1]]
    with pytest.raises(InvalidConfiguration):m.initial_state(positions=bad)
    with pytest.raises(ValueError,match='inertia'):
        NonlinearDynamics(x,tets,1000.,.3,600.,load_patches=[[0,1,2]],load_nodes=[0],handle_masses=[1.],handle_inertias=[])
    with pytest.raises(ValueError,match='triangle'):
        NonlinearDynamics(x,tets,1000.,.3,600.,load_patches=[[0,1,2]],load_nodes=[0],handle_masses=[1.],handle_inertias=[[1,1,4]])
    with pytest.raises(ValueError,match='Overlapping'):
        NonlinearDynamics(x,tets,1000.,.3,600.,fixture_nodes=[0],boundary_kind='point_pin',load_patches=[[0,1,2]],load_nodes=[0],handle_masses=[1.],handle_inertias=[[1,1,1]])
    with pytest.raises(ValueError,match='Positive finite'):
        m.step(m.initial_state(),float('nan'))
    with pytest.raises(RuntimeError,match='iteration cap'):
        m.step(m.initial_state(),.01,nodal_forces=np.ones_like(x),maximum_iterations=0)
    fixed=NonlinearDynamics(x,tets,1000.,.3,600.,fixture_nodes=[0],boundary_kind='point_pin')
    with pytest.raises(ValueError,match='fixture velocity'):
        fixed.initial_state(velocities=np.ones_like(x))
