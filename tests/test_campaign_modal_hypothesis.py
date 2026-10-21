"""Independent mechanics checks before fixed96 fit-only projection diagnosis."""
import numpy as np
import torch
from experiments.campaign_modal_hypothesis import FixedModalLinearDynamics,constitutive_parts
from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
from experiments.campaign_nonlinear_dynamics import NonlinearDynamics
from experiments.campaign_fixed_actuator_clock import FixedActuatorClock,KNOWN_ROLE
from test_campaign_nonlinear_dynamics import unit_tet,centered_cube


def clock(N,H=0):
    times=np.arange(361,dtype=float)/120;mid=(times[1:]+times[:-1])/2
    nodal=np.zeros((360,N,3));nodal[:,-1]=np.sin(np.pi*mid/3)[:,None]*np.array([.03,-.01,.02])
    return FixedActuatorClock(actuator_hz=120,endpoint_times_s=times,output_times_s=times[::6],
        handle_forces_world_N=np.zeros((360,H,3)),handle_torques_world_Nm=np.zeros((360,H,3)),
        nodal_forces_world_N=nodal,body_acceleration_world_m_s2=np.zeros((360,3)),information_role=KNOWN_ROLE)


def test_constitutive_split_matches_independent_affine_strain_energy():
    points,tets=unit_tet();physical=NonlinearDynamics(points,tets,1200.,.32,600.,fixture_nodes=[0],boundary_kind='point_pin')
    Ks,Kv=constitutive_parts(physical);G=np.array([[.1,.03,-.04],[.02,-.05,.01],[.07,.02,.08]])
    u=torch.tensor((points@G.T).flatten());eps=(G+G.T)/2
    np.testing.assert_allclose(.5*float(u@Ks@u),np.sum(eps**2)/6,atol=2e-17)
    np.testing.assert_allclose(.5*float(u@Kv@u),.5*np.trace(G)**2/6,atol=2e-17)
    mu=1200/(2*1.32);lam=1200*.32/(1.32*.36)
    torch.testing.assert_close(mu*Ks+lam*Kv,physical.K,atol=2e-12,rtol=1e-14)


def test_mass_orthonormal_basis_preserves_all_free_and_point_pin_rigid_zero_modes():
    points,tets=unit_tet()
    for kind,nodes,modes,zero_count in [('free',[],6,6),('point_pin',[0],3,3)]:
        full=LinearHypothesisDynamics(NonlinearDynamics(points,tets,1200.,.3,600.,fixture_nodes=nodes,boundary_kind=kind))
        modal=FixedModalLinearDynamics(full,mode_count=modes)
        assert modal.basis_diagnostics['known_rigid_zero_mode_count']==zero_count
        assert modal.basis_diagnostics['maximum_rigid_zero_mode_relative_projection_error']<1e-12
        assert modal.basis_diagnostics['M0_orthonormal_max_error']<2e-14


def test_full_modal_span_equals_full_linear_at_new_material_and_exact_moving_lift(monkeypatch):
    points,tets=unit_tet();nodes=[0,1,2];direction=np.array([.002,-.001,.003])
    target=lambda t:points[nodes]+np.sin(np.pi*t/3)**2*direction
    kwargs=dict(fixture_nodes=nodes,boundary_kind='moving_patch',fixture_target=target,
        fixture_velocity=lambda t:np.broadcast_to(np.pi/3*np.sin(2*np.pi*t/3)*direction,(3,3)))
    first=LinearHypothesisDynamics(NonlinearDynamics(points,tets,2000.,.25,700.,drag=.2,**kwargs))
    modal=FixedModalLinearDynamics(first,mode_count=3)
    second=LinearHypothesisDynamics(NonlinearDynamics(points,tets,8500.,.41,950.,drag=.8,**kwargs))
    monkeypatch.setattr(first.model,'step',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Nonlinear step forbidden')))
    material=dict(youngs_modulus_Pa=8500.,poissons_ratio=.41,density_kg_m3=950.,body_linear_drag_s_inverse=.8)
    reference=second.predict(clock(4));result=modal.predict(clock(4),material)
    torch.testing.assert_close(result['positions_m'],reference['positions_m'],atol=1e-14,rtol=1e-11)
    torch.testing.assert_close(result['velocities_point_chord_m_s'],reference['velocities_point_chord_m_s'],atol=2e-13,rtol=1e-9)


def test_complete_auxiliary_offdiagonal_COM_mass_is_projected_without_added_drag():
    points,tets=centered_cube();inertia=np.array([[.002,.0003,.0002],[.0003,.003,-.0001],[.0002,-.0001,.004]])
    physical=NonlinearDynamics(points,tets,1500.,.3,20.,load_patches=[list(range(9))],load_nodes=[0],
        handle_masses=[.7],handle_inertias=[inertia],handle_com_offsets=[[.02,-.03,.01]],drag=.4)
    full=LinearHypothesisDynamics(physical);modal=FixedModalLinearDynamics(full,mode_count=6);V=modal.basis
    torch.testing.assert_close(modal.Mh,V.T@full.Maux@V,atol=1e-15,rtol=1e-13)
    torch.testing.assert_close(20*modal.Mb+modal.Mh,V.T@full.M@V,atol=2e-15,rtol=1e-13)
    assert modal.basis_diagnostics['known_rigid_zero_mode_count']==6
    assert torch.linalg.matrix_norm(modal.Mh)>0
