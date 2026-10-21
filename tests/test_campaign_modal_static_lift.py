import numpy as np
import torch
from experiments.campaign_modal_hypothesis import FixedModalLinearDynamics
from experiments.campaign_modal_static_lift import translation_shapes,orthogonal_completion,complete_static_lift
from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
from experiments.campaign_nonlinear_dynamics import NonlinearDynamics
from test_campaign_nonlinear_dynamics import unit_tet,centered_cube
from test_campaign_modal_hypothesis import clock


def test_canonical_shape_cancels_uniform_fixture_elasticity_for_every_material():
    points,tets=unit_tet();full=LinearHypothesisDynamics(NonlinearDynamics(points,tets,1200.,.3,600.,fixture_nodes=[0],boundary_kind='point_pin'))
    plain=FixedModalLinearDynamics(full,mode_count=3);Q=translation_shapes(full);u=full.A@Q+plain.S
    torch.testing.assert_close(u,torch.eye(3,dtype=torch.float64).repeat(4,1),atol=0,rtol=0)
    torch.testing.assert_close(plain.Ks_full@u,torch.zeros_like(u),atol=1e-16,rtol=0)
    torch.testing.assert_close(plain.Kv_full@u,torch.zeros_like(u),atol=1e-16,rtol=0)


def test_two_pass_completion_rejects_dependent_columns_without_changing_old_basis():
    M=torch.diag(torch.tensor([2.,3.,5.],dtype=torch.float64));V=torch.tensor([[1/np.sqrt(2)],[0.],[0.]],dtype=torch.float64)
    original=V.clone();Q=torch.eye(3,dtype=torch.float64)
    completed,records=orthogonal_completion(V,Q,M)
    assert [r['added'] for r in records]==[False,True,True]
    torch.testing.assert_close(V,original,atol=0,rtol=0)
    torch.testing.assert_close(completed.T@M@completed,torch.eye(3,dtype=torch.float64),atol=4e-16,rtol=0)


def test_static_completion_restores_missing_moving_response_and_keeps_chord_lift_exact():
    points,tets=unit_tet();nodes=[0,1,2];direction=np.array([.002,-.001,.003]);target=lambda t:points[nodes]+np.sin(np.pi*t/3)**2*direction
    kwargs=dict(fixture_nodes=nodes,boundary_kind='moving_patch',fixture_target=target,
        fixture_velocity=lambda t:np.broadcast_to(np.pi/3*np.sin(2*np.pi*t/3)*direction,(3,3)))
    first=LinearHypothesisDynamics(NonlinearDynamics(points,tets,2000.,.25,700.,drag=.2,**kwargs))
    plain=FixedModalLinearDynamics(first,mode_count=1);old=plain.basis.clone();enriched=complete_static_lift(plain)
    assert plain.mode_count==1 and enriched.mode_count==3
    torch.testing.assert_close(plain.basis,old,atol=0,rtol=0)
    second=LinearHypothesisDynamics(NonlinearDynamics(points,tets,8500.,.41,950.,drag=.8,**kwargs))
    material=dict(youngs_modulus_Pa=8500.,poissons_ratio=.41,density_kg_m3=950.,body_linear_drag_s_inverse=.8)
    reference=second.predict(clock(4));result=enriched.predict(clock(4),material)
    torch.testing.assert_close(result['positions_m'],reference['positions_m'],atol=2e-14,rtol=1e-11)
    torch.testing.assert_close(result['velocities_point_chord_m_s'],reference['velocities_point_chord_m_s'],atol=2e-13,rtol=1e-9)


def test_complete_COM_auxiliary_mass_translation_survives_completion():
    points,tets=centered_cube();I=np.array([[.002,.0003,.0002],[.0003,.003,-.0001],[.0002,-.0001,.004]])
    full=LinearHypothesisDynamics(NonlinearDynamics(points,tets,1500.,.3,20.,load_patches=[list(range(9))],
        load_nodes=[0],handle_masses=[.7],handle_inertias=[I],handle_com_offsets=[[.02,-.03,.01]]))
    plain=FixedModalLinearDynamics(full,mode_count=6);enriched=complete_static_lift(plain);Q=translation_shapes(full)
    assert enriched.mode_count==6 and not any(r['added'] for r in enriched.basis_diagnostics['completion_records'])
    torch.testing.assert_close(full.JA@Q,torch.eye(3,dtype=torch.float64).repeat(6,1),atol=1e-16,rtol=0)
    torch.testing.assert_close(enriched.Mh,enriched.basis.T@full.Maux@enriched.basis,atol=1e-15,rtol=1e-13)
