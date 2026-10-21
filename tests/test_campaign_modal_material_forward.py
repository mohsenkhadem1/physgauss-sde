import numpy as np
import torch
from experiments.campaign_modal_material_forward import BatchedModalMaterialForward
from experiments.campaign_modal_hypothesis import FixedModalLinearDynamics
from experiments.campaign_modal_static_lift import complete_static_lift
from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
from experiments.campaign_nonlinear_dynamics import NonlinearDynamics
from test_campaign_nonlinear_dynamics import unit_tet
from test_campaign_modal_hypothesis import clock


def model():
    points,tets=unit_tet();nodes=[0,1,2];direction=np.array([.002,-.001,.003])
    full=LinearHypothesisDynamics(NonlinearDynamics(points,tets,2000.,.25,700.,drag=.2,
        fixture_nodes=nodes,boundary_kind='moving_patch',fixture_target=lambda t:points[nodes]+np.sin(np.pi*t/3)**2*direction,
        fixture_velocity=lambda t:np.broadcast_to(np.pi/3*np.sin(2*np.pi*t/3)*direction,(3,3))))
    return complete_static_lift(FixedModalLinearDynamics(full,mode_count=1))


def test_batched_physical_queries_match_independent_scalar_recurrence_and_cache_geometry():
    modal=model();batch=BatchedModalMaterialForward(modal);commands=clock(4)
    theta=torch.tensor([[8500.,950.,.41,.8],[3000.,600.,.25,.2]],dtype=torch.float64)
    result=batch.predict(commands,theta)
    for i,values in enumerate(theta):
        material=dict(zip(['youngs_modulus_Pa','density_kg_m3','poissons_ratio','body_linear_drag_s_inverse'],values))
        scalar=modal.predict(commands,material)
        torch.testing.assert_close(result['positions_m'][i],scalar['positions_m'],atol=2e-14,rtol=1e-11)
    assert len(batch._cache)==1 and result['positions_m'].shape==(2,61,4,3)


def test_four_physical_parameter_autograd_matches_central_finite_differences():
    modal=model();batch=BatchedModalMaterialForward(modal);commands=clock(4)
    theta=torch.tensor([[6000.,800.,.33,.6]],dtype=torch.float64,requires_grad=True)
    def objective(values):return (batch.predict(commands,values)['positions_m']-modal.full.points).square().sum()
    gradient=torch.autograd.grad(objective(theta),theta)[0]
    differences=[]
    for j,step in enumerate([.06,.008,3.3e-6,6e-6]):
        delta=torch.zeros_like(theta);delta[0,j]=step
        differences.append(float((objective(theta.detach()+delta)-objective(theta.detach()-delta))/(2*step)))
    assert torch.isfinite(gradient).all() and torch.all(gradient.abs()>1e-14)
    np.testing.assert_allclose(gradient.detach().numpy()[0],differences,atol=2e-12,rtol=2e-5)
