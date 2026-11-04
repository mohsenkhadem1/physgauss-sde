import numpy as np
from scripts.gaussian_boundary_flow_v02.robust_dynamic_step import DynamicSolveTargetMixin,FiniteRangeAttachmentFEMTargetV2
from scripts.gaussian_boundary_flow_v02.object_dynamics import DynamicObjectBoundaryBackend,falling_cube_payload


class Fake:
    def solve(self,theta,frame,initial,tolerance,maxiter):
        return None,None,{'residual':tolerance*.7}


class Target(DynamicSolveTargetMixin,Fake):
    inertial_target=None


def test_target_only_changes_dynamic_inner_request():
    m=Target();assert m.solve(None,0,tolerance=1e-9)[2]['effective_solver_tolerance']==1e-9
    m.inertial_target=1
    d=m.solve(None,0,tolerance=1e-9)[2]
    assert d['effective_solver_tolerance']==1e-8
    assert d['residual']<1e-7


def test_no_force_change_freefall():
    d=falling_cube_payload(duration=.03)
    b=DynamicObjectBoundaryBackend(d,floor_mode='off',substeps=2)
    baseline,_=b.rollout(np.zeros(b.parameter_dim),np.arange(4))
    b.model=FiniteRangeAttachmentFEMTargetV2(d,device='cpu',field=False)
    b.base_weights=b.model.weights.clone()
    updated,diag=b.rollout(np.zeros(b.parameter_dim),np.arange(4))
    np.testing.assert_allclose(updated,baseline,atol=1e-11)
    assert max(q['residual'] or 0 for q in diag)<=1e-7


def test_safeguard_requires_force_and_small_correction():
    import torch
    import pytest
    from scripts.gaussian_boundary_flow_v02.robust_dynamic_step import SafeguardedDynamicSolveMixin
    class Mock(SafeguardedDynamicSolveMixin):
        inertial_target=1;rest=torch.zeros((1,1),dtype=torch.float64)
        ndof=1;device='cpu';dtype=torch.float64
        force=5e-8;stiffness=1e4
        def evaluate(self,x,theta,frame,tangent=False):
            return torch.tensor(0.),torch.full_like(x,self.force),torch.tensor([[self.stiffness]],dtype=self.dtype)
        def min_jacobian(self,x):return 1.
    m=Mock();_,_,d=m.solve(None,0,tolerance=1e-9)
    assert d['safeguarded_termination']
    assert d['newton_correction_max']<=1e-9
    m.force=2e-7
    with pytest.raises(RuntimeError):m.solve(None,0,tolerance=1e-9,maxiter=1)
    m.force=5e-8;m.stiffness=1.
    with pytest.raises(RuntimeError):m.solve(None,0,tolerance=1e-9,maxiter=1)
