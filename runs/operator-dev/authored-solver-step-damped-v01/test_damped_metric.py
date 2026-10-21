"""Private optimizer metric isolation/quadratic solve only; no physics call."""
import torch
from experiments.campaign_solver_step_damped import install_half_direction


def test_half_direction_is_local_spd_metric_scaling_without_class_or_cache_edits():
    class Model:
        def __init__(self):self.factor=torch.linalg.cholesky(torch.tensor([[3.,.2],[.2,2.]],dtype=torch.float64))
        def preconditioner(self,state,h):return self.factor
    original_method=Model.preconditioner;changed=Model();untouched=Model();saved=changed.factor.clone()
    install_half_direction(changed);L=changed.preconditioner(None,1.)
    assert Model.preconditioner is original_method
    assert torch.equal(changed.factor,saved) and torch.equal(untouched.preconditioner(None,1.),saved)
    gradient=torch.tensor([.4,-.7],dtype=torch.float64)[:,None]
    old=torch.cholesky_solve(gradient,saved);new=torch.cholesky_solve(gradient,L)
    torch.testing.assert_close(new,.5*old,atol=2e-17,rtol=2e-16)
    assert bool(torch.linalg.eigvalsh(L@L.T).min()>0)
