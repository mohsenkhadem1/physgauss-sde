import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]/'runs/operator-dev/newton-runtime-v01/env/lib/python3.12/site-packages'))
import numpy as np
import pytest
import torch
from experiments.campaign_c38_verified_warp_solver import checked_linear_residual,IndependentlyCheckedWarpCGMixin
from experiments.campaign_p2_torch import LinearFailure
from experiments.campaign_p2_supported_preconditioner import StreamingSupportedP2
from experiments.campaign_c38_supported_grading import SupportedBoundaryGradedGeometry


def test_uses_original_inexact_CG_bound_not_final_static_absolute_tolerance():
    rhs=torch.ones((2,3),dtype=torch.float64);mask=torch.zeros_like(rhs,dtype=torch.bool)
    row=checked_linear_residual(rhs,rhs+.00001,mask,.001)
    assert row['passed'] and row['independent_original_Torch_HVP_true_R_N']>1e-10
    assert row['original_CG_acceptance_bound_N']==.002
    with pytest.raises(LinearFailure):checked_linear_residual(rhs,rhs+.1,mask,.001)
    with pytest.raises(ValueError):checked_linear_residual(rhs,rhs*float('nan'),mask,.001)


def test_same_returned_Warp_direction_passes_independent_CPU_Torch_operator():
    torch.set_num_threads(2)
    class Operator(IndependentlyCheckedWarpCGMixin,StreamingSupportedP2):pass
    m=SupportedBoundaryGradedGeometry((4,2,2),100000.,'both_ends',0.)
    op=Operator(m,'cpu',4000,8,support_kind='both_ends')
    rng=np.random.default_rng(91);u=torch.tensor(rng.normal(0,1e-6,m.rest.shape),dtype=torch.float64);u[op.mask]=0
    cache=op.evaluate(u);rhs=torch.tensor(rng.normal(0,.001,m.rest.shape),dtype=torch.float64);rhs[op.mask]=0
    direction,k=op.cg(cache,rhs,0.,rtol=1e-9,atol=1e-12)
    assert k>0 and op.independent_linear_checks[-1]['passed']
    np.testing.assert_array_equal(direction.numpy()[m.mask],0.)
