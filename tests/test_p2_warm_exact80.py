import numpy as np
import torch
from experiments.campaign_p2_geometry import P2Geometry
from experiments.campaign_p2_exact_quadrature import ExactBulkP2Geometry
from experiments.campaign_p2_streaming import StreamingTwoLevelP2
from experiments.campaign_p2_torch import TorchP2
from experiments.run_p2_warm_exact80 import solve_warm


def test_warm_quadrature_continuation_matches_independent_from_rest_exact80():
    torch.set_num_threads(2)
    q3model=P2Geometry((1,1,1));q3=StreamingTwoLevelP2(q3model).solve()['positions'].numpy()
    exact=ExactBulkP2Geometry((1,1,1));q3copy=q3.copy();solver,answer,initial_R=solve_warm(exact,q3)
    # Independent initialization and monolithic quadrature implementation.
    independent=TorchP2(exact).solve(force_tolerance=1e-10)
    np.testing.assert_array_equal(q3,q3copy)
    assert initial_R>1e-10
    assert float(torch.linalg.vector_norm(answer['residual']))<=1e-10
    assert float(answer['cache']['minimum_J'])>.1
    assert not bool((answer['displacement'][solver.mask]!=0).any())
    np.testing.assert_allclose(answer['positions'],independent['positions'],atol=2e-11,rtol=0)
