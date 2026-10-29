import numpy as np
from scripts.position_ood.score import metrics


def test_incremental_response_separates_static_shape_bias():
    rng=np.random.default_rng(17);truth=rng.normal(0,.003,(10,81,3));bias=rng.normal(0,.001,(1,81,3))
    result=metrics(truth+bias,truth)
    assert min(result['coordinate_rmse_mm'])>.5
    np.testing.assert_allclose(result['incremental_coordinate_rmse_mm'],0,atol=1e-12)
    np.testing.assert_allclose(result['incremental_hidden_surface_rmse_mm'],0,atol=1e-12)
