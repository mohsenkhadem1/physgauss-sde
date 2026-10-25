import numpy as np
import pytest
from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas
from experiments.campaign_nonlinear_residual_atlas_torch_v01 import TorchResidualAtlas


def test_tensor_contraction_matches_original_at_nodes_and_off_grid():
    rng = np.random.default_rng(9041)
    atlas = NonlinearResidualAtlas(rng.normal(size=(9, 9, 3, 24))*1e-5,
                                  ['a', 'b', 'c'], provenance_fingerprint='synthetic')
    accelerated = TorchResidualAtlas(atlas, device='cpu')
    nodes = np.stack(np.meshgrid(atlas.axis, atlas.axis, indexing='ij'), -1).reshape(-1, 2)
    queries = np.concatenate((nodes, rng.random((127, 2)), [[-1e-15, 1+1e-15]]))
    selected = ['c', 'a', 'c']
    expected = atlas.predict(queries, selected, chunk=13)
    actual = accelerated.predict(queries, selected, chunk=17)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-19)
    assert accelerated.fingerprint == atlas.fingerprint
    assert actual.shape == (209, 3, 24)


def test_acceleration_preserves_input_guards_and_empty_batch():
    atlas = NonlinearResidualAtlas(np.zeros((3, 3, 1, 2)), ['a'],
                                  provenance_fingerprint='synthetic')
    accelerated = TorchResidualAtlas(atlas, device='cpu')
    assert accelerated.predict(np.empty((0, 2)), ['a']).shape == (0, 1, 2)
    for q in ([[1.01, .5]], [[np.nan, .5]]):
        with pytest.raises(ValueError):
            accelerated.predict(q, ['a'])
    with pytest.raises(ValueError):
        accelerated.predict([[.5, .5]], ['absent'])
    with pytest.raises(ValueError):
        TorchResidualAtlas(atlas, device='cuda')
