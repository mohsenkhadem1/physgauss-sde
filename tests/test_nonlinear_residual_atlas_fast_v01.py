import numpy as np
import pytest
from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas
from experiments.campaign_nonlinear_residual_atlas_fast_v01 import FastResidualAtlas


def test_original_polynomial_parity_at_nodes_queries_and_roundoff_halo():
    rng = np.random.default_rng(9043)
    atlas = NonlinearResidualAtlas(rng.normal(size=(9, 9, 3, 24))*1e-5,
                                  ['a', 'b', 'c'], provenance_fingerprint='synthetic')
    fast = FastResidualAtlas(atlas)
    nodes = np.stack(np.meshgrid(atlas.axis, atlas.axis, indexing='ij'), -1).reshape(-1, 2)
    queries = np.concatenate((nodes, rng.random((127, 2)), [[-1e-15, 1+1e-15]]))
    for labels in (['a'], ['c', 'a', 'c'], ['a', 'b', 'c']):
        np.testing.assert_allclose(fast.predict(queries, labels, chunk=17),
                                   atlas.predict(queries, labels, chunk=13), rtol=0, atol=2e-19)
        for point in queries[::19]:
            np.testing.assert_allclose(fast.predict(point[None], labels),
                                       atlas.predict(point[None], labels), rtol=0, atol=2e-19)
    assert fast.fingerprint == atlas.fingerprint
    np.testing.assert_allclose(fast.coarsen(5).predict(queries, ['b']),
                               atlas.coarsen(5).predict(queries, ['b']), rtol=0, atol=2e-19)


def test_exact_table_cache_is_bounded_and_query_guards_are_preserved():
    atlas = NonlinearResidualAtlas(np.ones((3, 3, 3, 2)), ['a', 'b', 'c'],
                                  provenance_fingerprint='synthetic')
    fast = FastResidualAtlas(atlas, table_cache_bytes=200)
    for label in atlas.labels:
        np.testing.assert_array_equal(fast.predict([[.5, .5]], [label]), [[[1., 1.]]])
        assert fast.allocated_bytes <= 200
    assert len(fast.cache) == 1
    assert fast.predict(np.empty((0, 2)), ['a']).shape == (0, 1, 2)
    for query in ([[1.01, .5]], [[np.nan, .5]]):
        with pytest.raises(ValueError):
            fast.predict(query, ['a'])
    with pytest.raises(ValueError):
        fast.predict([[.5, .5]], ['missing'])
    with pytest.raises(ValueError):
        fast.predict([[.5, .5]], ['a'], chunk=0)
