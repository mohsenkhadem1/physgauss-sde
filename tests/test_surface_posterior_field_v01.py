import numpy as np
import pytest
from scipy.sparse.linalg import spsolve
from experiments import campaign_surface_p2_static_v02 as physics
from experiments.campaign_surface_posterior_field_v01 import PosteriorMeanField


def test_weighted_full_field_matches_sparse_responses_including_interior_force():
    f = physics.SparseStaticForward(2)
    force = np.column_stack([x.ravel()[f.free] for x in physics.qualification_loads(2).values()])
    # A nonzero interior force exercises the exact 1/E particular response.
    force = np.column_stack([force, np.random.default_rng(60019).normal(size=len(f.free))*1e-5])
    model = PosteriorMeanField(f.Kunit_free, f.W_free, force)
    theta = np.array([[2100., 7.], [4000., 12.], [7900., 28.]])
    p = np.array([.1, .3, .6])
    expected = sum(w*spsolve(E*f.Kunit_free+K*f.W_free, force)
                   for (E, K), w in zip(theta, p)).T
    for chunk in (1, 2, 10):
        np.testing.assert_allclose(model.mean(theta, p, chunk=chunk), expected, rtol=2e-10, atol=2e-14)
    assert model.diagnostics['all_modes_retained'] and model.diagnostics['exact_interior_particular']
    assert not model.diagnostics['eigenvalues_clipped']


def test_invalid_probability_and_hypotheses_are_rejected():
    f = physics.SparseStaticForward(1)
    model = PosteriorMeanField(f.Kunit_free, f.W_free, np.ones((len(f.free), 1))*1e-6)
    for theta, p in [([[4000, 12]], [.9]), ([[4000, 12]], [-1]),
                     ([[0, 12]], [1]), ([[np.nan, 12]], [1])]:
        with pytest.raises(ValueError):model.mean(theta, p)
