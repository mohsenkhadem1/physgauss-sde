"""Analytic checks of categorical/material reference integration measure."""
import numpy as np
import pytest
from experiments.campaign_point_material_reference import summarize, prefix_indices


def particles():
    ids = np.array([4, 4, 9, 9, 9, 9], dtype=np.int64)
    x = np.array([[1., 2., .3, .4], [3., 4., .4, .5],
                  [5., 6., .3, .4], [7., 8., .4, .5],
                  [9., 10., .3, .4], [11., 12., .4, .5]])
    return ids, x, {4: .8, 9: .2}


def test_constant_likelihood_retains_prior_with_unequal_stratum_counts():
    ids, x, p = particles()
    report, w = summarize(ids, x, np.full(6, -1000.), p)
    np.testing.assert_allclose(w, [.4, .4, .05, .05, .05, .05], atol=1e-14)
    np.testing.assert_allclose(report['log_evidence'], -1000., atol=1e-12)
    np.testing.assert_allclose([s['posterior_probability'] for s in report['support_categories']], [.8, .2])
    np.testing.assert_allclose([s['mean'] for s in report['material']], w@x)


def test_category_bayes_factors_use_prior_mass_not_sample_count():
    ids, x, p = particles()
    report, _ = summarize(ids, x, np.where(ids == 4, np.log(.25), 0.), p)
    np.testing.assert_allclose([s['posterior_probability'] for s in report['support_categories']], [.5, .5])
    np.testing.assert_allclose(report['log_evidence'], np.log(.4))
    np.testing.assert_allclose([s['conditional_material_ESS'] for s in report['support_categories']], [2., 4.])


def test_zero_likelihood_category_stays_declared_and_global_impossibility_fails():
    ids, x, p = particles()
    report, w = summarize(ids, x, np.where(ids == 4, -np.inf, 0.), p)
    assert len(report['support_categories']) == 2 and report['categories_dropped'] == []
    assert report['support_categories'][0]['posterior_probability'] == 0
    assert report['support_categories'][0]['conditional_log_evidence'] is None
    np.testing.assert_allclose(w, [0., 0., .25, .25, .25, .25])
    with pytest.raises(ValueError, match='impossible'):
        summarize(ids, x, np.full(6, -np.inf), p)


def test_missing_positive_mass_category_is_not_renormalized():
    ids, x, p = particles()
    with pytest.raises(ValueError, match='Complete'):
        summarize(ids[:2], x[:2], np.zeros(2), p)
    with pytest.raises(ValueError, match='Normalized'):
        summarize(ids, x, np.zeros(6), {4: .8, 9: .1})


def test_particle_refinement_keeps_equal_prefix_within_every_stratum():
    ids, _, p = particles()
    np.testing.assert_array_equal(prefix_indices(ids, 2, p), [0, 1, 2, 3])
    with pytest.raises(ValueError, match='Insufficient'):
        prefix_indices(ids, 3, p)
