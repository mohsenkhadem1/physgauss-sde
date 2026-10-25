import numpy as np
import pytest
from experiments.campaign_predictive_risk_v01 import summarize


def toy():
    m = np.array([1., 1., 9., 9.])[:, None, None] * np.array([[1., .7, .5], [1.2, .8, .6]])
    v = np.array([1., 3., 1., 5.])[:, None, None] * np.ones((1, 2, 3))
    return m, v


def test_ratio_of_means_not_average_individual_ratios():
    m, v = toy(); r = summarize(m, v, seed=123, resamples=400)
    assert r['cells'][0]['ratio'] == 2.
    assert not np.isclose(r['cells'][0]['ratio'], (m/v).mean(axis=0)[0, 0])
    assert len(r['cells']) == 6 and len(r['paired_reductions']) == 4
    assert r['no_pass_rule'] and not r['Gaussian_assumption'] and not r['observation_noise_added']


def test_shared_object_bootstrap_and_paired_gain_against_direct_reference():
    m, v = toy(); r = summarize(m, v, seed=123, resamples=400)
    indices = np.random.default_rng(123).integers(0, 4, size=(400, 4))
    for model in range(2):
        for phase in range(3):
            expected = m[indices, model, phase].mean(axis=1)/v[indices, model, phase].mean(axis=1)
            np.testing.assert_array_equal(r['cells'][3*model+phase]['ratio_percentile_95_interval'], np.quantile(expected, [.025, .975]))
        for action in range(2):
            expected = (m[:, model, 0]-m[:, model, action+1])[indices].mean(axis=1)
            np.testing.assert_array_equal(r['paired_reductions'][2*model+action]['MSE_reduction_percentile_95_interval'], np.quantile(expected, [.025, .975]))


def test_zero_denominators_preserved_without_floor_drop_or_redraw():
    m, v = toy(); v.fill(0)
    r = summarize(m, v, seed=123, resamples=400)
    assert r['complete'] and not r['ratio_intervals_all_defined']
    assert all(c['ratio'] is None and c['ratio_percentile_95_interval'] is None
               and c['undefined_bootstrap_ratios'] == 400 for c in r['cells'])
    v[0] = 1
    r = summarize(m, v, seed=123, resamples=400)
    assert all(c['ratio'] is not None and c['ratio_percentile_95_interval'] is None
               and c['undefined_bootstrap_ratios'] > 0 for c in r['cells'])


def test_missing_objects_block_comparison_without_filtering():
    m, v = toy(); m[0, 0, 0] = np.nan
    r = summarize(m, v, seed=123, resamples=400)
    assert r['objects'] == 4 and r['unresolved_entries'] == 1 and r['comparison_blocked']
    assert r['all_object_denominators_retained']


def test_scale_and_input_units_and_correlated_channel_shape_guard():
    m, v = toy(); a = summarize(m, v, seed=123, resamples=400)
    b = summarize(m*1000, v*1000, seed=123, resamples=400)
    np.testing.assert_allclose([c['ratio'] for c in a['cells']], [c['ratio'] for c in b['cells']])
    for bad in [np.zeros((4, 2, 3, 96)), np.full((4, 2, 3), -1.)]:
        with pytest.raises(ValueError): summarize(bad, bad, seed=123, resamples=400)
