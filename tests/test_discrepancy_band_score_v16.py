import numpy as np
import pytest
from scripts.compositional_neural_mechanics_reporting.discrepancy_band_score import band_metrics, Z90


def test_interval_score_penalizes_unneeded_width_and_missed_targets():
    exact = band_metrics(np.zeros(2), np.zeros(2), 0.)
    wide = band_metrics(np.zeros(2), np.zeros(2), 1.)
    missed = band_metrics(np.array([2., -2.]), np.zeros(2), 0.)
    assert exact == dict(coverage=1., mean_halfwidth_m=0., mean_interval_score_m=0.)
    assert wide['coverage'] == 1. and wide['mean_interval_score_m'] == 2*Z90
    assert missed['coverage'] == 0. and missed['mean_interval_score_m'] == 40.
    partly = band_metrics(np.array([2., -2.]), np.zeros(2), 1.)
    assert np.isclose(partly['mean_interval_score_m'], 2*Z90+20*(2-Z90))


def test_discrepancy_variance_rejects_nonfinite_and_does_not_shrink_bands():
    error = np.array([.1, .2, -.3])
    sd = np.array([.01, .1, .2])
    first, extra = band_metrics(error, sd, 0), band_metrics(error, sd, .01)
    assert extra['coverage'] >= first['coverage']
    assert extra['mean_halfwidth_m'] > first['mean_halfwidth_m']
    for invalid in (-1., np.nan, np.inf):
        with pytest.raises(ValueError):
            band_metrics(error, sd, invalid)
