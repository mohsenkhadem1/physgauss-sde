import numpy as np
import pytest
from experiments.campaign_cluster_calibration_v01 import (
    binomial_intervals, holm_adjust, primary_calibration, paired_cluster_intervals)


def test_binomial_extremes_and_family_correction():
    n = 64
    zero = binomial_intervals(0, n)
    full = binomial_intervals(n, n)
    tail = .05/12
    np.testing.assert_allclose(zero['simultaneous_Bonferroni_Clopper_Pearson'],
                               [0., 1-tail**(1/n)], rtol=0, atol=1e-15)
    np.testing.assert_allclose(full['simultaneous_Bonferroni_Clopper_Pearson'],
                               [tail**(1/n), 1.], rtol=0, atol=1e-15)
    middle = binomial_intervals(32, n)['pointwise_Wilson']
    assert .38 < middle[0] < .39 and .61 < middle[1] < .62


def test_holm_stepdown_unsorted_ties_and_monotonicity():
    np.testing.assert_allclose(holm_adjust([.03, .001, .03, .9]), [.09, .004, .09, .9])
    with pytest.raises(ValueError):
        holm_adjust([np.nan, .2])


def test_missing_objects_remain_in_64_denominator_and_block_conclusions():
    covered = np.ones((64, 6));pit = np.tile((np.arange(64)+.5)[:, None]/64, (1, 6))
    covered[:4, 0] = 0;covered[4:7, 0] = np.nan;pit[4:7, 0] = np.nan
    result = primary_calibration(covered, pit, labels=list('abcdef'))
    row = result['entries'][0]
    assert row['object_count'] == 64 and row['unresolved_coverage'] == 3
    assert row['coverage_bounds'] == [57/64, 60/64]
    assert row['intervals'] is None and row['PIT_diagnostic'] is None
    assert result['comparison_conclusions_blocked']
    assert all('KS_Holm_adjusted_pvalue' not in r['PIT_diagnostic']
               for r in result['entries'][1:])


def test_complete_uniform_PIT_diagnostics_keep_64_objects():
    pit = np.tile((np.arange(64)+.5)[:, None]/64, (1, 6))
    result = primary_calibration(np.ones((64, 6)), pit, labels=list('abcdef'))
    assert result['complete'] and not result['comparison_conclusions_blocked']
    for row in result['entries']:
        d = row['PIT_diagnostic']
        assert sum(d['histogram_10_equal_bins']) == 64
        assert d['KS_Holm_adjusted_pvalue'] > .99
        assert .20 < d['simultaneous_DKW_half_width'] < .21
    with pytest.raises(ValueError):
        primary_calibration(np.full((64, 6), .5), pit, labels=list('abcdef'))


def test_bootstrap_shares_object_resamples_and_refuses_missing_pairs():
    x = np.arange(64)/64
    values = np.column_stack((x, 2*x, np.full(64, 3.)))
    first = paired_cluster_intervals(values, labels=['x', 'twice', 'constant'], resamples=512)
    second = paired_cluster_intervals(values, labels=['x', 'twice', 'constant'], resamples=512)
    assert first == second
    e = first['entries']
    np.testing.assert_allclose(e[1]['percentile_95_interval'],
                               2*np.array(e[0]['percentile_95_interval']), rtol=0, atol=0)
    assert e[2]['percentile_95_interval'] == [3., 3.]
    values[0, 1] = np.nan
    with pytest.raises(ValueError, match='never drop'):
        paired_cluster_intervals(values, labels=['x', 'twice', 'constant'])
