import copy
import numpy as np
import pytest
from experiments import plot_noisy_poke_diagnostics_v01 as plot


def rows():
    return [dict(index=i, object_index=i % 8, replicate=i // 8,
                 metrics={p: dict(MSE_m2=float((i // 8) ** 2), posterior_variance_m2=float((i % 8) ** 2))
                          for p in plot.POLICIES}) for i in range(32)]


def test_all_repeats_and_roots_after_averaging():
    result = plot.aggregate(list(reversed(rows())))
    before = result['policies']['before']
    assert before['MSE_m2'] == 3.5
    assert before['RMSE_m'] == np.sqrt(3.5)
    assert before['RMSE_m'] != np.mean([0., 1., 2., 3.])
    assert before['posterior_variance_m2'] == 17.5
    assert before['posterior_RMS_SD_m'] == np.sqrt(17.5)
    assert len(result['objects']) == 8 and result['all32_retained']


def test_missing_duplicate_or_wrong_identity_rejected():
    value = rows()
    for altered in (value[:-1], value[:-1] + [value[0]]):
        with pytest.raises(ValueError, match='Every prescribed'):
            plot.aggregate(altered)
    altered = copy.deepcopy(value)
    altered[0]['replicate'] = 3
    with pytest.raises(ValueError, match='Every prescribed'):
        plot.aggregate(altered)


def test_negative_or_nonfinite_metrics_rejected():
    for bad in (-1., float('nan'), float('inf')):
        value = rows()
        value[0]['metrics']['adaptive']['MSE_m2'] = bad
        with pytest.raises(ValueError, match='Finite nonnegative'):
            plot.aggregate(value)
