import math
import pytest
from experiments.campaign_resolved_quantiles import bounded_continuous_quantile as old
from experiments.campaign_resolved_quantiles_v02 import bounded_continuous_quantile as new


@pytest.mark.parametrize('error', [math.ulp(1.), 1e-10])
def test_CDF_range_intersection_of_numerical_estimate_preserves_all_bounds(error):
    seen = []
    def cdf(x):
        if x == 1.:
            r = dict(cdf=1.+error, lower=1.-error, upper=1.)
        elif x == 0.:
            r = dict(cdf=-error, lower=0., upper=error)
        else:
            r = dict(cdf=x, lower=x, upper=x)
        seen.append(r.copy())
        return r
    with pytest.raises(FloatingPointError, match='ordered'):
        old(cdf, 0., 1., .5)
    r = new(cdf, 0., 1., .5)
    assert r['lower'] <= .5 <= r['upper']
    assert r['maximum_CDF_error_bound'] <= 1e-6
    assert seen[0] == dict(cdf=-error, lower=0., upper=error)


def test_valid_queries_have_bit_identical_root_values_brackets_and_calls():
    def cdf(x):
        p = x*x
        return dict(cdf=p, lower=p, upper=p)
    for p in (.025, .5, .975):
        a, b = old(cdf, 0., 1., p), new(cdf, 0., 1., p)
        for k in ('value', 'lower', 'upper', 'CDF_calls', 'iterations', 'maximum_CDF_error_bound'):
            assert a[k] == b[k]


def test_observed_public_trace_tuple_without_changing_numerical_bounds():
    observed = dict(cdf=1.0000000000000002, lower=0.9999999999999336, upper=1.)
    def cdf(x):
        return observed if x == 1. else dict(cdf=x, lower=x, upper=x)
    with pytest.raises(FloatingPointError):
        old(cdf, 0., 1., .975)
    r = new(cdf, 0., 1., .975)
    assert r['lower'] <= .975 <= r['upper']
    assert observed == dict(cdf=1.0000000000000002, lower=0.9999999999999336, upper=1.)
    assert r['maximum_CDF_error_bound'] <= 1e-6


def test_interior_inconsistent_estimate_or_invalid_bounds_still_fails():
    for cdf in (lambda x: dict(cdf=.5, lower=.6, upper=.7),
                lambda x: dict(cdf=.5, lower=.7, upper=.6),
                lambda x: dict(cdf=float('nan'), lower=0., upper=1.)):
        with pytest.raises(FloatingPointError):
            new(cdf, 0., 1., .5)
