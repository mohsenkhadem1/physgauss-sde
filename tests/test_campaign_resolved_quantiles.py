import math
import pytest
from experiments.campaign_resolved_quantiles import bounded_continuous_quantile as quantile


def exact(cdf):
    def f(x):
        p = min(1., max(0., cdf(x)))
        return dict(cdf=p, lower=p, upper=p)
    return f


@pytest.mark.parametrize('amplitude', [1e-3, 1e-9, 1e-15])
def test_distinct_uniform_tails_far_below_original_absolute_tolerance(amplitude):
    cdf = exact(lambda y: y/amplitude)
    results = [quantile(cdf, 0., amplitude, p) for p in (.025, .5, .975)]
    assert results[0]['value'] < results[1]['value'] < results[2]['value']
    for p, r in zip((.025, .5, .975), results):
        assert abs(r['value']/amplitude-p) <= 1e-6
        assert r['upper']-r['lower'] <= min(1e-8, amplitude*1e-8)
        assert r['maximum_CDF_error_bound'] <= 1e-6
    if amplitude < 1e-8:
        # Old absolute-only termination could return this same midpoint at all
        # three probabilities before any refinement, collapsing the interval.
        assert abs(.5-.025) > results[0]['maximum_CDF_error_bound']


@pytest.mark.parametrize('amplitude', [1e-3, 1e-12])
def test_log_uniform_material_quadratic_inverse_stiffness_pushforward(amplitude):
    # E/Eref ~ logUniform[1,2], Y=A*(Eref/E)^2.
    cdf = exact(lambda y: 1+math.log(y/amplitude)/(2*math.log(2)))
    for p in (.025, .5, .975):
        expected = amplitude*math.exp(-2*math.log(2)*(1-p))
        r = quantile(cdf, amplitude/4, amplitude, p)
        assert r['lower'] <= expected <= r['upper']
        assert r['maximum_CDF_error_bound'] <= 1e-6


def test_nonmonotone_quadratic_two_preimages_has_analytic_quantiles():
    amplitude = 1e-12
    # U~Uniform[0,1], Y=A*(U-.5)^2, CDF=2sqrt(Y/A).
    cdf = exact(lambda y: 2*math.sqrt(max(0., y/amplitude)))
    for p in (.025, .5, .975):
        r = quantile(cdf, 0., amplitude/4, p)
        assert r['lower'] <= amplitude*(p/2)**2 <= r['upper']
        assert r['maximum_CDF_error_bound'] <= 1e-6


def test_unresolved_CDF_uncertainty_is_failure_not_relaxed_interval():
    def cdf(x):
        y = min(1., max(0., x))
        return dict(cdf=y, lower=max(0., y-.01), upper=min(1., y+.01))
    with pytest.raises(FloatingPointError, match='unresolved'):
        quantile(cdf, 0., 1., .5)


def test_discrete_jump_requires_explicit_atom_handling():
    cdf = exact(lambda y: float(y >= .5))
    with pytest.raises(FloatingPointError):
        quantile(cdf, 0., 1., .025)


def test_actual_polynomial_posterior_tiny_targets_and_explicit_atoms():
    import numpy as np
    from experiments.campaign_c38_exact_posterior_memo import BoundsMemoLikelihood, MemoPolynomialPosterior
    from experiments.campaign_resolved_quantiles import polynomial_coordinate_quantile
    # Zero observations have a constant likelihood, hence the exact logE prior.
    model = lambda: BoundsMemoLikelihood(np.zeros((3, 1)), np.zeros(1), np.eye(1),
                       young_bounds=(75000., 150000.), young_reference_Pa=75000.,
                       root_digits=50, root_log_energy_tolerance=1e-12, log_drops=(1., 4., 16., 64.))
    for method, order in (('adaptive', 64), ('legendre', 64), ('legendre', 128)):
        post = MemoPolynomialPosterior({'a': model(), 'b': model()}, method=method, order=order)
        for amplitude in (1e-3, 1e-12):
            targets = {c: np.array([0., amplitude, 0.]) for c in post.categories}
            for p in (.025, .5, .975):
                r = polynomial_coordinate_quantile(post, targets, p)
                expected = amplitude*2**(p-1)
                assert r['lower_m'] <= expected <= r['upper_m']
                assert r['CDF_bracket']['maximum_CDF_error_bound'] <= 1e-6
        common = {c: np.array([1e-12, 0., 0.]) for c in post.categories}
        assert polynomial_coordinate_quantile(post, common, .5)['kind'] == 'common_constant_atom'
        mixed = {'a': np.array([0., 0., 0.]), 'b': np.array([1e-12, 0., 0.])}
        assert polynomial_coordinate_quantile(post, mixed, .025)['kind'] == 'category_mixture_atom'
