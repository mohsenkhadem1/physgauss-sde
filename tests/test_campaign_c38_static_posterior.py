import numpy as np
import pytest
from scipy.special import ndtr, ndtri
from experiments.campaign_c38_static_posterior import StaticPosterior, compare_integrators


def test_analytic_uniform_log_prior_and_category_odds():
    p = StaticPosterior(lambda category, E: 0. if category == 'a' else np.log(3.),
                        categories=('a', 'b'))
    np.testing.assert_allclose(p.category_probabilities, [.25, .75], atol=1e-14)
    assert p.log_normalizer == pytest.approx(np.log(2.), abs=1e-13)
    assert p.cdf(np.sqrt(75000.*150000.)) == pytest.approx(.5, abs=1e-13)
    assert p.quantile(.025) == pytest.approx(75000.*2**.025, rel=1e-12)


def test_truncated_gaussian_log_material_matches_closed_form():
    mean, sigma = .35, .09
    def likelihood(category, E):
        return -.5*((np.log(E/75000.)/np.log(2.)-mean)/sigma)**2 - 10000.
    p = StaticPosterior(likelihood)
    lower, upper = ndtr(-mean/sigma), ndtr((1-mean)/sigma)
    for probability in [.025, .5, .975]:
        u = mean+sigma*ndtri(lower+probability*(upper-lower))
        assert p.quantile(probability) == pytest.approx(75000.*2**u, rel=1e-10)
    report = compare_integrators(likelihood)
    for row in report['comparisons']:
        assert row['maximum_material_cdf_difference'] < 1e-10
        assert row['maximum_log_material_quantile_difference'] < 1e-10
    assert not report['physical_calibration_established']


def test_bimodal_support_and_order_failure_not_hidden():
    def likelihood(category, E):
        x = np.log(E/75000.)/np.log(2.)
        center = .237 if category == 'both_ends' else .763
        return -.5*((x-center)/.018)**2
    p = StaticPosterior(likelihood)
    np.testing.assert_allclose(p.category_probabilities, [.5, .5], atol=1e-10)
    assert p.cdf(np.sqrt(75000.*150000.)) == pytest.approx(.5, abs=1e-10)
    report = compare_integrators(likelihood, orders=(4, 8))
    assert max(r['maximum_material_cdf_difference'] for r in report['comparisons']) > .01


def test_invalid_likelihood_is_not_dropped():
    with pytest.raises(FloatingPointError):
        StaticPosterior(lambda category, E: np.nan)
