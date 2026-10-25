import numpy as np
import pytest
from experiments.campaign_finite_probe_reference_v01 import FiniteProbeReference, normalize


def model():
    return FiniteProbeReference(np.zeros((2, 1)), np.array([[0., -1.], [0., 1.]]),
                                np.array([[-2., -1.], [2., 1.]]), sigma_m=1.)


def test_exact_gaussian_posterior_and_independent_one_poke_updates():
    m = model(); base = m.prefix_posterior([0.])
    np.testing.assert_allclose(np.exp(base), [.5, .5], rtol=0, atol=1e-16)
    post = m.after_one_poke(base, 1, 1.)
    np.testing.assert_allclose(np.exp(post), [1 / (1 + np.exp(2)), 1 / (1 + np.exp(-2))])
    np.testing.assert_allclose(m.after_one_poke(base, 0, 3.), base, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(base, m.prefix_posterior([0.]))
    with pytest.raises(ValueError): m.prefix_posterior([0., 1.])


def test_uninformative_action_preserves_risk_and_informative_action_qualifies():
    m = model(); base = m.prefix_posterior([0.]); prior = m.moments(base)['coordinate_variance']
    for order in (8, 16, 32, 64, 128):
        assert m.expected_risk(base, 0, order=order)['risk_m2'] == pytest.approx(prior, rel=1e-14)
    choice = m.choose([0.])
    assert choice['qualified'] and choice['selected_action'] == 1
    assert 0 < choice['actions'][1]['risk_m2'] < prior


def test_target_translation_scaling_and_hypothesis_permutation():
    m = model(); base = m.prefix_posterior([0.]); original = m.expected_risk(base, 1, order=128)['risk_m2']
    moved = FiniteProbeReference(m.prefix, m.candidate, 3 * m.target + 7, 1.)
    assert moved.expected_risk(base, 1, order=128)['risk_m2'] == pytest.approx(9 * original, rel=1e-13)
    permuted = FiniteProbeReference(m.prefix[::-1], m.candidate[::-1], m.target[::-1], 1.)
    assert permuted.expected_risk(base, 1, order=128)['risk_m2'] == pytest.approx(original, rel=1e-14)
    np.testing.assert_allclose(permuted.moments(base)['mean'], m.moments(base)['mean'])


def test_catalog_immutable_and_complete_ties_declared():
    m = model()
    with pytest.raises(ValueError): m.target[0, 0] = 3
    tied = FiniteProbeReference(m.prefix, np.zeros((2, 3)), m.target, 1.).choose([0.])
    assert tied['qualified'] and tied['numerical_ties'] == [0, 1, 2] and tied['selected_action'] == 0
    with pytest.raises(ValueError): FiniteProbeReference(m.prefix, m.candidate, m.target, 0.)
    with pytest.raises(ValueError): normalize([0., np.nan])


def test_expected_risk_matches_independent_binary_continuous_integral():
    from scipy.integrate import quad
    from scipy.stats import norm
    m = model(); base = m.prefix_posterior([0.])
    expected, error = quad(lambda y: .5 * (norm.pdf(y - 1) + norm.pdf(y + 1)) *
                           2.5 / np.cosh(y) ** 2, -12, 12, epsabs=1e-12, epsrel=1e-12)
    assert error < 1e-10
    assert m.expected_risk(base, 1, order=128)['risk_m2'] == pytest.approx(expected, rel=1e-7)
