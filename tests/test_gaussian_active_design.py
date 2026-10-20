import numpy as np
import pytest

from experiments.gaussian_active_design import condition, predictive_risk, score_actions


def test_condition_matches_binary_bayes_and_preserves_zero_mass():
    means = np.array([[0.], [2.], [100.]])
    weights, diagnostics = condition(means, np.array([0.]), 1., [.5, .5, 0.])
    np.testing.assert_allclose(weights, [1/(1+np.exp(-2)), 1/(1+np.exp(2)), 0])
    assert 1 < diagnostics['effective_sample_size'] < 2


def test_predictive_risk_is_coordinate_mse_and_translation_invariant():
    targets = np.array([[0., 2.], [2., 0.]])
    assert predictive_risk(targets, [.25, .75]) == pytest.approx(.75)
    assert predictive_risk(targets+100, [.25, .75]) == pytest.approx(.75)


def test_policy_prefers_target_information_over_unrelated_variation():
    # Two independent binary latents; target depends only on the first.
    target = np.array([[-1.], [-1.], [1.], [1.]])
    means = np.array([[[-1.], [1.], [-1.], [1.]], target])
    result = score_actions(target, means, .01, fantasies=128, seed=9)
    assert result['selected_index'] == 1
    assert result['expected_risk_reduction'][0] == pytest.approx(0, abs=1e-10)
    assert result['expected_risk_reduction'][1] > .999


def test_identical_actions_and_permutation_are_consistent():
    target = np.arange(6.)[:, None]
    predictions = np.stack([target, target, target*0])
    result = score_actions(target, predictions, .2, seed=4, fantasies=64)
    assert result['score_per_cost'][0] == result['score_per_cost'][1]
    permuted = score_actions(target, predictions[[2, 1, 0]], .2, seed=4, fantasies=64)
    np.testing.assert_allclose(result['score_per_cost'], np.array(permuted['score_per_cost'])[[2, 1, 0]])


def test_correlated_covariance_matches_transformed_measurement():
    means = np.array([[0., 0.], [1., 2.]])
    covariance = np.array([[2., .4], [.4, 1.]])
    observation = np.array([.2, 1.3])
    first, _ = condition(means, observation, covariance, [.2, .8])
    factor = np.linalg.cholesky(covariance)
    second, _ = condition(np.linalg.solve(factor, means.T).T,
                          np.linalg.solve(factor, observation), 1., [.2, .8])
    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize('covariance', [0., -1., np.array([[1., 2.], [2., 1.]])])
def test_rejects_invalid_noise(covariance):
    with pytest.raises(ValueError):
        condition(np.zeros((2, 2)), np.zeros(2), covariance, [.5, .5])
