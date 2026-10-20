"""Bayesian acquisition on weighted physical hypotheses, without simulator access.

The target is a future deformation or withheld-view feature vector. Candidate
means are predictions under the SAME hypotheses. Only a selected measurement
may be passed to ``condition``. Noise includes sensor and surrogate discrepancy;
it must be estimated separately from evaluation outcomes. This particle update
is an approximation, not evidence that an amortized flow is calibrated.
"""
from __future__ import annotations

import numpy as np


def normalized_weights(weights, count=None):
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.ndim != 1 or (count is not None and len(weights) != count)
            or not np.isfinite(weights).all() or (weights < 0).any()
            or weights.sum() <= 0):
        raise ValueError('Finite nonnegative weights with positive sum required')
    return weights / weights.sum()


def whitened(predictions, covariance):
    """Return predictions transformed by a strictly positive noise covariance."""
    predictions = np.asarray(predictions, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if predictions.ndim != 2 or not np.isfinite(predictions).all():
        raise ValueError('Finite [particles, features] predictions required')
    dimension = predictions.shape[-1]
    if covariance.ndim == 0:
        covariance = np.full(dimension, covariance)
    if covariance.ndim == 1:
        if covariance.shape != (dimension,) or not np.isfinite(covariance).all() or (covariance <= 0).any():
            raise ValueError('Positive finite noise variances required')
        factor = np.diag(np.sqrt(covariance))
    else:
        if (covariance.shape != (dimension, dimension) or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-12)):
            raise ValueError('Symmetric finite covariance of matching size required')
        try:
            factor = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError('Strictly positive definite covariance required') from error
    return np.linalg.solve(factor, predictions.T).T, factor


def condition(predictions, observation, covariance, weights):
    """Condition an existing posterior on ONE additional Gaussian measurement.

    Existing observations must not be applied again. A flow posterior already
    conditioned on the video would otherwise count that evidence twice.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    if observation.shape != predictions.shape[1:] or not np.isfinite(observation).all():
        raise ValueError('Finite measurement with matching feature dimension required')
    prior = normalized_weights(weights, len(predictions))
    residual, _ = whitened(predictions - observation, covariance)
    log_weights = np.full_like(prior, -np.inf)
    positive = prior > 0
    log_weights[positive] = np.log(prior[positive]) - .5*np.square(residual[positive]).sum(-1)
    log_weights -= log_weights.max()
    posterior = normalized_weights(np.exp(log_weights))
    return posterior, {'effective_sample_size': float(1/np.square(posterior).sum()),
                       'largest_weight': float(posterior.max())}


def predictive_risk(targets, weights):
    """Posterior expected mean squared error of the posterior target mean."""
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim < 2 or not np.isfinite(targets).all():
        raise ValueError('Finite [particles, ...target dimensions] required')
    targets = targets.reshape(len(targets), -1)
    weights = normalized_weights(weights, len(targets))
    mean = weights @ targets
    return float(weights @ np.square(targets - mean).mean(-1))


def score_actions(targets, predictions, covariances, weights=None, *, fantasies=256,
                  seed=0, costs=None):
    """Expected reduction of squared predictive risk per acquisition cost.

    ``predictions`` is [actions, particles, features]; no actual candidate
    observations enter selection. Common latent and standardized noise draws
    reduce Monte Carlo comparison noise. Report the uncertainty of score
    differences; very small gains do not establish a useful policy.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 3 or min(predictions.shape) < 1:
        raise ValueError('Predictions must be [actions, particles, features]')
    actions, particles, dimensions = predictions.shape
    weights = normalized_weights(np.ones(particles) if weights is None else weights, particles)
    if len(targets) != particles:
        raise ValueError('Targets and acquisition predictions must share particles')
    if fantasies < 2:
        raise ValueError('At least two fantasies required')
    covariances = np.asarray(covariances, dtype=np.float64)
    # A single covariance is shared; action-specific full/diagonal matrices
    # require an explicit leading action axis, including the scalar case.
    if covariances.ndim == 0:
        covariances = np.broadcast_to(covariances, (actions,))
    elif covariances.shape == (dimensions,):
        covariances = np.broadcast_to(covariances, (actions, dimensions))
    elif covariances.shape == (dimensions, dimensions):
        covariances = np.broadcast_to(covariances, (actions, dimensions, dimensions))
    elif covariances.shape not in ((actions, dimensions), (actions, dimensions, dimensions)):
        raise ValueError('Noise covariance shape is ambiguous or incompatible')
    costs = np.ones(actions) if costs is None else np.asarray(costs, dtype=np.float64)
    if costs.shape != (actions,) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError('Positive finite per-action costs required')
    rng = np.random.default_rng(seed)
    ancestors = rng.choice(particles, size=fantasies, p=weights)
    standard_noise = rng.normal(size=(fantasies, dimensions))
    risk_before = predictive_risk(targets, weights)
    gains = np.empty((actions, fantasies))
    ess = np.empty_like(gains)
    for action in range(actions):
        whitened_predictions, factor = whitened(predictions[action], covariances[action])
        observations = predictions[action, ancestors] + standard_noise @ factor.T
        whitened_observations = np.linalg.solve(factor, observations.T).T
        for index, observation in enumerate(whitened_observations):
            log_weights = np.full(particles, -np.inf)
            positive = weights > 0
            log_weights[positive] = (np.log(weights[positive])
                -.5*np.square(whitened_predictions[positive]-observation).sum(-1))
            posterior = normalized_weights(np.exp(log_weights-log_weights.max()))
            gains[action, index] = risk_before-predictive_risk(targets, posterior)
            ess[action, index] = 1/np.square(posterior).sum()
    score = gains.mean(-1)/costs
    selected = int(np.argmax(score))
    paired = gains[selected][None]/costs[selected]-gains/costs[:, None]
    return {'selected_index': selected, 'risk_before': risk_before,
            'expected_risk_reduction': gains.mean(-1).tolist(),
            'score_per_cost': score.tolist(),
            'score_standard_error': (gains.std(-1, ddof=1)/np.sqrt(fantasies)/costs).tolist(),
            'selected_advantage_standard_error': (paired.std(-1, ddof=1)/np.sqrt(fantasies)).tolist(),
            'expected_effective_sample_size': ess.mean(-1).tolist(),
            'fantasies': fantasies, 'seed': seed,
            'selection_inputs': 'Posterior predictions, discrepancy covariance and action costs only'}
