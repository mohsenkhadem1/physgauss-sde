"""Restricted known-catalog controls, not a generalized physical posterior.

Both nonlinear and linear controls must receive the SAME finite prior. This
module has no file access, random draws, case identity, or true-state input to
selection. A caller must seal continuous-policy choices before constructing
catalogs from the physical reference data. Observation noise is unchanged.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from numpy.polynomial.hermite import hermgauss


def normalize(log_weights):
    value = np.asarray(log_weights, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError('Finite log weight for every declared hypothesis')
    return value - logsumexp(value)


@dataclass(frozen=True)
class FiniteProbeReference:
    prefix: np.ndarray
    candidate: np.ndarray
    target: np.ndarray
    sigma_m: float = 20e-6

    def __post_init__(self):
        arrays = []
        for name in ('prefix', 'candidate', 'target'):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
                raise ValueError('Every catalog response must be finite and two-dimensional')
            value.flags.writeable = False
            object.__setattr__(self, name, value)
            arrays.append(value)
        if len({v.shape[0] for v in arrays}) != 1 or len(arrays[0]) < 2:
            raise ValueError('Same complete hypothesis catalog in all response roles')
        if not np.isfinite(self.sigma_m) or self.sigma_m <= 0:
            raise ValueError('Positive unchanged measurement noise')

    def prefix_posterior(self, observation):
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != self.prefix.shape[1:] or not np.isfinite(observation).all():
            raise ValueError('Only the declared finite noisy prefix')
        return normalize(-.5 * np.sum(((self.prefix - observation) / self.sigma_m) ** 2, axis=1))

    def _weights(self, log_weights):
        value = normalize(log_weights)
        if value.shape != (len(self.target),):
            raise ValueError('Every hypothesis retained')
        return value, np.exp(value)

    def moments(self, log_weights):
        _, weight = self._weights(log_weights)
        mean = weight @ self.target
        variance = weight @ ((self.target - mean) ** 2)
        return dict(mean=mean, SD=np.sqrt(variance), variance=variance,
                    coordinate_variance=float(variance.mean()))

    def after_one_poke(self, prefix_log_weights, action, observation):
        base, _ = self._weights(prefix_log_weights)
        if type(action) is not int or not 0 <= action < self.candidate.shape[1]:
            raise ValueError('Action in the complete predeclared catalog')
        if not np.isscalar(observation) or not np.isfinite(observation):
            raise ValueError('One finite scalar poke outcome')
        return normalize(base - .5 * ((self.candidate[:, action] - observation) / self.sigma_m) ** 2)

    def expected_risk(self, prefix_log_weights, action, *, order):
        base, weight = self._weights(prefix_log_weights)
        if type(action) is not int or not 0 <= action < self.candidate.shape[1]:
            raise ValueError('Action in the complete predeclared catalog')
        if type(order) is not int or order < 2:
            raise ValueError('Positive Gaussian-Hermite refinement order')
        nodes, quadrature = hermgauss(order)
        means = self.candidate[:, action]
        # Integrate every component of the Gaussian observation mixture.
        outcomes = (means[:, None] + np.sqrt(2) * self.sigma_m * nodes).reshape(-1)
        log_posterior = base[None] - .5 * ((outcomes[:, None] - means[None]) / self.sigma_m) ** 2
        log_posterior -= logsumexp(log_posterior, axis=1)[:, None]
        posterior = np.exp(log_posterior)
        conditional_mean = posterior @ self.target
        # Center before squaring: avoid subtraction of nearly equal moments.
        conditional_variance = np.sum(posterior[:, :, None] *
                                      (self.target[None] - conditional_mean[:, None]) ** 2,
                                      axis=1).mean(axis=1).reshape(len(weight), order)
        risk = float(weight @ conditional_variance @ (quadrature / np.sqrt(np.pi)))
        prior = self.moments(base)['coordinate_variance']
        return dict(order=order, risk_m2=risk, prior_variance_m2=prior,
                    all_hypotheses_retained=True,
                    log_weight_range=float(np.ptp(base)),
                    floating_underflow_hypotheses=int(np.count_nonzero(weight == 0)))

    def choose(self, observation, *, orders=(8, 16, 32, 64, 128), relative_gate=1e-4):
        if tuple(orders) != (8, 16, 32, 64, 128) or relative_gate != 1e-4:
            raise ValueError('Frozen finite-reference refinement plan')
        base = self.prefix_posterior(observation)
        prior = self.moments(base)['coordinate_variance']
        rows = []
        for action in range(self.candidate.shape[1]):
            levels = [self.expected_risk(base, action, order=o) for o in orders]
            error = abs(levels[-1]['risk_m2'] - levels[-2]['risk_m2'])
            rounding = 128 * np.finfo(np.float64).eps * max(prior, np.finfo(float).tiny)
            risk = levels[-1]['risk_m2']
            qualified = bool(np.isfinite(risk) and risk >= 0 and risk <= prior + error + rounding
                             and error <= relative_gate * max(prior, np.finfo(float).tiny))
            rows.append(dict(action=action, levels=levels, risk_m2=risk,
                             refinement_indicator_m2=error, qualified=qualified))
        qualified = all(row['qualified'] for row in rows)
        best = min(rows, key=lambda row: (row['risk_m2'], row['action']))
        ties = [row['action'] for row in rows if row['risk_m2'] - best['risk_m2'] <=
                row['refinement_indicator_m2'] + best['refinement_indicator_m2']]
        return dict(qualified=qualified, actions=rows, numerical_ties=ties,
                    selected_action=min(ties) if qualified else None,
                    deterministic_tie_rule='Smallest action index, declared before outcomes',
                    prior_variance_m2=prior, refinement_indicators_are_not_rigorous_bounds=True,
                    scope='Known-catalog finite uniform prior; no generalized inference/calibration claim')
