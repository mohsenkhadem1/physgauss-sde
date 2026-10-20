"""Bayesian timescale adaptation with exact conditional force integration.

Each component indexes (support/load geometry, force correlation length).
The posterior retains uncertainty over BOTH; no empirical-Bayes plug-in.
"""
from functools import lru_cache
import numpy as np
from beam_boundary import force_covariance, observation_matrix
from beam_active import State, predictive, target_risk


class AdaptiveForcePosterior:
    def __init__(self, boundaries, design, lengths, sigma=.25, elements=40, length_weights=None):
        lengths = np.asarray(lengths, dtype=float)
        if len(lengths) == 0 or np.any(lengths <= 0):
            raise ValueError('Positive force correlation lengths required')
        prior_weights = np.ones(len(lengths)) if length_weights is None else np.asarray(length_weights, dtype=float)
        if prior_weights.shape != lengths.shape or np.any(prior_weights <= 0):
            raise ValueError('Strictly positive prior weights, one per timescale')
        prior_weights /= prior_weights.sum()
        self.boundary_index = np.repeat(np.arange(len(boundaries)), len(lengths))
        self.length_index = np.tile(np.arange(len(lengths)), len(boundaries))
        self.lengths = lengths
        self.component_lengths = lengths[self.length_index]
        self.prior_weights = prior_weights[self.length_index]/len(boundaries)
        self.design, self.sigma = design, sigma
        h = np.stack([observation_matrix(b, design, elements) for b in boundaries])
        self.h = h[self.boundary_index]
        self.prior = np.stack([force_covariance(np.arange(1, 9), length=l) for l in self.component_lengths])
        maps, covs, normalizers, whitened, innovation = [], [], [], [], []
        for matrix, covariance in zip(self.h, self.prior):
            l = np.linalg.cholesky(covariance)
            w = matrix@l/sigma
            a = np.eye(8)+w.T@w
            aw = np.linalg.solve(a, w.T)
            maps.append(l@aw/sigma)
            cov = l@np.linalg.solve(a, l.T)
            covs.append(.5*(cov+cov.T))
            normalizers.append(np.linalg.slogdet(a)[1])
            whitened.append(w); innovation.append(aw)
        self.maps, self.cov, self.logdet, self.whitened_h, self.innovation_maps = map(
            np.asarray, (maps, covs, normalizers, whitened, innovation))

    def fit(self, observations):
        y = np.atleast_2d(observations)
        b = np.einsum('cmt,nm->nct', self.whitened_h, y/self.sigma)
        m = np.einsum('ctm,nm->nct', self.innovation_maps, y/self.sigma)
        scores = -.5*(np.sum((y/self.sigma)**2, axis=1)[:, None]
                       - np.sum(b*m, axis=2)+self.logdet[None, :])
        scores += np.log(self.prior_weights)[None, :]
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores); weights /= weights.sum(axis=1, keepdims=True)
        means = np.einsum('ctm,nm->nct', self.maps, y)
        return weights, means

    def length_probabilities(self, weights):
        return np.asarray(weights).reshape(*np.asarray(weights).shape[:-1], -1, len(self.lengths)).sum(axis=-2)

    def expand_models(self, models):
        h, offset, var = models
        return h[:, self.boundary_index], offset[:, self.boundary_index], var


@lru_cache(maxsize=16)
def legendre(order):
    return np.polynomial.legendre.leggauss(order)


def expected_risk_grid(state, h, offset, noise_var, spatial, tolerance=2e-5, max_order=511):
    """Converged 1D quadrature, avoiding O(component_count**2) GH nodes.

    Bounds contain every component's +/-9 SD. Check probability mass AND the
    tower identity E_y E[u^2|y] = E[u^2] against analytic prior moments.
    Split near component means and scales so narrow smooth-model predictions
    are resolved inside the much broader fast-model predictive distribution.
    """
    mu, var, ch = predictive(state, h, offset, noise_var)
    low, high = np.min(mu-9*np.sqrt(var)), np.max(mu+9*np.sqrt(var))
    landmarks = np.r_[mu-np.sqrt(var), mu, mu+np.sqrt(var)]
    edges = np.unique(np.r_[low, np.quantile(landmarks, np.linspace(0, 1, 13)), high])
    centers, widths = (edges[:-1]+edges[1:])/2, (edges[1:]-edges[:-1])/2
    norm = np.mean(spatial**2, axis=1)
    # For each component, conditional target means are AFFINE in y and second
    # moments are QUADRATIC. Precompute these coefficients rather than build a
    # nodes x components x forces array at every integration refinement.
    gain = ch/var[:, None]
    alpha = state.mean-gain*mu[:, None]
    cov = state.cov-ch[:, :, None]*ch[:, None, :]/var[:, None, None]
    intercept = (alpha[:, :, None]*spatial[:, None, :]).reshape(len(mu), -1)
    slope = (gain[:, :, None]*spatial[:, None, :]).reshape(len(mu), -1)
    quadratic = np.mean(gain*gain, axis=1)*norm
    linear = 2*np.mean(alpha*gain, axis=1)*norm
    constant = np.mean(alpha*alpha+np.diagonal(cov, axis1=1, axis2=2), axis=1)*norm
    prior_second = float(state.weights@(np.mean(state.mean**2+np.diagonal(state.cov, axis1=1, axis2=2), axis=1)*norm))
    risk_scale = max(target_risk(state, spatial), 1e-20)
    previous = None
    for order in [7, 15, 31, 63, 127, 255, 511]:
        if order > max_order:
            break
        nodes, quadrature_weights = legendre(order)
        values = (centers[:, None]+widths[:, None]*nodes).ravel()
        quadrature_weights = (widths[:, None]*quadrature_weights).ravel()
        density = np.exp(-.5*(values[:, None]-mu)**2/var)/np.sqrt(2*np.pi*var) @ state.weights
        mass = quadrature_weights*density
        logw = np.log(np.maximum(state.weights, 1e-300))[None, :]
        logw = logw-.5*((values[:, None]-mu)**2/var+np.log(2*np.pi*var))
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw); w /= w.sum(axis=1, keepdims=True)
        second = (w@quadratic)*values**2+(w@linear)*values+w@constant
        pred = w@intercept+values[:, None]*(w@slope)
        score = float(mass@(second-np.mean(pred**2, axis=1)))
        mass_error = abs(mass.sum()-1)
        tower_error = abs(mass@second-prior_second)/max(prior_second, 1e-20)
        change = abs(score-previous)/risk_scale if previous is not None else float('inf')
        if mass_error < tolerance and tower_error < tolerance and change < tolerance:
            return score, {'order': order, 'nodes': len(values), 'mass_error': float(mass_error),
                           'relative_tower_error': float(tower_error), 'relative_score_change': float(change)}
        previous = score
    raise RuntimeError(f'Predictive quadrature did not converge: order={order}, mass={mass_error}, tower={tower_error}, change={change}')


def select_adaptive_action(state, models, spatial, available, tolerance=2e-5):
    h, offset, var = models
    scores, diagnostics = {}, {}
    for a in available:
        scores[a], diagnostics[a] = expected_risk_grid(state, h[a], offset[a], var[a], spatial, tolerance)
    return min(scores, key=scores.get), scores, diagnostics
