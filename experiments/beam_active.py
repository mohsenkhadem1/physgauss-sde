"""Sequential Bayesian experimental design for the quasistatic beam.

The policy receives only posterior state and candidate measurement models.
Unknown original loads remain analytically integrated, including during planning.
"""
from dataclasses import dataclass, replace
import numpy as np
from beam_boundary import candidates, response, camera, observation_matrix


@dataclass
class State:
    weights: np.ndarray  # boundary mixture C
    mean: np.ndarray     # original force amplitudes C x 8
    cov: np.ndarray      # C x 8 x 8


def action_pool():
    # One scalar tracked vertical displacement per action; all cost one unit.
    views = [dict(kind='view', time_s=t, x_m=x)
             for t in [2, 3, 5, 6, 7] for x in [.4, .7]]
    probes = [dict(kind='probe', load_x_m=x, force_N=.04,
                   observe_x_m=.7, camera_time_s=4)
              for x in [.3, .55, .8]]
    return views + probes


def action_models(boundaries, actions, elements=40, sigma=.25):
    """y = h_beta F + offset_beta + epsilon.

    A probe uses an unloaded/reset beam, known force and reference position.
    Its sigma includes the total measurement noise; no noisy frame subtraction.
    """
    h = np.zeros((len(actions), len(boundaries), 8))
    offset = np.zeros((len(actions), len(boundaries)))
    for a, action in enumerate(actions):
        for c, boundary in enumerate(boundaries):
            if action['kind'] == 'view':
                h[a, c] = observation_matrix(boundary, [
                    (action['time_s']-1, action['x_m'], 1)], elements)[0]
            elif action['kind'] == 'probe':
                probe = replace(boundary, load_center=action['load_x_m'])
                offset[a, c] = (response(probe, np.array([action['observe_x_m']]), elements)[0]
                    * action['force_N'] * camera(action['camera_time_s'])[0][1, 1])
            else:
                raise ValueError(action['kind'])
    return h, offset, np.full(len(actions), sigma**2)


def predictive(state, h, offset, noise_var):
    ch = np.einsum('cij,cj->ci', state.cov, h)
    mu = np.einsum('ci,ci->c', h, state.mean) + offset
    variance = np.einsum('ci,ci->c', h, ch) + noise_var
    return mu, variance, ch


def hypothetical_update(state, h, offset, noise_var, values):
    """Exact conditioning for an array of possible scalar observations."""
    mu, variance, ch = predictive(state, h, offset, noise_var)
    residual = np.atleast_1d(values)[:, None] - mu
    log_weights = (np.log(np.maximum(state.weights, 1e-300))[None, :]
                   - .5*(residual**2 / variance + np.log(2*np.pi*variance)))
    log_weights -= log_weights.max(axis=1, keepdims=True)
    weights = np.exp(log_weights)
    weights /= weights.sum(axis=1, keepdims=True)
    means = state.mean[None, :, :] + residual[:, :, None]*(ch/variance[:, None])[None, :, :]
    cov = state.cov - ch[:, :, None]*ch[:, None, :]/variance[:, None, None]
    cov = .5*(cov + cov.transpose(0, 2, 1))
    return weights, means, cov


def update(state, h, offset, noise_var, value):
    w, m, c = hypothetical_update(state, h, offset, noise_var, [value])
    return State(w[0], m[0], c)


def target_components(state, spatial):
    """Original load sequence at hidden coordinates; shape C x time x space."""
    means = state.mean[:, :, None]*spatial[:, None, :]
    variances = np.diagonal(state.cov, axis1=1, axis2=2)[:, :, None]*spatial[:, None, :]**2
    return means, np.maximum(variances, 0)


def target_risk(state, spatial):
    means, variances = target_components(state, spatial)
    prediction = np.einsum('c,ctx->tx', state.weights, means)
    return float(np.mean(np.einsum('c,ctx->tx', state.weights, variances+means**2)-prediction**2))


def expected_risk(state, h, offset, noise_var, spatial, quadrature=15):
    """Integrate posterior prediction variance over the predictive mixture.

    Gauss-Hermite quadrature is performed under EVERY boundary component;
    no MAP boundary, plug-in force, or simulated truth enters this score.
    """
    mu, variance, _ = predictive(state, h, offset, noise_var)
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature)
    values = (mu[:, None] + np.sqrt(2*variance[:, None])*nodes).ravel()
    mass = (state.weights[:, None]*weights[None, :]/np.sqrt(np.pi)).ravel()
    w, m, cov = hypothetical_update(state, h, offset, noise_var, values)
    # E[||target||^2 | y] averaged over all time/space targets, then subtract
    # ||E[target | y]||^2. This avoids a Q x C x time x space allocation.
    spatial_norm = np.mean(spatial**2, axis=1)
    second = np.mean(m**2 + np.diagonal(cov, axis1=1, axis2=2)[None, :, :], axis=2)*spatial_norm
    pred = np.einsum('qc,qct,cx->qtx', w, m, spatial, optimize=True)
    risks = np.sum(w*second, axis=1) - np.mean(pred**2, axis=(1, 2))
    return float(mass @ risks)


def select_action(state, models, spatial, available, quadrature=15):
    h, offset, var = models
    scores = {a: expected_risk(state, h[a], offset[a], var[a], spatial, quadrature)
              for a in available}
    # Stable tie break by declared pool order.
    return min(scores, key=scores.get), scores
