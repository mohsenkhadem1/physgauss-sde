"""Finite quadrature for the two-compliance stem, with outcome-blind planning."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp, ndtr
from scripts.gap_first_v01.stem_mechanics import response_row
from scripts.gap_first_v01.quantiles import gaussian_mixture_quantiles


@dataclass
class Posterior:
    theta: np.ndarray
    weights: np.ndarray
    log_likelihood: np.ndarray

    def update(self, row, value, sigma):
        ll = self.log_likelihood - .5*((self.theta @ row-value)/sigma)**2
        return Posterior(self.theta, np.exp(ll-logsumexp(ll)), ll)

    def predictions(self, row, tolerance):
        q = self.theta @ row
        mean = float(self.weights @ q)
        # Tie-break a rank-deficient likelihood by distance to prior center,
        # never by distance to true parameters.
        candidates = np.flatnonzero(self.log_likelihood >= self.log_likelihood.max()-1e-9)
        center = self.theta.mean(0)
        index = candidates[np.argmin(((self.theta[candidates]-center)**2).sum(1))]
        mle = float(q[index])
        interval = gaussian_mixture_quantiles(self.weights, q, np.zeros_like(q))[:,0]
        return {'posterior_mean': mean, 'maximum_likelihood': mle,
                'posterior_mean_risk': float(self.weights @ (abs(q-mean)>tolerance)),
                'maximum_likelihood_risk': float(self.weights @ (abs(q-mle)>tolerance)),
                'interval90_m': interval.tolist(),
                'parameter_mean': (self.weights @ self.theta).tolist(),
                'parameter_mle': self.theta[index].tolist()}


def prior(config, grid_size=None):
    n = grid_size or config['grid_size']
    lo, hi = config['compliance_prior_m_per_n']
    # Equal-volume midpoint quadrature of the continuous uniform prior.
    axis = lo + (np.arange(n)+.5)*(hi-lo)/n
    theta = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1,2)
    return Posterior(theta, np.full(n*n, 1/(n*n)), np.zeros(n*n))


def action_menu(condition, length, config):
    F = config['initial_force_n']
    look_x = 0 if condition == 'visible_base' else length
    menu = [{'name':'look', 'kind':'look',
             'row':response_row(look_x,length,length,F),
             'sigma':config['look_sigma_m'],
             'cost':config['look_cost_in_failure_probability_units'],
             'location_fraction':look_x/length}]
    for a in config['push_candidates_fraction']:
        menu.append({'name':f'push_{a:.2f}', 'kind':'push',
            'row':response_row(a*length,a*length,length,F),
            'sigma':config['push_sigma_m'],
            'cost':config['push_cost_in_failure_probability_units'],
            'location_fraction':a})
    return menu


def expected_risk(posterior, target_row, action, tolerance, bins=161):
    """Bayes predictive integration over quantized possible measurements.

    Each bin likelihood is an exact normal-CDF difference. It averages over
    hypotheses and unknown future observations, never using hidden truth.
    The returned error concerns the physical target, without sensor noise.
    """
    q = posterior.theta @ target_row
    z = posterior.theta @ action['row']
    sigma = action['sigma']
    # Use entire support so even tiny hypotheses retain predictive mass.
    edges = np.linspace(z.min()-8*sigma, z.max()+8*sigma, bins+1)
    edges[0], edges[-1] = -np.inf, np.inf
    expected_failure = 0.
    for start in range(0, bins, 32):
        e = edges[start:min(start+32,bins)+1]
        likelihood = np.diff(ndtr((e[None,:]-z[:,None])/sigma), axis=1)
        joint = posterior.weights[:,None]*likelihood
        mass = joint.sum(0)
        pred = np.divide(q @ joint, mass, out=np.zeros_like(mass), where=mass>1e-300)
        expected_failure += float((joint*(abs(q[:,None]-pred[None,:])>tolerance)).sum())
    cov = np.sum(posterior.weights*(q-posterior.weights@q)*(z-posterior.weights@z))
    vz = np.sum(posterior.weights*(z-posterior.weights@z)**2)+sigma**2
    return dict(expected_failure=expected_failure,
                objective=expected_failure+action['cost'],
                covariance_variance_reduction=float(cov**2/vz))


def plan(initial_y, condition, length, config, grid_size=None, bins=None):
    p = prior(config,grid_size).update(response_row(length,length,length,
        config['initial_force_n']),initial_y,config['initial_sigma_m'])
    a = length if condition == 'repeat_tip' else config['heldout_load_fraction']*length
    target = response_row(a,a,length,config['heldout_force_n'])
    predictions = p.predictions(target,config['risk_tolerance_m'])
    menu = action_menu(condition,length,config)
    scores = {item['name']:expected_risk(p,target,item,config['risk_tolerance_m'],
        bins or config['quadrature_predictive_observation_bins']) for item in menu}
    current = predictions['posterior_mean_risk']
    if current <= config['proceed_risk_threshold']:
        selected = 'proceed'
    else:
        best = min(scores,key=lambda name:scores[name]['objective'])
        selected = best if scores[best]['objective'] < current else 'proceed'
    covariance = max(scores,key=lambda name:scores[name]['covariance_variance_reduction'])
    q=p.theta@target
    variance=float(p.weights@((q-p.weights@q)**2))
    variance_objectives={item['name']:min(1.,max(0.,variance-scores[item['name']][
        'covariance_variance_reduction'])/config['risk_tolerance_m']**2)+item['cost'] for item in menu}
    variance_objectives['proceed']=min(1.,variance/config['risk_tolerance_m']**2)
    covariance_cost=min(variance_objectives,key=variance_objectives.get)
    def normal_risk(var):
        return float(2*ndtr(-config['risk_tolerance_m']/np.sqrt(max(var,1e-300))))
    gaussian_objectives={item['name']:normal_risk(max(0.,variance-scores[item['name']][
        'covariance_variance_reduction']))+item['cost'] for item in menu}
    gaussian_objectives['proceed']=normal_risk(variance)
    covariance_gaussian=min(gaussian_objectives,key=gaussian_objectives.get)
    return p,target,dict(predictions=predictions,action_scores=scores,
                        selected_action=selected,covariance_action=covariance,
                        covariance_cost_action=covariance_cost,
                        covariance_cost_objectives=variance_objectives,
                        covariance_gaussian_action=covariance_gaussian,
                        covariance_gaussian_objectives=gaussian_objectives)
