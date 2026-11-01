"""Weighted continuous-prior quadrature update and response diagnostics."""
import numpy as np
from scipy.special import logsumexp


class Likelihood:
    def __init__(self, query):
        means, covariance = query.checked(len(query.means))
        C = np.diag(covariance) if covariance.ndim == 1 else covariance
        self.factor = np.linalg.cholesky(C)
        self.means = np.linalg.solve(self.factor, means.T).T
        self.normalizer = -.5*means.shape[1]*np.log(2*np.pi)-np.log(np.diag(self.factor)).sum()

    def log_density(self, observed):
        observation = np.asarray(observed, dtype=float)
        if observation.shape != (self.means.shape[1],) or not np.isfinite(observation).all():
            raise ValueError('Finite measured sensor feature vector required')
        white = np.linalg.solve(self.factor, observation)
        return self.normalizer-.5*np.square(self.means-white).sum(1)


def posterior(prior, log_likelihoods):
    prior = np.asarray(prior, dtype=float)
    if prior.ndim != 1 or not np.isfinite(prior).all() or np.any(prior <= 0) or not np.isclose(prior.sum(),1):
        raise ValueError('Normalized strictly positive quadrature weights required')
    log_density = np.log(prior)
    for value in log_likelihoods:
        value = np.asarray(value,dtype=float)
        if value.shape != prior.shape or not np.isfinite(value).all():
            raise ValueError('One finite likelihood per admitted hypothesis required')
        log_density = log_density+value
    evidence = float(logsumexp(log_density))
    return np.exp(log_density-evidence), evidence


def moments(values, weights):
    values = np.asarray(values)
    mean = np.einsum('h,h...->...', weights, values)
    variance = np.einsum('h,h...->...', weights, (values-mean)**2)
    return mean, np.sqrt(np.maximum(variance,0.))


def scale_summary(rows, weights):
    values = np.array([row['scale'] for row in rows])
    nodes, index = np.unique(values,return_inverse=True)
    mass = np.bincount(index,weights=weights,minlength=len(nodes))
    centers = np.cumsum(mass)-mass/2
    mean, sd = moments(values,weights)
    return dict(mean=float(mean),sd=float(sd),
                quantiles_05_50_95=np.interp([.05,.5,.95],centers,nodes).tolist(),
                quantile_method='Interpolated weighted quadrature CDF; discretization checked separately')


def prediction_metrics(expanded, weights, reference_fields, reference_wrench, action_indices):
    fields, _ = moments(expanded['fields'][:,action_indices],weights)
    wrench, _ = moments(expanded['wrenches'][:,action_indices],weights)
    from scripts.phystwin.sequential_fixture import points
    rest = points()
    aligned_squared = []
    for predicted, target in zip(fields, reference_fields):
        a = rest+predicted; b=rest+target
        a=a-a.mean(0); b=b-b.mean(0)
        u,_,vt=np.linalg.svd(a.T@b)
        correction=np.eye(3); correction[-1,-1]=np.linalg.det(u@vt)
        aligned_squared.append(np.mean((a@(u@correction@vt)-b)**2))
    return dict(coordinate_rmse_mm=float(1000*np.sqrt(np.mean((fields-reference_fields)**2))),
                aligned_shape_rmse_mm=float(1000*np.sqrt(np.mean(aligned_squared))),
                reaction_force_rmse_N=float(np.sqrt(np.mean((wrench[:,:3]-reference_wrench[:,:3])**2))),
                reaction_torque_rmse_Nm=float(np.sqrt(np.mean((wrench[:,3:]-reference_wrench[:,3:])**2))))
