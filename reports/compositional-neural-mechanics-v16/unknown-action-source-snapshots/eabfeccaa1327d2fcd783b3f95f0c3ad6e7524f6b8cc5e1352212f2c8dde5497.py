"""Local Gaussian mechanics uncertainty and linearized action scoring.

Inputs are sensitivities of predicted observations, not held-out true responses.
This approximation is not a calibrated posterior and cannot resolve a genuinely
unobserved direction. Coverage must be measured independently at the object level.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve


def covariance_from_jacobian(observation_jacobian,prior_precision):
    """J must be whitened by declared observation noise, in fixed latent units."""
    J=np.asarray(observation_jacobian,float);prior=np.asarray(prior_precision,float)
    if J.ndim!=2 or prior.shape!=(J.shape[1],J.shape[1]) or not np.isfinite(J).all() or not np.isfinite(prior).all():raise ValueError('Finite compatible sensitivities/precision required')
    if not np.allclose(prior,prior.T,rtol=1e-12,atol=1e-12):raise ValueError('Symmetric prior precision required')
    cho_factor(prior,lower=True) # A proper prior, not an invented numerical jitter.
    precision=prior+J.T@J;factor=cho_factor(precision,lower=True);covariance=cho_solve(factor,np.eye(len(prior)))
    covariance=(covariance+covariance.T)/2
    return dict(covariance=covariance,precision=precision,precision_condition=float(np.linalg.cond(precision)),
                observation_singular_values=np.linalg.svd(J,compute_uv=False),
                scope='Local linear-Gaussian approximation with explicit prior; not a calibrated physical posterior')


def predictive_variance(target_jacobian,covariance,discrepancy_variance=0.):
    """Diagonal functional variance; target rows must share stated physical units."""
    A=np.asarray(target_jacobian,float);C=np.asarray(covariance,float);noise=np.asarray(discrepancy_variance,float)
    if A.ndim!=2 or not len(A) or C.shape!=(A.shape[1],A.shape[1]) or not np.isfinite(A).all() or not np.isfinite(C).all():raise ValueError('Finite nonempty target sensitivities and covariance required')
    if not np.allclose(C,C.T,rtol=1e-12,atol=1e-14):raise ValueError('Symmetric covariance required')
    eigenvalues=np.linalg.eigvalsh(C)
    if eigenvalues.min() < -1e-12*max(float(np.max(abs(eigenvalues))),1e-30):raise ValueError('Positive semidefinite covariance required')
    if noise.shape not in ((),(len(A),)):raise ValueError('Discrepancy variance must be scalar or one value per target row')
    if not np.isfinite(noise).all() or (noise<0).any():raise ValueError('Nonnegative finite discrepancy variance required')
    variance=np.einsum('ij,jk,ik->i',A,C,A)
    if (variance<-1e-10).any():raise ValueError('Invalid negative functional variance')
    return np.maximum(variance,0)+noise


def rank_position_probes(covariance,target_jacobian,candidates):
    """Equal-cost probes: expected reduction of mean target variance.

Candidate matrices are noise-whitened measurement sensitivities evaluated at
the SAME current inferred state. Target commands/query positions are public
desired tasks. No candidate outcome or target truth is an input.
"""
    C=np.asarray(covariance,float);A=np.asarray(target_jacobian,float)
    if not np.allclose(C,C.T,rtol=1e-12,atol=1e-12):raise ValueError('Symmetric covariance required')
    precision=cho_solve(cho_factor(C,lower=True),np.eye(len(C)))
    before=float(np.mean(predictive_variance(A,C)));rows=[]
    for name,jacobian in candidates.items():
        J=np.asarray(jacobian,float)
        if J.ndim!=2 or J.shape[1]!=len(C) or not np.isfinite(J).all():raise ValueError('Compatible finite candidate sensitivities required')
        after_covariance=cho_solve(cho_factor(precision+J.T@J,lower=True),np.eye(len(C)))
        after=float(np.mean(predictive_variance(A,after_covariance)))
        rows.append(dict(action=name,expected_mean_variance_before=before,expected_mean_variance_after=after,
                         expected_reduction=before-after))
    return sorted(rows,key=lambda row:(-row['expected_reduction'],row['action']))
