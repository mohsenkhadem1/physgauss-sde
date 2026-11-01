"""Resolve continuous common-stiffness scale using its exact linear wrench law.

This changes integration coordinates, not the model, observations or noise.
Conditional normal quadrature is reweighted by the log-uniform 1/s prior.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_ndtr, ndtr, logsumexp
from scipy.stats import truncnorm

LOWER,UPPER=.75,1.5


def log_normal_mass(alpha,beta):
    if not np.isfinite([alpha,beta]).all() or alpha>=beta:
        raise ValueError('Finite ordered standardized scale bounds required')
    if alpha>=0:
        high,low=log_ndtr(-alpha),log_ndtr(-beta)
    elif beta<=0:
        high,low=log_ndtr(beta),log_ndtr(alpha)
    else:
        return float(np.log(ndtr(beta)-ndtr(alpha)))
    return float(high+np.log(-np.expm1(low-high)))


def integrate(A,B,log_at_reference,reference,count=25):
    """Integrate exp(L(s)) dlog(s)/log(upper/lower).

    L(s)-L(reference)=(B-A*reference)*(s-reference)-A/2*(s-reference)^2.
    The caller evaluates L(reference) from the original whitened residuals.
    """
    if count not in (25,49,97) or not np.isfinite([A,B,log_at_reference,reference]).all() or A<0:
        raise ValueError('Finite quadratic likelihood and declared quadrature required')
    if not LOWER<=reference<=UPPER:
        raise ValueError('Reference scale must be in prior support')
    nodes,weights=leggauss(count);u=(nodes+1)/2;weights=weights/2
    if A*(UPPER-LOWER)**2<=.5:
        scales=np.exp(np.log(LOWER)+u*np.log(UPPER/LOWER))
        delta=scales-reference
        logweights=np.log(weights)+log_at_reference+(B-A*reference)*delta-.5*A*delta**2
        logZ=float(logsumexp(logweights));posterior_weights=np.exp(logweights-logZ)
        method='log-prior coordinates for broad likelihood'
    else:
        mu=B/A;sigma=1/np.sqrt(A)
        alpha,beta=(LOWER-mu)/sigma,(UPPER-mu)/sigma
        scales=truncnorm.ppf(u,alpha,beta,loc=mu,scale=sigma)
        if not np.isfinite(scales).all() or np.any(np.diff(scales)<0) or scales.min()<LOWER-1e-12 or scales.max()>UPPER+1e-12:
            raise ValueError('Conditional normal inverse-CDF failed bounds/monotonicity')
        scales=np.clip(scales,LOWER,UPPER)
        reweighted=weights/scales
        posterior_weights=reweighted/reweighted.sum()
        logZ=float(log_at_reference+.5*((reference-mu)/sigma)**2+np.log(sigma)+.5*np.log(2*np.pi)
                   +log_normal_mass(alpha,beta)+np.log(reweighted.sum())-np.log(np.log(UPPER/LOWER)))
        method='conditional truncated-normal coordinates, 1/scale prior correction'
    mean=float(posterior_weights@scales)
    second=float(posterior_weights@(scales**2))
    if not np.isfinite([logZ,mean,second]).all():
        raise ValueError('Nonfinite conditional posterior integral')
    return dict(scales=scales,weights=posterior_weights,log_evidence=logZ,mean=mean,second_moment=second,method=method)


def reference_from_coefficients(A,B):
    A=np.asarray(A);B=np.asarray(B)
    return np.clip(np.divide(B,A,out=np.full_like(B,(LOWER+UPPER)/2,dtype=float),where=A>0),LOWER,UPPER)
