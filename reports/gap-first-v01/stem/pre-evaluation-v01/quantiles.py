"""Marginal inverse CDFs for finite Gaussian mixtures, including exact atoms.

An interval endpoint at a point mass must equal that atom, not a bisection
midpoint one ULP above/below it. This module leaves historical inference frozen.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import ndtr, ndtri


def _prepare(weights, means, variances):
    w=np.asarray(weights,dtype=np.float64)
    mu=np.asarray(means,dtype=np.float64)
    var=np.asarray(variances,dtype=np.float64)
    if w.ndim!=1 or not len(w) or np.any(~np.isfinite(w)) or np.any(w<0):
        raise ValueError('Weights must be finite nonnegative one-dimensional values')
    if mu.ndim==1:mu=mu[:,None]
    if var.ndim==1:var=var[:,None]
    if mu.ndim!=2 or mu.shape!=var.shape or mu.shape[0]!=len(w):
        raise ValueError('Means/variances must match [component,target]')
    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(var)) or np.any(var<0):
        raise ValueError('Means must be finite and variances finite/nonnegative')
    active=w>0
    if not np.any(active):raise ValueError('At least one component must have positive weight')
    w=w[active];w=w/math.fsum(w.tolist())
    return w,mu[active],np.sqrt(var[active])


def gaussian_mixture_cdf(values, weights, means, variances, *, left_limit=False):
    """CDF, or its left limit, evaluated at one value per target coordinate."""
    w,mu,sd=_prepare(weights,means,variances)
    x=np.broadcast_to(np.asarray(values,dtype=np.float64),(mu.shape[1],))
    if np.any(np.isnan(x)):raise ValueError('CDF arguments cannot be NaN')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        z=np.divide(x-mu,sd,out=np.zeros_like(mu),where=sd>0)
    atoms=(x>mu) if left_limit else (x>=mu)
    components=np.where(sd>0,ndtr(z),atoms)
    return np.array([math.fsum((w*components[:,j]).tolist()) for j in range(mu.shape[1])])


def gaussian_mixture_quantiles(weights, means, variances, probabilities=(.05,.95)):
    """Return generalized inverse CDF inf{x:F(x)>=p}, shape [p,target].

    Zero variances are deterministic atoms, with duplicate atoms combined.
    Nonzero variance is never thresholded to zero. Probabilities must be finite
    and strictly inside (0,1). Weights need not initially sum to one.
    """
    w,mu,sd=_prepare(weights,means,variances)
    ps=np.asarray(probabilities,dtype=np.float64)
    if ps.ndim!=1 or np.any(~np.isfinite(ps)) or np.any((ps<=0)|(ps>=1)):
        raise ValueError('Probabilities must be finite and strictly inside (0,1)')
    result=np.empty((len(ps),mu.shape[1]))
    for j in range(mu.shape[1]):
        m,s=mu[:,j],sd[:,j];atom=s==0;locations=np.unique(m[atom])
        if np.all(atom):
            # Sorted cumulative mass gives O(H log H), not H full-CDF scans.
            order=np.argsort(m,kind='stable')
            values,first=np.unique(m[order],return_index=True)
            grouped=np.add.reduceat(w[order].astype(np.longdouble),first)
            cumulative=np.cumsum(grouped,dtype=np.longdouble)
            cumulative[-1]=1  # normalized total, guard final rounding only
            indices=np.searchsorted(cumulative,ps.astype(np.longdouble),side='left')
            result[:,j]=values[np.minimum(indices,len(values)-1)]
            continue
        def cdf(x,left=False):
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                z=np.divide(x-m,s,out=np.zeros_like(m),where=~atom)
            values=np.where(atom,(x>m) if left else (x>=m),ndtr(z))
            return math.fsum((w*values).tolist())
        # Explicit jump detection preserves exact representable atom endpoints.
        jumps=[(a,cdf(a,True),cdf(a)) for a in locations]
        for i,p in enumerate(ps):
            snapped=next((a for a,left,right in jumps if left<=p<=right),None)
            if snapped is not None:
                result[i,j]=snapped;continue
            if np.all(atom):
                # Boundary of a plateau: generalized inverse is its first atom.
                result[i,j]=next(a for a,_,right in jumps if right>=p);continue
            component_q=m+s*ndtri(p)
            lo,hi=float(component_q.min()),float(component_q.max())
            # Floating-point normal-CDF roundoff may move the theoretical bracket.
            span=max(hi-lo,float(s.max()),np.finfo(float).tiny)
            for _ in range(64):
                if cdf(lo)<p:break
                lo-=span;span*=2
            span=max(hi-lo,float(s.max()),np.finfo(float).tiny)
            for _ in range(64):
                if cdf(hi)>=p:break
                hi+=span;span*=2
            if not (cdf(lo)<p<=cdf(hi)):
                raise FloatingPointError('Could not bracket mixture quantile')
            for _ in range(160):
                mid=lo+(hi-lo)/2
                if mid==lo or mid==hi:break
                if cdf(mid)<p:lo=mid
                else:hi=mid
            result[i,j]=hi
    return result


def marginal_quantiles(posterior,T,probabilities=(.05,.95),offset=None):
    """Adapter for the existing posterior target interface; no source mutation."""
    target=posterior.target(T,offset)
    return gaussian_mixture_quantiles(posterior.weights,target.component_means,
                                      target.component_variances,probabilities)
