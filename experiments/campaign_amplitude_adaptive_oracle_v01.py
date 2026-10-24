"""Independent direct-a adaptive integral for amplitude/ratio diagnostics.

Uses original Gaussian residuals divided by a, with no Gaussian-CDF change of
variables. This is a numerical cross-check, not a convergence certificate.
No observations, parameters or automatic execution are embedded.
"""
import numpy as np
from scipy.integrate import quad


def adaptive_integral(ratio_predict,observed,*,sigma,E_bounds=(2000.,8000.),K_bounds=(5.,30.),CDF=None,epsrel=1e-8):
    """ratio_predict receives a vector of z=log(K/E), and returns g(exp(z)).
    CDF optionally ('E',physical_threshold) or ('K',physical_threshold).

    Returns log evidence (or log unnormalized CDF numerator). Evaluate full
    and partial domains independently, then subtract their log integrals.
    """
    y=np.asarray(observed,float).reshape(-1);E0,E1=E_bounds;K0,K1=K_bounds
    if not y.size or not np.isfinite(y).all()or not np.isfinite([sigma,E0,E1,K0,K1,epsrel]).all()or sigma<=0 or not 0<epsrel<1 or not(0<E0<E1 and 0<K0<K1):raise ValueError('Finite data, positive noise and prior bounds')
    if CDF is not None and(CDF[0]not in('E','K')or not np.isfinite(CDF[1])or CDF[1]<=0):raise ValueError('Physical positive CDF threshold')
    z0=np.log(K0/E1);z1=np.log(K1/E0);breaks=[z0,z1,np.log(K0/E0),np.log(K1/E1)]
    if CDF is not None:
        kind,q=CDF
        breaks.extend([np.log(K0/q),np.log(K1/q)]if kind=='E'else[np.log(q/E0),np.log(q/E1)])
    breaks=sorted(set(v for v in breaks if z0<=v<=z1));inner_calls=0;max_inner_error=0.
    normalizer=-len(y)*np.log(sigma*np.sqrt(2*np.pi));prior_norm=np.log(E1/E0)*np.log(K1/K0)
    def integrand(z):
        nonlocal inner_calls,max_inner_error
        r=np.exp(z);lo=max(1/E1,r/K1);hi=min(1/E0,r/K0)
        if CDF is not None:lo=max(lo,1/CDF[1]if CDF[0]=='E'else r/CDF[1])
        if lo>=hi:return 0.
        g=np.asarray(ratio_predict(np.array([z])),float).reshape(-1)
        if g.shape!=y.shape or not np.isfinite(g).all():raise ValueError('Common finite observation vector')
        gg=float(g@g)
        if gg==0:return float(np.exp(-.5*np.sum((y/sigma)**2))*np.log(hi/lo))
        mean=float(g@y/gg);sd=sigma/np.sqrt(gg);mode=min(max(mean,lo),hi)
        peak=-.5*float(np.sum(((y-mode*g)/sigma)**2))
        # Resolve the Gaussian peak and the narrow one-sided boundary layer
        # even when the unconstrained Gaussian center lies outside the prior.
        width=sd/max(1.,abs((mode-mean)/sd));points=[mean+k*sd for k in(-8,-3,-1,0,1,3,8)]
        points.extend(mode+k*width for k in(-8,-3,-1,1,3,8));points=sorted(set(v for v in points if lo<v<hi))
        def original_a_density(a):
            return np.exp(-.5*np.sum(((y-a*g)/sigma)**2)-peak)/a
        value,error=quad(original_a_density,lo,hi,points=points,epsabs=1e-300,epsrel=epsrel,limit=300)
        inner_calls+=1;max_inner_error=max(max_inner_error,error/max(value,np.finfo(float).tiny))
        return float(np.exp(peak)*value)
    total=0.;error=0.
    for lo,hi in zip(breaks,breaks[1:]):
        value,err=quad(integrand,lo,hi,epsabs=1e-300,epsrel=epsrel,limit=300);total+=value;error+=err
    if total<=0 or not np.isfinite(total):raise FloatingPointError('Unresolved/underflowed independent adaptive integral')
    return dict(log_integral=float(normalizer+np.log(total/prior_norm)),
        estimated_outer_relative_error=float(error/total),maximum_estimated_inner_relative_error=float(max_inner_error),
        inner_integrations=inner_calls,independent_direct_a_quadrature=True,
        scope='QUADPACK error estimates only; numerical refinement and cross-method agreement still required')
