"""Synthetic density controls only; no physical study data or observations."""
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from experiments.campaign_positive_amplitude_pwl_v01 import PositiveAmplitudePWL


def independent(mu,sd,lo,hi,cut=None):
    mode=min(max(mu,lo),hi);peak=-.5*((mode-mu)/sd)**2
    width=sd/max(1,abs((mode-mu)/sd))
    points=[mu+k*sd for k in(-8,-3,-1,0,1,3,8)]+[mode+k*width for k in(-20,-8,-3,-1,1,3,8,20)]
    def integral(a,b):
        if a==b:return 0.
        return quad(lambda x:np.exp(-.5*((x-mu)/sd)**2-peak)/x,a,b,points=sorted(set(x for x in points if a<x<b)),epsabs=1e-14,epsrel=1e-11,limit=200)[0]
    mass=integral(lo,hi)
    return np.log(mass)+peak, None if cut is None else integral(cut,hi)/mass


def test_positive_mass_exact_PWL_primitive_and_endpoints():
    density=PositiveAmplitudePWL([.0003,-.0002,.001],[.00001,.00001,.00002],[.000125,.0002,.000125],[.0005,.00021,.0002])
    assert np.all(density.density_weights>0)
    np.testing.assert_allclose(density.mass0,density.panel_mass.sum(1),atol=1e-12,rtol=1e-15)
    np.testing.assert_allclose(logsumexp(density.log_weights,axis=1),density.log_kernel_mass+np.log(density.mass0),atol=1e-12,rtol=0)
    np.testing.assert_array_equal(density.cdf(density.lower),0)
    np.testing.assert_array_equal(density.tail(density.lower),1)
    np.testing.assert_array_equal(density.cdf(density.upper),1)
    np.testing.assert_array_equal(density.tail(density.upper),0)
    assert density.diagnostics()['continuous_CDF'] and not density.diagnostics()['atomic_CDF']
    # At a panel boundary, continuous mass equals precisely the same positive
    # Gauss hypothesis mass on preceding panels (not an atomic-node CDF).
    boundary=64;cut=1/density.h[:,boundary]
    expected=density.density_weights[:,:2*boundary].sum(1)/density.mass0
    np.testing.assert_allclose(density.cdf(cut),expected,atol=2e-10,rtol=0)


def test_continuous_monotone_complementary_CDF_including_40sigma_tails():
    mu=np.array([.0003,-.0002,.001,0]);sd=np.array([.00001,.00001,.00002,np.inf]);lo=np.array([.000125,.0002,.000125,.000125]);hi=np.array([.0005,.00021,.0002,.0005])
    for N in(128,256,512):
        density=PositiveAmplitudePWL(mu,sd,lo,hi,panels=N)
        cuts=lo[None]+(hi-lo)[None]*np.linspace(0,1,513)[:,None]
        C=np.array([density.cdf(q)for q in cuts]);T=np.array([density.tail(q)for q in cuts])
        assert np.min(np.diff(C,axis=0))>=-1e-12
        assert np.max(np.diff(T,axis=0))<=1e-12
        np.testing.assert_allclose(C+T,1,atol=3e-13,rtol=0)
        assert C.max()<=1+1e-12 and T.min()>=0
        # Continuous primitive is not the step-CDF of finite hypotheses.
        index=80;q=(density.amplitude[0,index]+density.amplitude[0,index+1])/2
        q2=np.nextafter(q,density.upper[0]);assert density.cdf([q2,lo[1],lo[2],lo[3]])[0]>=density.cdf([q,lo[1],lo[2],lo[3]])[0]


def test_original_a_independent_mass_tail_and_refinement():
    cases=[(.0003,.00001,.000125,.0005,.0003),(-.0002,.00001,.0002,.00021,.0002001),(.001,.00002,.000125,.0002,.000199)]
    errors=[]
    for N in(128,256,512):
        worst=0.
        for mu,sd,lo,hi,cut in cases:
            d=PositiveAmplitudePWL([mu],[sd],[lo],[hi],panels=N);logmass,tail=independent(mu,sd,lo,hi,cut)
            error=max(abs(logsumexp(d.log_weights[0])-logmass),abs(d.tail([cut])[0]-tail));worst=max(worst,error)
        errors.append(worst)
    assert errors[-1]<1e-4
    assert errors[2]<errors[1]<errors[0]
    from scipy.optimize import brentq
    mu,sd,lo,hi,_=cases[0];d=PositiveAmplitudePWL([mu],[sd],[lo],[hi],panels=512)
    median=brentq(lambda q:d.tail([q])[0]-.5,lo,hi,xtol=1e-16)
    assert abs(independent(mu,sd,lo,hi,median)[1]-.5)<1e-4


def test_gzero_exact_logprior_and_gauss_hypothesis_moments():
    lo=np.array([.000125,.0002]);hi=np.array([.0005,.00021]);d=PositiveAmplitudePWL([0,0],[np.inf,np.inf],lo,hi,panels=128)
    weights=np.exp(d.log_weights);mass=np.log(hi/lo)
    np.testing.assert_allclose(weights.sum(1),mass,atol=1e-15,rtol=1e-14)
    q=np.sqrt(lo*hi);np.testing.assert_allclose(d.tail(q),.5,atol=1e-14)
    np.testing.assert_allclose(d.cdf(q),.5,atol=1e-14)
    np.testing.assert_allclose((weights*d.amplitude).sum(1),hi-lo,atol=1e-14,rtol=1e-10)


def test_no_clipping_invalid_inputs_refused():
    import pytest
    for args in [([0],[0],[1],[2]),([0],[1],[0],[2]),([np.nan],[1],[1],[2]),([0],[1],[2],[1])]:
        with pytest.raises(ValueError):PositiveAmplitudePWL(*args)
    with pytest.raises(ValueError):PositiveAmplitudePWL([0],[1],[1],[2],panels=64)
