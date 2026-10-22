import numpy as np
import pytest
from scipy.integrate import quad
from experiments.campaign_bar_noise_population import PopulationPosterior,inverse_integral,S,C_LO,C_HI
from experiments import campaign_nonlinear_bar_inverse as bar

def test_inverse_series_matches_independent_quad_and_cdf_direction():
    m=np.array([.0033,.0045,.0065])
    for power in (1,2,3):
        for z in (-3.,0.,2.,11.):
            upper=np.minimum(C_HI,m+z*S)
            expected=np.array([S*quad(lambda t:np.exp(-t*t/2)/(a+S*t)**power,-10,min(z,10),epsabs=1e-10,epsrel=1e-12)[0] for a in m])
            np.testing.assert_allclose(inverse_integral(m,upper,power),expected,rtol=2e-12,atol=1e-13)

def test_raw_likelihood_factorization_and_whole_prior_guard():
    obs=bar.predict(bar.TRUTH[None])[0]+np.array([.00007,-.00003,.00002,-.00009])
    p=PopulationPosterior(obs[None],rtol=1e-7,atol=1e-13)
    for u,k in ((.1,110),(.45,222),(.9,390)):
        E,m,d=p.parts(u);c=1/k
        direct=bar.log_likelihood(np.array([[u,(np.log(k)-bar.LOWER[1])/(bar.UPPER[1]-bar.LOWER[1])]]),obs)[0]+4*np.log(bar.SIGMA*np.sqrt(2*np.pi))
        np.testing.assert_allclose(direct,-d[0]-.5*((c-m[0])/S)**2,rtol=1e-12)
    assert p.series_bound<1e-14
    assert p.cdf('K',100)[0]<1e-12 and p.cdf('K',400)[0]>1-1e-12
    assert p.cdf('left',.005)[0]<1e-12 and p.cdf('left',.02)[0]>1-1e-12
    # Held-right bound depends on E; cannot reduce it to the left CDF.
    assert p.cdf('right',.018)[0]<p.cdf('left',.018)[0]
    with pytest.raises(ValueError,match='Entire-prior'):PopulationPosterior(np.full((1,4),.2))

def test_conditional_reduction_against_direct_nested_quadrature():
    obs=bar.predict(bar.TRUTH[None])[0]+np.array([.00007,-.00003,.00002,-.00009])
    p=PopulationPosterior(obs[None],rtol=1e-8,atol=1e-14)
    # Independent K integration in data-scaled c coordinate, exact 1/c prior.
    def outer(u,upper_k=400.):
        E,m,d=p.parts(u);a=float(m[0])
        lo=max(-12.,(1/upper_k-a)/S);hi=min(12.,(C_HI-a)/S)
        val=quad(lambda z:np.exp(-z*z/2)/(a+S*z),lo,hi,epsabs=1e-10,epsrel=1e-11)[0]
        return np.exp(p.shift[0]-d[0])*S*val
    Z=quad(outer,0,1,epsabs=1e-14,epsrel=1e-9)[0]
    expected=quad(lambda u:outer(u,220.),0,1,epsabs=1e-14,epsrel=1e-9)[0]/Z
    np.testing.assert_allclose(p.cdf('K',220.)[0],expected,atol=2e-9)
    mean,std=p.moments()
    assert np.all(std>0) and np.all(np.isfinite(mean))

def test_independent_original_2d_adapter_and_coverage_accounting():
    from experiments.run_bar_noise_population import direct_checks,coverage,compare
    # Deterministic synthetic test, NOT the prospective new noise seed/data.
    obs=bar.predict(bar.TRUTH[None])[0]+np.array([.00007,-.00003,.00002,-.00009])
    observations=np.repeat(obs[None],64,axis=0)
    p=PopulationPosterior(observations,rtol=1e-8,atol=1e-14)
    means,std=p.moments();truth=np.r_[bar.TRUTH,bar.predict(bar.TRUTH[None],[2.])[0]]
    summary={k:dict(mean=means[:,j],std=std[:,j],cdf_truth=p.cdf(k,truth[j]),q05=means[:,j]-2*std[:,j],q95=means[:,j]+2*std[:,j]) for j,k in enumerate(('E','K','left','right'))}
    check=direct_checks(observations,p,summary)
    assert check['passed'] and check['maximum_CDF_difference']<1e-8
    assert compare(summary,summary)['passed']
    metrics=coverage(summary)
    assert all(row['total']==64 for row in metrics.values())
    assert metrics['E']['exact_binomial_95percent_CI'][1]==1.
