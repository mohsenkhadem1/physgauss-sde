import numpy as np
from scipy.integrate import quad
from experiments.campaign_c38_bar_posterior import conditional_nodes,BarPosteriorBatch,C_LO,C_HI
from experiments.campaign_c38_bar_reference import SIGMA_M


def test_finite_prior_truncation_both_boundaries_and_far_outside():
    # Exact prior limits remain active even when the Gaussian center is outside.
    for m in (.001,.0025,.005,.01,.012):
        c,w,logmass=conditional_nodes(m,.0002,order=48)
        assert (c>C_LO).all() and (c<C_HI).all()
        # Normalize scalar density around its constrained mode for independent quad.
        mode=np.clip(m,C_LO,C_HI);shift=((mode-m)/.0002)**2/2
        Z=quad(lambda x:np.exp(shift-((x-m)/.0002)**2/2)/x,C_LO,C_HI,epsabs=1e-10,epsrel=1e-10)[0]
        estimated=np.exp(shift+logmass)*np.sqrt(2*np.pi)*.0002*np.sum(w/c)
        assert abs(estimated/Z-1)<3e-5


def test_left_only_preserves_material_prior_and_conditional_K_matches_independent():
    p=BarPosteriorBatch([[.005]],np.array([1.]),np.array([0]),rtol=1e-8,conditional_order=48)
    E=np.exp(np.log(4500)+.37*np.log(2))
    np.testing.assert_allclose(p.cdf('E',E),.37,atol=2e-10)
    norm=quad(lambda c:np.exp(-.5*((c-.005)/SIGMA_M)**2)/c,C_LO,C_HI,epsabs=1e-12)[0]
    expected=1-quad(lambda c:np.exp(-.5*((c-.005)/SIGMA_M)**2)/c,C_LO,1/205.,epsabs=1e-12)[0]/norm
    np.testing.assert_allclose(p.cdf('K',205),expected,atol=2e-6)
    theta,w,diag=p.quadrature_cloud(0,outer_order=16,conditional_order=48)
    assert diag['normalizer_relative_error']<1e-9
    assert abs(np.sum(w*np.log(theta[:,0]))-np.log(np.sqrt(4500*9000)))<1e-10
    mean,std=p.moments();assert np.isfinite(mean).all() and (std>0).all()
