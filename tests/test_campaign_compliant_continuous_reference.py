import numpy as np
import torch
import pytest
from scipy.special import ndtr
from scipy.integrate import cubature
from experiments import campaign_compliant_continuous_reference as reference
from experiments import campaign_compliant_patch_inverse as original

torch.set_num_threads(2)


def test_exact_vectorized_forward_random_prior_and_posterior_scale_extremes():
    rng=np.random.default_rng(2026121100)
    x=np.vstack([rng.uniform(size=(12,4)),np.zeros((1,4)),np.ones((1,4)),
        np.array([[.55,.4,.5,.4],[.6,.4,.5,.4],[.57,.39,.51,.41]])])
    theta=reference.physical(x,(0,1,2,3));fast=reference.ExactVectorForward()
    for program in ('A','B'):
        exact,_=original.forward(theta,program)
        np.testing.assert_allclose(fast(theta,program),exact,atol=2e-15,rtol=2e-11)
    observation=original.selected_observations(original.forward(original.TRUTH[None],'A')[0])[0]
    expected=original.condition(original.selected_observations(original.forward(theta,'A')[0]),observation,np.ones(len(x))/len(x))[0]
    ll=fast.log_likelihood(x,(0,1,2,3),observation);weights=np.exp(ll-ll.max());weights/=weights.sum()
    np.testing.assert_allclose(weights,expected,atol=2e-12,rtol=2e-10)


def test_slice_preserves_original_material_prior_and_fixed_nuisances():
    unit=np.array([[0.,0.],[1.,1.],[.5,.5]])
    theta=reference.physical(unit,(0,3))
    np.testing.assert_allclose(theta[:,[1,2]],np.tile(original.TRUTH[[1,2]],(3,1)),atol=1e-16)
    np.testing.assert_allclose(theta[:,0],[2000,8000,4000],rtol=1e-14)
    np.testing.assert_allclose(theta[:,3],[5,30,np.sqrt(150)],rtol=1e-14)
    with pytest.raises(ValueError):reference.physical(np.array([[-.1,0.]]),(0,3))


def test_independent_adaptive_cubature_matches_analytic_truncated_gaussian():
    mu=np.array([.42,.57]);sd=np.array([.08,.13])
    f=lambda x:np.exp(-.5*np.sum(((x-mu)/sd)**2,1))
    result=cubature(f,np.zeros(2),np.ones(2),rtol=1e-9,atol=1e-12,max_subdivisions=256)
    exact=np.prod(sd*np.sqrt(2*np.pi)*(ndtr((1-mu)/sd)-ndtr(-mu/sd)))
    assert result.status=='converged' and result.estimate==pytest.approx(exact,rel=1e-9)
    bound=.46;cdf=cubature(f,np.zeros(2),[bound,1.],rtol=1e-9,atol=1e-12,max_subdivisions=256).estimate/result.estimate
    expected=(ndtr((bound-mu[0])/sd[0])-ndtr(-mu[0]/sd[0]))/(ndtr((1-mu[0])/sd[0])-ndtr(-mu[0]/sd[0]))
    assert cdf==pytest.approx(expected,abs=1e-9)


def test_empirical_cdf_and_mcmc_fixed_independence_density_correction():
    np.testing.assert_array_equal(reference.empirical_cdf([.1,.2,.9],[0.,.1,.5,1.]),[0.,1/3,2/3,1.])
    # Zero likelihood on [0,1]^2: strongly nonuniform fitted proposal must NOT
    # replace the uniform prior. This is synthetic algorithm validation only.
    pilot=np.random.default_rng(1).uniform(.25,.75,(1000,2))
    draws,report=reference.continuous_mcmc(lambda x:np.zeros(len(x)),pilot,seed=19,warmup=200,draws=1800)
    assert np.max(np.abs(draws.mean((0,1))-.5))<.025
    assert np.max(np.abs(draws.var((0,1))-1/12))<.008
    assert report['all_initial_states_independent_prior']
