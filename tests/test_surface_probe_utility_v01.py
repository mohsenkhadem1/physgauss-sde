import numpy as np
import pytest
from experiments.campaign_surface_probe_utility_v01 import covariance_utility,select_probe


def test_independent_channel_has_no_target_utility_despite_variance():
    draws=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])*1e-3;weights=np.full(4,.25)
    target=draws[:,:1];candidates=[draws[:,1:]*10,draws[:,:1]]
    selected,scores=select_probe(weights,candidates,target,noise_SD_m=1e-3)
    assert selected==1
    assert scores[0]['expected_linear_target_MSE_reduction_m2']<1e-25
    assert scores[0]['Gaussian_observation_information_nats']>scores[1]['Gaussian_observation_information_nats']
    assert abs(scores[1]['expected_linear_target_MSE_reduction_m2']-5e-7)<1e-20


def test_weighted_covariance_translation_and_duplicate_channels():
    x=np.array([0.,1.,2.])[:,None]*1e-3;w=np.array([.2,.3,.5]);sigma=2e-4
    a=covariance_utility(w,x,x,noise_SD_m=sigma)
    b=covariance_utility(w,x+2,x-5,noise_SD_m=sigma)
    assert np.isclose(a['expected_linear_target_MSE_reduction_m2'],b['expected_linear_target_MSE_reduction_m2'],rtol=1e-10)
    c=covariance_utility(w,np.repeat(x,2,axis=1),x,noise_SD_m=sigma)
    assert a['expected_linear_target_MSE_reduction_m2']<c['expected_linear_target_MSE_reduction_m2']<a['prior_target_MSE_m2']


def test_no_invalid_noise_or_weights():
    x=np.zeros((2,1))
    with pytest.raises(ValueError):covariance_utility([.5,.5],x,x,noise_SD_m=0)
    with pytest.raises(ValueError):covariance_utility([1.,1.],x,x,noise_SD_m=1)
    with pytest.raises(ValueError):covariance_utility([.5,.5],x,np.empty((2,0)),noise_SD_m=1)
