"""Independent dense hypotheses versus grouped moments; synthetic inputs only."""
import numpy as np
import pytest
from experiments.campaign_surface_probe_utility_v01 import covariance_utility
from experiments.campaign_surface_conditional_amplitude_v02 import (
    ConditionalAmplitudeReferenceV02,grouped_utilities)


def synthetic_groups():
    rng=np.random.default_rng(62043);z=np.array([-6.,-5.4,-4.8,-4.2]);amplitude=rng.uniform(1.3e-4,4.8e-4,size=(4,7))
    rho=np.array([.1,.3,.6,0.]);conditional=rng.dirichlet(np.ones(7),size=4)
    average=np.sum(conditional*amplitude,axis=1)
    variance=np.sum(conditional*(amplitude-average[:,None])**2,axis=1)
    grouped=dict(log_ratio=z,probability=rho,amplitude_mean=average,amplitude_variance=variance)
    labels=('A','B','C','T0','T1');coeff=rng.normal(size=(len(labels),5,3))
    def ratio(z,selected):
        feature=np.column_stack([np.ones(len(z)),np.exp(z),np.exp(2*z)])
        return np.einsum('bi,lqi->blq',feature,coeff[[labels.index(l)for l in selected]])
    return grouped,amplitude,conditional,labels,ratio


def test_grouped_covariance_matches_independent_dense_positive_hypotheses():
    grouped,a,conditional,labels,ratio=synthetic_groups()
    probability=(grouped['probability'][:,None]*conditional).ravel()
    dense=(a[:,:,None,None]*ratio(grouped['log_ratio'],labels)[:,None]).reshape(-1,len(labels),5)
    expected=[covariance_utility(probability,dense[:,j],dense[:,3:],noise_SD_m=20e-6)for j in range(3)]
    actual=grouped_utilities(grouped,ratio,labels[:3],labels[3:],sigma=20e-6)
    assert len(actual)==3
    for x,y in zip(actual,expected):
        for key in x:assert x[key]==pytest.approx(y[key],rel=5e-12,abs=1e-20)
    # Conditional variance is essential: using only conditional means is NOT
    # the dense posterior covariance when amplitude remains uncertain.
    wrong=grouped_utilities(dict(grouped,amplitude_variance=np.zeros(4)),ratio,labels[:3],labels[3:],sigma=20e-6)
    assert max(abs(x['expected_linear_target_MSE_reduction_m2']-y['expected_linear_target_MSE_reduction_m2'])for x,y in zip(actual,wrong))>1e-12


def test_single_ratio_group_keeps_within_amplitude_uncertainty():
    g=dict(log_ratio=np.array([-5.]),probability=np.ones(1),amplitude_mean=np.array([.0003]),amplitude_variance=np.array([4e-10]))
    def ratio(z,labels):return np.stack([np.full((len(z),1),1. if l=='A'else 2.)for l in labels],axis=1)
    score=grouped_utilities(g,ratio,['A'],['T'],sigma=20e-6)[0]
    v=4e-10;noise2=(20e-6)**2
    assert score['prior_target_MSE_m2']==pytest.approx(4*v,rel=1e-13)
    assert score['expected_linear_target_MSE_reduction_m2']==pytest.approx((2*v)**2/(v+noise2),rel=1e-13)
    assert score['Gaussian_observation_information_nats']==pytest.approx(.5*np.log1p(v/noise2),rel=1e-13)


def toy_ratio(r,labels):
    return np.stack([np.zeros((len(r),2))if l=='zero'else np.column_stack([np.ones(len(r)),1/r])for l in labels],axis=1)


def test_grouped_engine_mean_SD_match_all_dense_retained_nodes_and_prior():
    engine=ConditionalAmplitudeReferenceV02(toy_ratio,[('zero',np.zeros(2))],['target'])
    result=engine.integrate(outer=8,inner=128,keep=True)
    grouped=result['grouped'];u=result['unit_grid'];p=result['quadrature_probability']
    E=np.exp(np.log(2000)+u[:,0]*np.log(4));K=np.exp(np.log(5)+u[:,1]*np.log(6));dense=np.column_stack([1/E,1/K])
    mean=p@dense;variance=p@((dense-mean)**2)
    np.testing.assert_allclose(result['predictive_mean'],mean,rtol=2e-12,atol=2e-17)
    np.testing.assert_allclose(result['predictive_SD'],np.sqrt(variance),rtol=2e-12,atol=2e-17)
    assert np.isclose(grouped['probability'].sum(),1,atol=1e-12)
    assert np.all(grouped['amplitude_variance']>=0)
    bounds=np.array([[2000.,8000.],[5.,30.]]);width=np.log(bounds[:,1]/bounds[:,0]);expected_mean=(1/bounds[:,0]-1/bounds[:,1])/width
    expected_second=(bounds[:,0]**-2-bounds[:,1]**-2)/(2*width)
    np.testing.assert_allclose(mean,expected_mean,rtol=1e-9,atol=1e-15)
    np.testing.assert_allclose(variance,expected_second-expected_mean**2,rtol=1e-8,atol=1e-17)


def test_narrow_conditionals_large_response_offsets_and_exact_zero_variance():
    # Huge common response with tiny physical amplitude spread stresses the
    # centered within-group variance term; raw second moments would cancel.
    amplitude=.0003+np.array([-2.,-1.,1.,2.])*1e-12
    probability=np.array([.1,.2,.3,.4]);average=probability@amplitude
    variance=probability@((amplitude-average)**2)
    grouped=dict(log_ratio=np.array([-5.]),probability=np.ones(1),amplitude_mean=np.array([average]),amplitude_variance=np.array([variance]))
    values={'A':np.array([1e8,2e8]),'T':np.array([3e8,-1e8])}
    def ratio(z,labels):return np.broadcast_to(np.stack([values[l]for l in labels])[None],(len(z),len(labels),2))
    actual=grouped_utilities(grouped,ratio,['A'],['T'],sigma=20e-6)[0]
    dense=amplitude[:,None,None]*ratio(np.zeros(len(amplitude)),['A','T'])
    expected=covariance_utility(probability,dense[:,0],dense[:,1],noise_SD_m=20e-6)
    for key in actual:assert actual[key]==pytest.approx(expected[key],rel=2e-7,abs=1e-15)
    zero=grouped_utilities(dict(grouped,amplitude_variance=np.zeros(1)),ratio,['A'],['T'],sigma=20e-6)[0]
    assert all(zero[k]==0 for k in zero)
