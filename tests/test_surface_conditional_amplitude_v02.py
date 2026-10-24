import numpy as np
import pytest
from experiments.campaign_surface_conditional_amplitude_v02 import ConditionalAmplitudeReferenceV02,AmplitudePWLPlan
from experiments.campaign_surface_conditional_amplitude_v01 import LOG_WIDTH,BOUNDS


def test_same_positive_density_CDF_moments_and_raw_joint_omission():
    def ratio(z,labels):return np.broadcast_to(np.array([1.,2.])[None,None,:],(len(z),len(labels),2))
    e=ConditionalAmplitudeReferenceV02(None,[('p',np.array([.00025,.0005]))],['t'],log_ratio_predict=ratio)
    x=e.integrate(outer=32,inner=128,keep=True)
    u=x['unit_grid'];w=x['quadrature_probability'];a=x['inverse_amplitude']
    np.testing.assert_allclose(w@u,x['unit_mean'],atol=1e-13)
    np.testing.assert_allclose((u-w@u).T@(w[:,None]*(u-w@u)),x['unit_covariance'],atol=1e-13)
    y=a[:,None]*[1.,2.];np.testing.assert_allclose(w@y,x['predictive_mean'],atol=1e-16)
    np.testing.assert_allclose(np.sqrt(np.sum(w[:,None]*(y-w@y)**2,axis=0)),x['predictive_SD'],atol=1e-16)
    for d in(0,1):
        values=[e.cdf(d,q,outer=32,inner=128,log_evidence=x['log_evidence'])for q in np.linspace(0,1,129)]
        assert min(np.diff(values))>=-1e-12 and min(values)>=0 and max(values)<=1+1e-14
    lean=ConditionalAmplitudeReferenceV02(None,[('p',np.array([.00025,.0005]))],['t'],log_ratio_predict=ratio,retain_hypotheses=False).integrate(outer=32,inner=128,keep=True)
    assert 'unit_grid'not in lean and 'quadrature_probability'not in lean
    np.testing.assert_array_equal(lean['unit_mean'],x['unit_mean'])
    np.testing.assert_array_equal(lean['plugin_unit'],x['plugin_unit'])


def test_original_prior_evidence_and_independent_oracle_checks_preserved():
    def zero(z,labels):return np.zeros((len(z),len(labels),1))
    answer=ConditionalAmplitudeReferenceV02(None,[('p',np.zeros(1))],['t'],log_ratio_predict=zero,retain_hypotheses=False).qualify(plan=AmplitudePWLPlan(inner=(128,256),outer=(64,128)))
    assert answer['qualified']
    assert answer['diagnostics']['independent_adaptive']['passed']
    assert answer['diagnostics']['independent_adaptive']['CDF_checks_per_epsrel']==6
    np.testing.assert_allclose(answer['local_final']['unit_mean'],.5,atol=1e-10)
    np.testing.assert_allclose(answer['local_final']['unit_quantiles'],[[.05,.5,.95]]*2,atol=2e-5)


def test_changed_evidence_cannot_be_used_to_normalize_CDF():
    def zero(z,labels):return np.zeros((len(z),len(labels),1))
    e=ConditionalAmplitudeReferenceV02(None,[('p',np.zeros(1))],['t'],log_ratio_predict=zero)
    x=e.integrate(outer=16,inner=128,intrinsic=False)
    with pytest.raises(ValueError,match='same full'):e.cdf(0,.5,outer=16,inner=128,log_evidence=x['log_evidence']+.1)


def test_refinement_plan_rejects_duplicate_unsorted_or_single_levels():
    for kwargs in (dict(inner=(128,128,512)),dict(inner=(256,128)),dict(inner=(128,)),dict(inner=(32,64)),dict(outer=(128,128)),dict(outer=(256,128)),dict(outer=(0,128)),dict(moment_chunk=0)):
        with pytest.raises(ValueError):AmplitudePWLPlan(**kwargs)


def test_grouped_utility_refuses_empty_channels_and_nonfinite_moments():
    from experiments.campaign_surface_conditional_amplitude_v02 import grouped_utilities
    group=dict(log_ratio=np.array([-5.]),probability=np.ones(1),amplitude_mean=np.ones(1),amplitude_variance=np.ones(1))
    with pytest.raises(ValueError,match='Finite grouped'):grouped_utilities(group,lambda z,l:np.zeros((1,len(l),0)),['a'],['t'])
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises(FloatingPointError,match='Nonfinite grouped'):grouped_utilities(group,lambda z,l:np.full((1,len(l),1),1e300),['a'],['t'])
