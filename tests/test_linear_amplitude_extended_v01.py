import numpy as np
import pytest
from experiments.campaign_surface_conditional_amplitude_v02 import ConditionalAmplitudeReferenceV02
from experiments.campaign_linear_amplitude_extended_v01 import LinearAmplitudeExtendedReferenceV01,ExtendedAmplitudePWLPlan


def engine(cls,zero=False):
    def ratio(z,labels):return np.broadcast_to((np.zeros(len(z))if zero else .4+.02*np.exp(z))[:,None,None],(len(z),len(labels),1))
    return cls(None,[('observe',np.array([80e-6 if not zero else 0.]))],['target'],log_ratio_predict=ratio,retain_hypotheses=True)


def same(old,new):
    assert type(old)==type(new)
    if isinstance(old,dict):
        assert old.keys()==new.keys()
        for k in old:same(old[k],new[k])
    else:np.testing.assert_array_equal(old,new)


@pytest.mark.parametrize('panels',[128,256,512])
@pytest.mark.parametrize('zero',[False,True])
def test_every_old_order_linear_moments_evidence_CDF_hypotheses_bit_identical(panels,zero):
    old=engine(ConditionalAmplitudeReferenceV02,zero).integrate(outer=8,inner=panels,keep=True)
    new=engine(LinearAmplitudeExtendedReferenceV01,zero).integrate(outer=8,inner=panels,keep=True)
    same(old,new)


def test_1024_constant_likelihood_retains_original_uniform_prior():
    e=engine(LinearAmplitudeExtendedReferenceV01,True);last=e.integrate(outer=32,inner=1024,keep=True)
    np.testing.assert_allclose(last['unit_mean'],[.5,.5],atol=2e-6)
    for d in range(2):
        for q in(.1,.5,.9):np.testing.assert_allclose(e.cdf(d,q,outer=32,inner=1024,log_evidence=last['log_evidence']),q,atol=2e-3)
    expected=-np.log(20e-6*np.sqrt(2*np.pi));np.testing.assert_allclose(last['log_evidence'],expected,atol=2e-14)


def test_new_resolution_preserves_independent_oracle_failure_gate(monkeypatch):
    import experiments.campaign_amplitude_adaptive_oracle_v01 as base
    monkeypatch.setattr(base,'adaptive_integral',lambda *a,**k:dict(log_integral=0.,estimated_outer_relative_error=0.,maximum_estimated_inner_relative_error=0.))
    result=engine(LinearAmplitudeExtendedReferenceV01,True).qualify(plan=ExtendedAmplitudePWLPlan(outer=(8,16)))
    assert not result['qualified']and not result['diagnostics']['independent_adaptive']['passed']
    assert result['diagnostics']['schema']=='conditional-amplitude-linear-extended-v01'
