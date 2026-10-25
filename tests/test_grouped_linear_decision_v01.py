"""Independent dense positive-bank parity; no FE, actual data or RNG."""
import numpy as np
import pytest
from experiments import campaign_grouped_linear_decision_v01 as g
from experiments import campaign_layout_mixture_decision_v01 as dense

def fixture():
    b=g.GroupedLinearBank([.2,.3,.5],[0,0,1],[[.1,.2,.4],[.15,.3,.5],[.2,.4,.8]],[[.1,.4,.5],[.5,.3,.2],[.2,.6,.2]],[[1.,-.4],[.3,.6],[-.6,1.]],[[1.,2.,-.3],[.5,-1.,1.],[2.,.4,-.2]])
    b.target_ratio=np.tile(b.target_ratio,(1,16)) #48target coordinates, all retained
    p=(b.row_probability[:,None]*b.conditional).ravel();a=b.amplitude;A=(a[:,:,None]*b.candidate_ratio[:,None,:]).reshape(-1,2);T=(a[:,:,None]*b.target_ratio[:,None,:]).reshape(-1,48)
    return b,dense.PredictionBank(p,np.repeat(b.category,3),A,T)

def test_grouped_prior_48_style_moments_equal_full_positive_bank():
    b,d=fixture();r=b.prior_moments();mu=d.probability@d.target;var=d.probability@((d.target-mu)**2);np.testing.assert_allclose(r['mean'],mu,rtol=1e-14,atol=1e-15);np.testing.assert_allclose(r['SD'],np.sqrt(var),rtol=1e-14)
    dec=dense.variance_decomposition(d);assert r['total_variance']==pytest.approx(dec['total'],abs=1e-15);assert r['between_layout_variance']==pytest.approx(dec['between_layout'],abs=1e-15)

def test_two_pass_conditional_moments_exact_dense_at_every_outcome():
    b,d=fixture();y=np.array([-.4,0.,.35,1.])
    for action in range(2):
        lp,v=g.outcome_risk(b,action,y,.1,row_chunk=1);ld,vd=dense.outcome_risk(d,action,y,.1,latent_chunk=2);np.testing.assert_allclose(lp,ld,atol=2e-14);np.testing.assert_allclose(v,vd,atol=1e-15,rtol=1e-13)

def test_full_outcome_integral_and_risk_rank_equal_dense_without_target_matrix():
    b,d=fixture()
    for action in range(2):
        a=g.expected_risk(b,action,.1,order=8,row_chunk=1,outcome_chunk=3);z=dense.expected_risk(d,action,.1,order=8,outcome_chunk=3)
        for name in('expected_target_variance','prior_target_variance','predictive_mass','omitted_risk_bound'):assert a[name]==pytest.approx(z[name],rel=2e-13,abs=1e-15)

def test_zero_rows_constant_target_and_narrow_amplitude_preserve_mass():
    b=g.GroupedLinearBank([1.,0.],[0,1],[[1.,1.+1e-9],[2.,3.]],[[.5,.5],[.3,.7]],[[1.],[100.]],[[2.],[2.]])
    lp,v=g.outcome_risk(b,0,np.array([1.,1.2]),.1);assert np.isfinite(lp).all()and np.all(v>=0)and np.max(v)<1e-15
    with pytest.raises(ValueError):g.GroupedLinearBank([1e308,1e308],[0,1],[[1.],[2.]],[[1.],[1.]],[[1.],[1.]],[[1.],[1.]])


def test_new_decision_quadrature_retains_original_prior_and_matches_reference():
    from experiments.campaign_surface_conditional_amplitude_v02 import ConditionalAmplitudeReferenceV02
    def ratio(r,labels):return np.zeros((len(r),len(labels),1))
    refs=[ConditionalAmplitudeReferenceV02(ratio,[('prefix',[0.])],['target'],retain_hypotheses=False)for _ in range(4)]
    finals=[ref.integrate(outer=512,inner=512,intrinsic=False)for ref in refs]
    bank,meta=g.decision_bank(refs,['action'],['target'],outer=512,amplitude_order=16)
    assert g.reference_moment_checks(meta,finals)['passed']
    np.testing.assert_allclose(meta['layout_probability'],[.25]*4,atol=1e-14)
    assert bank.hypotheses==4*1536*16 and bank.prior_moments()['total_variance']==0
    for value in meta['log_evidence']:assert value==pytest.approx(-np.log(20e-6*np.sqrt(2*np.pi)),abs=1e-12)
