import numpy as np
from experiments.campaign_nonlinear_conditional_reference_v01 import NonlinearConditionalReference
from experiments.campaign_nonlinear_adaptive_oracle_v01 import nonlinear_adaptive_integral
from experiments.campaign_surface_conditional_amplitude_v01 import BOUNDS,LOG_WIDTH


def test_general_response_moments_match_explicit_dense_positive_measure():
    def ratio(z,labels):return np.zeros((len(z),len(labels),1))
    def full(u,labels):
        v=12e-6*(u[:,0]**2+np.sin(u[:,1]))
        return np.broadcast_to(v[:,None,None],(len(u),len(labels),1))
    engine=NonlinearConditionalReference(ratio,full,[('observe',np.array([8e-6]))],['target'],retain_hypotheses=True)
    result=engine.integrate(outer=12,inner=128,keep=True)
    p=result['quadrature_probability'];u=result['unit_grid'];v=full(u,['target']).ravel()
    np.testing.assert_allclose(result['predictive_mean'],[p@v],atol=1e-20)
    np.testing.assert_allclose(result['predictive_SD'],[np.sqrt(p@((v-p@v)**2))],atol=1e-20)
    assert np.min(np.diff(result['unit_CDF'],axis=1))>=0
    assert 'grouped' not in result


def test_independent_full_nonlinear_oracle_constant_and_parameter_density():
    # FullLL=3*unitE has exact evidence (e^3−1)/3 regardless of zero linear proposal.
    def zero(z):return np.zeros((len(z),1))
    def loglik(u):return 3*u[:,0]
    total=nonlinear_adaptive_integral(loglik,zero,[0.],sigma=20e-6,epsrel=1e-8)
    expected=np.log(np.expm1(3)/3)
    np.testing.assert_allclose(total['log_integral'],expected,atol=1e-10)
    q=np.sqrt(BOUNDS[0,0]*BOUNDS[0,1])
    part=nonlinear_adaptive_integral(loglik,zero,[0.],sigma=20e-6,CDF=('E',q),epsrel=1e-8)
    np.testing.assert_allclose(np.exp(part['log_integral']-total['log_integral']),np.expm1(1.5)/np.expm1(3),atol=1e-10)


def test_qualifier_rejects_independent_oracle_disagreement(monkeypatch):
    from experiments.campaign_surface_conditional_amplitude_v02 import AmplitudePWLPlan
    import experiments.campaign_nonlinear_adaptive_oracle_v01 as oracle
    def ratio(z,labels):return np.zeros((len(z),len(labels),1))
    def full(u,labels):return np.zeros((len(u),len(labels),1))
    engine=NonlinearConditionalReference(ratio,full,[('observe',np.array([0.]))],['target'])
    def wrong(*args,**kwargs):return dict(log_integral=0.,estimated_outer_relative_error=0.,maximum_estimated_inner_relative_error=0.)
    monkeypatch.setattr(oracle,'nonlinear_adaptive_integral',wrong)
    result=engine.qualify(plan=AmplitudePWLPlan(inner=(128,256),outer=(8,16)))
    assert not result['qualified']
    assert not result['diagnostics']['independent_adaptive']['passed']


def test_duplicate_reset_observations_remain_independent_likelihood_terms():
    def ratio(z,labels):return np.zeros((len(z),len(labels),1))
    def full(u,labels):return np.broadcast_to((3e-6*u[:,0]**2)[:,None,None],(len(u),len(labels),1))
    u=np.array([[.1,.3],[.9,.8]])
    one=NonlinearConditionalReference(ratio,full,[('same',np.array([7e-6]))],['target'])
    two=NonlinearConditionalReference(ratio,full,[('same',np.array([7e-6])),('same',np.array([7e-6]))],['target'])
    np.testing.assert_allclose(two.full_loglik(u),2*one.full_loglik(u),rtol=0,atol=1e-14)


def test_structured_residual_matches_generic_and_bounds_every_full_callback():
    calls=[]
    def ratio(z,labels):return np.broadcast_to((1+.03*z)[:,None,None],(len(z),len(labels),1))
    def residual(u,labels):
        calls.append(len(u));assert len(u)<=32
        return np.broadcast_to((2e-6*u[:,0]**2*u[:,1])[:,None,None],(len(u),len(labels),1))
    def total(u,labels):
        calls.append(len(u));assert len(u)<=32
        a=np.exp(-np.log(BOUNDS[0,0])-u[:,0]*LOG_WIDTH[0]);z=np.log(BOUNDS[1,0])+u[:,1]*LOG_WIDTH[1]+np.log(a)
        return a[:,None,None]*ratio(z,labels)+2e-6*(u[:,0]**2*u[:,1])[:,None,None]
    kwargs=dict(observations=[('observe',np.array([150e-6]))],target_labels=['target'],retain_hypotheses=True,evaluation_chunk=32)
    generic=NonlinearConditionalReference(ratio,total,**kwargs)
    structured=NonlinearConditionalReference(ratio,None,residual_predict=residual,**kwargs)
    a=generic.integrate(outer=8,inner=128,moment_chunk=16,keep=True)
    b=structured.integrate(outer=8,inner=128,moment_chunk=16,keep=True)
    for key in('log_evidence','predictive_mean','predictive_SD','unit_CDF','unit_mean','quadrature_probability','log_likelihood'):
        np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=1e-12)
    assert max(calls)<=32
