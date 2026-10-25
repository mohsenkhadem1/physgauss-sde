import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from experiments.campaign_nonlinear_mis_pwl_v01 import NonlinearMISPWL


def make(ll,*,panels=512,sd=.08,peak=7.,chunk=53):
    calls=[]
    def callback(a,z):
        calls.append(len(a));return ll(a)
    p=NonlinearMISPWL(np.array([.45]),np.array([sd]),np.array([.1]),np.array([1.]),log_ratio=np.array([0.]),linear_peak=np.array([peak]),full_loglik=lambda u:None,full_loglik_az=callback,panels=panels,chunk=chunk)
    assert max(calls)<=chunk
    return p


@pytest.mark.parametrize('sd',[.08,np.inf])
def test_nonzero_peak_and_zero_signal_original_measure(sd):
    p=make(lambda a:np.full_like(a,3.),sd=sd)
    logmass=logsumexp(p.log_weights[0])+7
    assert abs(logmass-(3+np.log(np.log(10))))<3e-5
    cuts=np.linspace(.1,1,501)
    cdf=np.array([p.cdf(np.array([q]))[0]for q in cuts])
    tail=np.array([p.tail(np.array([q]))[0]for q in cuts])
    assert np.max(abs(cdf-np.log(cuts/.1)/np.log(10)))<3e-5
    assert np.all(np.diff(cdf)>=0)
    assert np.max(abs(cdf+tail-1))<2e-14
    assert cdf[0]==0 and cdf[-1]==1


def test_shifted_nonlinear_likelihood_against_original_a_integral():
    # Full likelihood peak far outside the narrow linear proposal.
    ll=lambda a:2.-.5*((a-.82)/.055)**2
    p=make(ll,sd=.012)
    mass=np.exp(logsumexp(p.log_weights[0])+7)
    exact=quad(lambda a:np.exp(ll(a))/a,.1,1,epsabs=1e-12,points=[.82])[0]
    assert abs(mass/exact-1)<5e-5
    for q in(.6,.8,.9):
        expected=quad(lambda a:np.exp(ll(a))/a,.1,q,epsabs=1e-12)[0]/exact
        assert abs(p.cdf(np.array([q]))[0]-expected)<1e-4
    # Same positive finite quadrature atoms define moments independently.
    weights=np.exp(p.log_weights[0]-logsumexp(p.log_weights[0]))
    expected=quad(lambda a:np.exp(ll(a)),.1,1,epsabs=1e-12)[0]/exact
    assert abs(weights@p.amplitude[0]-expected)<5e-5


def test_mixture_prior_floor_bounds_scaled_knot_density():
    p=make(lambda a:np.zeros_like(a),sd=.001,peak=0.)
    actual=np.exp(p.row_shift[:,None,None])*p.h
    assert np.max(actual)<=np.log(10)*(1+2e-14)
    assert p.amplitude.shape==(1,2048)
    assert np.isfinite(p.log_weights).all()


def test_invalid_callback_and_chunk_fail_closed():
    with pytest.raises(ValueError):make(lambda a:a,chunk=0)
    with pytest.raises(FloatingPointError):make(lambda a:np.full_like(a,np.nan))


def test_engine_constant_likelihood_original_2d_prior_and_dense_moments():
    from experiments.campaign_nonlinear_conditional_reference_v03 import NonlinearConditionalReferenceV03
    def ratio(z,labels):return np.zeros((len(z),len(labels),1))
    def full(u,labels):return np.broadcast_to((4e-6*u[:,0]**2)[:,None,None],(len(u),len(labels),1))
    # Engine requires observations: constant response observation, arbitrary target.
    def response(u,labels):
        return np.stack([np.zeros((len(u),1))if label=='observe'else full(u,[label])[:,0]for label in labels],axis=1)
    engine=NonlinearConditionalReferenceV03(ratio,response,[('observe',np.array([0.]))],['target'],retain_hypotheses=True)
    result=engine.integrate(outer=16,inner=256,keep=True)
    expected_logZ=-np.log(20e-6*np.sqrt(2*np.pi))
    np.testing.assert_allclose(result['log_evidence'],expected_logZ,atol=2e-14)
    p=result['quadrature_probability'];u=result['unit_grid'];v=response(u,['target']).ravel()
    np.testing.assert_allclose(result['predictive_mean'],[p@v],atol=1e-20)
    np.testing.assert_allclose(result['predictive_SD'],[np.sqrt(p@((v-p@v)**2))],atol=1e-20)
    np.testing.assert_allclose(result['unit_mean'],[.5,.5],atol=3e-6)
    assert np.min(np.diff(result['unit_CDF'],axis=1))>=0
    state=engine._state(16,256)
    primitive=state['primitive']
    np.testing.assert_allclose(primitive.panel_mass.sum(axis=(1,2)),primitive.density_weights.sum(axis=1),rtol=2e-15)


def test_original_qualification_rejects_independent_oracle_disagreement(monkeypatch):
    from experiments.campaign_nonlinear_conditional_reference_v03 import NonlinearConditionalReferenceV03
    from experiments.campaign_surface_conditional_amplitude_v02 import AmplitudePWLPlan
    import experiments.campaign_nonlinear_adaptive_oracle_v01 as oracle
    def ratio(z,labels):return np.zeros((len(z),len(labels),1))
    def full(u,labels):return np.zeros((len(u),len(labels),1))
    engine=NonlinearConditionalReferenceV03(ratio,full,[('observe',np.array([0.]))],['target'])
    monkeypatch.setattr(oracle,'nonlinear_adaptive_integral',lambda *args,**kwargs:dict(log_integral=0.,estimated_outer_relative_error=0.,maximum_estimated_inner_relative_error=0.))
    result=engine.qualify(plan=AmplitudePWLPlan(inner=(128,256),outer=(8,16)))
    assert not result['qualified']
    assert not result['diagnostics']['independent_adaptive']['passed']


def test_exact_zero_mass_proposal_stream_contributes_zero_not_failure():
    p=make(lambda a:-.5*((a-.82)/.001)**2,sd=.0001)
    assert p.panel_mass[0,0].sum()==0
    assert p.panel_mass[0,1].sum()>0
    cuts=np.linspace(.1,1,301)
    cdf=np.array([p.cdf(np.array([q]))[0]for q in cuts])
    tail=np.array([p.tail(np.array([q]))[0]for q in cuts])
    assert np.all(np.diff(cdf)>=0)
    np.testing.assert_allclose(cdf+tail,1,atol=2e-14)
    assert cdf[0]==0 and cdf[-1]==1
