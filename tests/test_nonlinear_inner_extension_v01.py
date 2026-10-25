"""Source-only numerical controls; no campaign observation or draw access."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from experiments.campaign_positive_amplitude_pwl_v01 import PositiveAmplitudePWL
from experiments.campaign_positive_amplitude_pwl_v02 import PositiveAmplitudePWLV02
from experiments.campaign_nonlinear_mis_pwl_v01 import NonlinearMISPWL
from experiments.campaign_nonlinear_mis_pwl_v02 import NonlinearMISPWLV02
from experiments.campaign_nonlinear_conditional_reference_v03 import NonlinearConditionalReferenceV03
from experiments.campaign_nonlinear_conditional_reference_v04 import NonlinearConditionalReferenceV04,ExtendedAmplitudePWLPlan


def parameters():
    return dict(mu=np.array([.45,.45,.45]),SD=np.array([.08,np.inf,.0001]),lower=np.full(3,.1),upper=np.ones(3))


def mis(cls,n,shift=0.):
    return cls(**parameters(),panels=n,log_ratio=np.zeros(3),linear_peak=np.array([7.,-3.,11.]),
               full_loglik=lambda u:None,full_loglik_az=lambda a,z:2+shift-.5*((a-.82)/.055)**2,chunk=71)


@pytest.mark.parametrize('panels',[128,256,512])
def test_all_old_orders_bit_identical_normal_zero_and_separated_proposals(panels):
    old=PositiveAmplitudePWL(**parameters(),panels=panels)
    new=PositiveAmplitudePWLV02(**parameters(),panels=panels)
    for name in vars(old):np.testing.assert_array_equal(getattr(old,name),getattr(new,name),err_msg=name)
    old=mis(NonlinearMISPWL,panels);new=mis(NonlinearMISPWLV02,panels)
    for name in vars(old):np.testing.assert_array_equal(getattr(old,name),getattr(new,name),err_msg=name)
    for q in(.1,.3,.82,.999,1.):
        np.testing.assert_array_equal(old.cdf(np.full(3,q)),new.cdf(np.full(3,q)))
        np.testing.assert_array_equal(old.tail(np.full(3,q)),new.tail(np.full(3,q)))


def test_1024_shift_invariance_and_zero_stream_keeps_original_measure():
    a=mis(NonlinearMISPWLV02,1024);b=mis(NonlinearMISPWLV02,1024,shift=13.)
    np.testing.assert_allclose(logsumexp(b.log_weights,axis=1)-logsumexp(a.log_weights,axis=1),13.,atol=4e-15,rtol=0)
    np.testing.assert_allclose(a.cdf(np.full(3,.8)),b.cdf(np.full(3,.8)),atol=3e-15,rtol=0)
    p=NonlinearMISPWLV02(**{k:v[2:]for k,v in parameters().items()},panels=1024,log_ratio=np.zeros(1),linear_peak=np.zeros(1),full_loglik=lambda u:None,full_loglik_az=lambda a,z:-.5*((a-.82)/.001)**2)
    assert p.panel_mass[0,0].sum()==0 and p.panel_mass[0,1].sum()>0
    c=np.array([p.cdf(np.array([q]))[0]for q in np.linspace(.1,1,101)])
    assert np.all(np.diff(c)>=0) and c[0]==0 and c[-1]==1


def test_512_to_1024_integral_and_continuous_cdf_against_independent_a_quad():
    ll=lambda a:2-.5*((a-.82)/.055)**2
    exact=quad(lambda a:np.exp(ll(a))/a,.1,1,epsabs=1e-12,epsrel=1e-12)[0]
    exact_cdf=quad(lambda a:np.exp(ll(a))/a,.1,.8,epsabs=1e-12,epsrel=1e-12)[0]/exact
    errors=[]
    for n in(512,1024):
        p=mis(NonlinearMISPWLV02,n)
        mass=np.exp(logsumexp(p.log_weights[0])+7)
        errors.append((abs(mass/exact-1),abs(p.cdf(np.full(3,.8))[0]-exact_cdf)))
    assert np.all(np.asarray(errors[1])<np.asarray(errors[0]))
    assert errors[1][0]<1e-5 and errors[1][1]<1e-4


def engine(cls):
    def ratio(z,labels):return np.zeros((len(z),len(labels),1))
    def response(u,labels):return np.stack([np.zeros((len(u),1))if label=='observe'else (4e-6*u[:,0]**2)[:,None]for label in labels],axis=1)
    return cls(ratio,response,[('observe',np.zeros(1))],['target'],retain_hypotheses=True)


@pytest.mark.parametrize('panels',[128,256,512])
def test_old_reference_evidence_moments_cdf_and_hypotheses_bit_identical(panels):
    old=engine(NonlinearConditionalReferenceV03).integrate(outer=8,inner=panels,keep=True)
    new=engine(NonlinearConditionalReferenceV04).integrate(outer=8,inner=panels,keep=True)
    assert old.keys()==new.keys()
    for name in old:np.testing.assert_array_equal(old[name],new[name],err_msg=name)


def test_extended_plan_refuses_undeclared_orders_and_uses_unchanged_qualification(monkeypatch):
    import experiments.campaign_nonlinear_adaptive_oracle_v01 as oracle
    for order in(64,2048,True):
        with pytest.raises(ValueError):ExtendedAmplitudePWLPlan(inner=(128,order))
    assert ExtendedAmplitudePWLPlan().inner==(128,256,512,1024)
    with pytest.raises(ValueError):ExtendedAmplitudePWLPlan(outer=(16,8))
    # A false independent normalizer must still reject otherwise smooth refinement.
    monkeypatch.setattr(oracle,'nonlinear_adaptive_integral',lambda *args,**kwargs:dict(log_integral=0.,estimated_outer_relative_error=0.,maximum_estimated_inner_relative_error=0.))
    r=engine(NonlinearConditionalReferenceV04).qualify(plan=ExtendedAmplitudePWLPlan(outer=(8,16)))
    d=r['diagnostics']
    assert not r['qualified'] and not d['independent_adaptive']['passed']
    assert [(x['coarse'],x['fine'])for x in d['successive']['inner']]==[(128,256),(256,512),(512,1024)]
    assert [(x['outer'],x['inner'])for x in d['levels']]==[(16,128),(16,256),(16,512),(16,1024),(8,1024)]
    assert d['gates']==dict(logZ=.01,CDF=.005,quantile=.002,unit_mean=.001,target_absolute_m=2e-6,target_relative_SD=.01)


def test_1024_zero_signal_constant_likelihood_exact_log_uniform_measure():
    p=NonlinearMISPWLV02(mu=np.array([.45]),SD=np.array([np.inf]),lower=np.array([.1]),upper=np.array([1.]),log_ratio=np.zeros(1),linear_peak=np.array([7.]),full_loglik=lambda u:None,full_loglik_az=lambda a,z:np.full_like(a,3.),panels=1024)
    np.testing.assert_allclose(logsumexp(p.log_weights[0])+7,3+np.log(np.log(10)),atol=2e-15,rtol=0)
    for q in(.1,.3,.8,1.):
        np.testing.assert_allclose(p.cdf(np.array([q])),np.log(q/.1)/np.log(10),atol=3e-15,rtol=0)
    assert p.amplitude.shape==(1,4096)
