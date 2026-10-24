"""Source-only synthetic Gaussian controls; no actual physics/cohort inputs."""
import numpy as np
import pytest
from scipy.integrate import quad
from experiments.campaign_surface_conditional_amplitude_v01 import (
    BOUNDS,LOG_WIDTH,RatioModal,ConditionalAmplitudeReference,AmplitudePlan,
    log_gaussian_interval,amplitude_nodes)
from experiments.campaign_surface_inverse_reference_v01 import ModalStaticReference,posterior_grid


def adaptive_amp(mu,SD,lo,hi):
    # Independent explicit Gaussian/a integration in physical a. No PPF/logmass.
    peak=min(max(mu,lo),hi);exponent=-.5*((peak-mu)/SD)**2
    points=[mu+k*SD for k in(-8,-3,-1,0,1,3,8)]
    if mu<lo:points += [lo+k*SD/max(1,abs((lo-mu)/SD))for k in(1,3,8,20)]
    if mu>hi:points += [hi-k*SD/max(1,abs((hi-mu)/SD))for k in(1,3,8,20)]
    edges=[lo]+sorted(set(x for x in points if lo<x<hi))+[hi]
    total=sum(quad(lambda a:np.exp(-.5*((a-mu)/SD)**2-exponent)/a,l,h,
        epsabs=1e-14,epsrel=2e-11,limit=200)[0]for l,h in zip(edges,edges[1:]))
    return np.log(total)+exponent


def test_stable_interval_mass_and_independent_amplitude_tail_integrals():
    for alpha,beta in [(-41.,-40.),(40.,41.),(-1e-8,2e-8),(-2.,3.),(10.,10.001)]:
        mode=min(max(0.,alpha),beta);peak=-.5*mode**2
        raw=quad(lambda x:np.exp(-.5*x*x-peak)/np.sqrt(2*np.pi),alpha,beta,epsabs=1e-14,epsrel=1e-11)[0]
        assert float(log_gaussian_interval(alpha,beta))==pytest.approx(np.log(raw)+peak,abs=2e-10)
    cases=[(.0003,.00001,.000125,.0005),(-.0002,.00001,.0002,.00021),(.001,.00002,.000125,.0002)]
    for mu,sd,lo,hi in cases:
        a,lw=amplitude_nodes([mu],[sd],[lo],[hi],128)
        actual=float(np.logaddexp.reduce(lw[0]));expected=adaptive_amp(mu,sd,lo,hi)
        assert abs(actual-expected)<2e-5
        assert np.all((a>=lo)&(a<=hi))


def test_ratio_all_modes_equivalent_physical_modal_including_label_order():
    modal=ModalStaticReference(np.diag([1.,2.,3.]),np.diag([.2,.7,1.]),np.arange(6.).reshape(3,2)+1,np.eye(3))
    ratio=RatioModal(modal,['A','B']);theta=np.array([[3000.,7.],[7000.,25.]])
    expected=modal(theta).transpose(0,2,1)[:,[1,0],:]
    actual=ratio(theta[:,1]/theta[:,0],['B','A'])/theta[:,0,None,None]
    assert np.allclose(actual,expected,rtol=2e-14,atol=1e-18)


def prior_ratio(r,labels):
    return np.stack([np.zeros((len(r),2))if label=='zero'else np.column_stack([np.ones(len(r)),1/r])for label in labels],axis=1)


def test_gzero_prior_branch_original_loguniform_law_and_positive_moments():
    engine=ConditionalAmplitudeReference(prior_ratio,[('zero',[0.,0.])],['target'])
    result=engine.integrate(outer=32,inner=32,keep=True)
    assert result['log_evidence']==pytest.approx(-2*np.log(20e-6*np.sqrt(2*np.pi)),abs=1e-11)
    assert np.max(abs(result['unit_mean']-.5))<1e-12
    assert np.max(abs(result['unit_covariance']-np.eye(2)/12))<1e-12
    assert np.max(abs(result['unit_CDF']-np.linspace(0,1,33)))<1e-12
    assert np.max(abs(result['unit_quantiles']-[.05,.5,.95]))<1e-8
    mean=(1/BOUNDS[:,0]-1/BOUNDS[:,1])/LOG_WIDTH
    second=(BOUNDS[:,0]**-2-BOUNDS[:,1]**-2)/(2*LOG_WIDTH)
    assert np.allclose(result['predictive_mean'],mean,rtol=1e-12,atol=1e-15)
    assert np.allclose(result['predictive_SD'],np.sqrt(second-mean*mean),rtol=1e-12,atol=1e-15)
    assert np.isclose(result['quadrature_probability'].sum(),1,atol=1e-12)
    assert result['weighted_hypotheses']==3*32*32


def test_both_refinement_axes_qualify_prior_and_preserve_levels():
    engine=ConditionalAmplitudeReference(prior_ratio,[('zero',[0.,0.])],['target'])
    answer=engine.qualify(plan=AmplitudePlan(inner=(8,16,32),outer=(8,16,32)))
    assert answer['qualified']
    assert len(answer['diagnostics']['levels'])==5
    assert all(answer['diagnostics']['successive'][axis][-1]['passed']for axis in('inner','outer'))


def informative_ratio(r,labels):
    g=np.column_stack([np.ones(len(r)),200*r])
    return np.stack([g if label=='observe'else np.column_stack([np.ones(len(r)),.1/r])for label in labels],axis=1)


def test_nonzero_Gaussian_matches_independent_original_coordinate_tensor():
    y=np.array([.00025,.00015]);engine=ConditionalAmplitudeReference(informative_ratio,[('observe',y)],['target'])
    result=engine.integrate(outer=80,inner=96,intrinsic=False)
    def decode(u):
        physical=np.exp(np.log(BOUNDS[:,0])+u*LOG_WIDTH);return 1/physical[:,0],physical[:,1]/physical[:,0]
    def ll(u):
        a,r=decode(u);pred=a[:,None]*informative_ratio(r,['observe'])[:,0]
        return -.5*np.sum(((pred-y)/20e-6)**2,axis=1)-2*np.log(20e-6*np.sqrt(2*np.pi))
    def predict(u):
        a,r=decode(u);return a[:,None]*informative_ratio(r,['target'])[:,0]
    tensor=posterior_grid(ll,n=257,predict=predict)
    assert abs(result['log_evidence']-tensor['log_evidence'])<2e-5
    assert np.max(abs(result['unit_mean']-tensor['unit_mean']))<1e-5
    assert np.max(abs(result['predictive_mean']-tensor['predictive_mean']))<1e-7
    assert np.max(abs(result['predictive_SD']-tensor['predictive_SD']))<1e-7
    for dim in(0,1):
        for j,q in enumerate((.05,.5,.95)):
            cdf=engine.cdf(dim,float(tensor['unit_quantiles'][dim,j]),outer=80,inner=96,log_evidence=result['log_evidence'])
            assert abs(cdf-q)<.0003


def test_underresolved_outer_axis_fails_closed_and_keeps_negative_levels():
    def sharp(r,labels):return np.broadcast_to(np.column_stack([100*np.ones(len(r)),20000*r])[:,None,:],(len(r),len(labels),2))
    engine=ConditionalAmplitudeReference(sharp,[('observe',[.025,.015])],['target'])
    answer=engine.qualify(plan=AmplitudePlan(inner=(8,16),outer=(4,8)))
    assert not answer['qualified']
    assert len(answer['diagnostics']['levels'])==3
    assert not answer['diagnostics']['successive']['outer'][-1]['passed']


def test_informative_Gaussian_full_inner_outer_qualification():
    engine=ConditionalAmplitudeReference(informative_ratio,[('observe',[.00025,.00015])],['target'])
    answer=engine.qualify(plan=AmplitudePlan(inner=(16,32,64),outer=(32,64,128)))
    assert answer['qualified']
    assert answer['local_final']['weighted_hypotheses']==3*128*64
    assert len(answer['diagnostics']['levels'])==5
    assert all(answer['diagnostics']['successive'][axis][-1]['passed']for axis in('inner','outer'))
