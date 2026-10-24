import numpy as np
from experiments.campaign_amplitude_adaptive_oracle_v01 import adaptive_integral
from experiments.campaign_surface_conditional_amplitude_v01 import ConditionalAmplitudeReference


def test_original_amplitude_adaptive_integral_matches_prior_and_cdf():
    sigma=20e-6
    oracle=lambda z:np.zeros((len(z),2))
    full=adaptive_integral(oracle,[0,0],sigma=sigma)
    expected=-2*np.log(sigma*np.sqrt(2*np.pi))
    assert abs(full['log_integral']-expected)<1e-10
    for kind,q,bounds in [('E',4000.,(2000.,8000.)),('K',12.,(5.,30.))]:
        partial=adaptive_integral(oracle,[0,0],sigma=sigma,CDF=(kind,q))
        expected_cdf=np.log(q/bounds[0])/np.log(bounds[1]/bounds[0])
        assert abs(np.exp(partial['log_integral']-full['log_integral'])-expected_cdf)<1e-10


def test_independent_direct_amplitude_outer_matches_transformed_gaussian():
    def ratio(r,labels):
        g=np.column_stack([np.ones(len(r)),200*r])
        return np.broadcast_to(g[:,None,:],(len(r),len(labels),2))
    y=[.00025,.00015];sigma=20e-6
    engine=ConditionalAmplitudeReference(ratio,[('observe',y)],['target'])
    result=engine.integrate(outer=128,inner=128,intrinsic=False)
    oracle=lambda z:ratio(np.exp(z),['observe'])[:,0]
    full=adaptive_integral(oracle,y,sigma=sigma)
    assert abs(full['log_integral']-result['log_evidence'])<1e-5
    for dim,kind,q,bounds in [(0,'E',4000.,(2000.,8000.)),(1,'K',12.,(5.,30.))]:
        part=adaptive_integral(oracle,y,sigma=sigma,CDF=(kind,q))
        expected=np.exp(part['log_integral']-full['log_integral'])
        unit=np.log(q/bounds[0])/np.log(bounds[1]/bounds[0])
        actual=engine.cdf(dim,unit,outer=128,inner=128,log_evidence=result['log_evidence'])
        assert abs(actual-expected)<1e-5


def test_nonfinite_or_empty_oracle_inputs_are_rejected():
    import pytest
    callback=lambda z:np.zeros((len(z),2))
    for y,sigma in [([],20e-6),([0,0],float('nan')),([0,0],float('inf'))]:
        with pytest.raises(ValueError):adaptive_integral(callback,y,sigma=sigma)
