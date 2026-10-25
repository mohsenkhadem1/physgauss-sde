import numpy as np
from scipy.integrate import quad
from experiments.campaign_nonlinear_tilted_pwl_v01 import NonlinearTiltedPWL
from experiments.campaign_positive_amplitude_pwl_v01 import PositiveAmplitudePWL
from experiments.campaign_surface_conditional_amplitude_v01 import BOUNDS,LOG_WIDTH


def amplitude(unit):return np.exp(-np.log(BOUNDS[0,0])-unit[:,0]*LOG_WIDTH[0])


def test_zero_correction_recovers_frozen_positive_measure():
    mu=np.array([.00024]);sd=np.array([.00003]);lo=np.array([.000125]);hi=np.array([.0004]);peak=np.array([-3.])
    def ll(u):return -3-.5*((amplitude(u)-mu[0])/sd[0])**2
    base=PositiveAmplitudePWL(mu,sd,lo,hi,panels=128)
    new=NonlinearTiltedPWL(mu,sd,lo,hi,log_ratio=np.log([.002]),linear_peak=peak,full_loglik=ll,panels=128)
    np.testing.assert_allclose(new.log_weights,base.log_weights,atol=2e-12,rtol=0)
    for q in np.linspace(lo[0],hi[0],99):np.testing.assert_allclose(new.cdf([q]),base.cdf([q]),atol=2e-14)


def test_nonlinear_tilt_matches_original_a_integral_and_cdf():
    mu=.00024;sd=.00003;lo=.000125;hi=.0004
    def logf(a):return -.5*((a-mu)/sd)**2-1.7*((a-mu)/sd)**4/20
    def ll(u):return logf(amplitude(u))
    primitive=NonlinearTiltedPWL([mu],[sd],[lo],[hi],log_ratio=np.log([.002]),linear_peak=[0.],full_loglik=ll,panels=512)
    mass=quad(lambda a:np.exp(logf(a))/a,lo,hi,epsabs=1e-12,epsrel=1e-11)[0]
    np.testing.assert_allclose(np.exp(primitive.log_kernel_mass)*primitive.mass0,[mass],rtol=2e-5)
    previous=0.
    for cut in np.linspace(lo,hi,101):
        expected=quad(lambda a:np.exp(logf(a))/a,lo,cut,epsabs=1e-12,epsrel=1e-11)[0]/mass
        actual=primitive.cdf([cut])[0]
        assert actual>=previous-1e-14;previous=actual
        assert abs(actual-expected)<2e-5
        np.testing.assert_allclose(primitive.cdf([cut])+primitive.tail([cut]),1.,atol=2e-14)


def test_zero_linear_signal_is_proposal_not_logprior_posterior():
    lo=.000125;hi=.0004
    def ll(u):return 3*np.log(amplitude(u)/lo)
    p=NonlinearTiltedPWL([0.],[np.inf],[lo],[hi],log_ratio=np.log([.002]),linear_peak=[0.],full_loglik=ll,panels=512)
    cut=.00023;exact=(cut**3-lo**3)/(hi**3-lo**3)
    assert abs(p.cdf([cut])[0]-exact)<2e-6
    assert abs(p.cdf([cut])[0]-np.log(cut/lo)/np.log(hi/lo))>.1
    coarse=NonlinearTiltedPWL([0.],[np.inf],[lo],[hi],log_ratio=np.log([.002]),linear_peak=[0.],full_loglik=ll,panels=256)
    exactmass=((hi/lo)**3-1)/3
    fineerror=abs(np.exp(p.log_weights).sum()/exactmass-1)
    coarseerror=abs(np.exp(coarse.log_weights).sum()/exactmass-1)
    assert fineerror<1e-5 and fineerror<coarseerror/3.9


def test_nonzero_peak_is_added_exactly_once_to_mass():
    peak=-7.;lo=.000125;hi=.0004
    for sd in(.00003,np.inf):
        def ll(u):return np.full(len(u),-5.) if np.isinf(sd) else peak-.5*((amplitude(u)-.00024)/sd)**2
        p=NonlinearTiltedPWL([.00024],[sd],[lo],[hi],log_ratio=np.log([.002]),linear_peak=[peak],full_loglik=ll,panels=512)
        mass=np.exp(peak+np.logaddexp.reduce(p.log_weights[0]))
        exact=quad(lambda a:np.exp(-5. if np.isinf(sd) else peak-.5*((a-.00024)/sd)**2)/a,lo,hi,epsabs=1e-14)[0]
        np.testing.assert_allclose(mass,exact,rtol=1e-5)
