import numpy as np
import pytest
from scipy.special import ndtr
from experiments.campaign_centered_cubature import integrate_centered,normalize_moments


def test_known_truncated_gaussian_variance_and_large_shift_invariance():
    mu=np.array([.42,.57]);sd=np.array([.08,.13]);slope=np.array([.002,.001]);offset=np.array([.00885,.01777])
    density=lambda x:np.exp(-.5*np.sum(((x-mu)/sd)**2,1))
    results=[]
    for shift in (0.,.25):
        predict=lambda x:offset+shift+slope*x
        results.append(integrate_centered(density,predict,offset+shift+slope*mu,.0002,rtol=1e-8,atol=1e-15))
    a=-mu/sd;b=(1-mu)/sd;Z=ndtr(b)-ndtr(a);phi=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)
    expected=sd**2*(1+(a*phi(a)-b*phi(b))/Z-((phi(a)-phi(b))/Z)**2)*slope**2
    np.testing.assert_allclose(results[0]['held_std_m'],np.sqrt(expected),rtol=2e-9,atol=1e-13)
    np.testing.assert_allclose(results[1]['held_std_m'],results[0]['held_std_m'],atol=3e-11,rtol=0)
    np.testing.assert_allclose(results[1]['held_mean_m']-.25,results[0]['held_mean_m'],atol=3e-11,rtol=0)
    assert np.max(results[0]['estimated_std_error_m'])<1e-8
    assert results[0]['error_estimates_are_NOT_rigorous_bounds']


def test_unresolved_mass_or_negative_variance_fails_without_repair():
    with pytest.raises(ValueError,match='mass'):normalize_moments([1.,0.,1.],[1.,0.,0.],[0.],1.)
    with pytest.raises(ValueError,match='negative'):normalize_moments([1.,2.,1.],[1e-10,1e-10,1e-10],[0.],1.)


def test_nonzero_centered_mean_is_corrected_before_variance():
    answer=normalize_moments([2.,4.,10.],[1e-10]*3,[1000.],.0002)
    assert answer['held_mean_m'][0]==pytest.approx(1000.0004)
    assert answer['held_std_m'][0]==pytest.approx(.0002)
