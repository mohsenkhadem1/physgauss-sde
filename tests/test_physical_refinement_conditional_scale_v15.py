import numpy as np
import pytest
from scipy.integrate import quad
from scripts.physical_refinement_v15.conditional_scale import integrate,reference_from_coefficients,LOWER,UPPER


@pytest.mark.parametrize('w,y,sd',[(0.,0.,1.),(.01,.01,1.),(1.,1.1,.003),(1.,.75,.003),(1.,.70,.003),(1.,1.6,.003)])
def test_conditional_quadrature_against_original_residual_integration(w,y,sd):
    A=(w/sd)**2;B=w*y/sd**2
    reference=float(reference_from_coefficients(A,B))
    logref=-.5*((y-reference*w)/sd)**2-np.log(sd*np.sqrt(2*np.pi))
    result=integrate(A,B,logref,reference,97)
    # Independent quadrature in log-scale, with mode/width breakpoints, and
    # the original residual likelihood rather than completed-square algebra.
    mode=np.clip(y/w if w else 1.,LOWER,UPPER)
    width=sd/abs(w) if w else 1.
    peak=-.5*((y-mode*w)/sd)**2-np.log(sd*np.sqrt(2*np.pi))
    breaks=sorted(set(np.log(np.clip(mode+width*np.array([-8,-4,-1,0,1,4,8]),LOWER,UPPER))))
    def value(t,power):
        scale=np.exp(t)
        loglike=-.5*((y-scale*w)/sd)**2-np.log(sd*np.sqrt(2*np.pi))
        return np.exp(loglike-peak)*scale**power/np.log(UPPER/LOWER)
    integrals=[quad(lambda t:value(t,k),np.log(LOWER),np.log(UPPER),points=breaks,epsabs=1e-12,epsrel=1e-10,limit=300)[0] for k in (0,1,2)]
    assert np.isclose(result['log_evidence'],peak+np.log(integrals[0]),atol=1e-5)
    assert np.isclose(result['mean'],integrals[1]/integrals[0],atol=1e-5)
    assert np.isclose(result['second_moment'],integrals[2]/integrals[0],atol=2e-5)
