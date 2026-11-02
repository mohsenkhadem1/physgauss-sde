import numpy as np
import pytest
from scripts.compositional_neural_mechanics.reference_posterior_math import integrate as failed_integrate,compare,LOGK_MEAN
from scripts.compositional_neural_mechanics.reference_posterior_recovery_math import integrate,ConditionalCDF


def test_narrow_likelihood_reproduces_failure_and_recovers_positive_cdf():
    r=np.array([[LOGK_MEAN,LOGK_MEAN]]);a=1e5;b=a*np.exp(.2);abc=[[a,b,b*b/a]]
    with pytest.raises(ValueError,match='nonmonotone'):failed_integrate(r,[0],[0],abc,0,'VF',1025)
    low=integrate(r,[0],[0],abc,0,'VF',1025);fine=integrate(r,[0],[0],abc,0,'VF',2049)
    old_fine=failed_integrate(r,[0],[0],abc,0,'VF',4097)
    assert compare(low,fine)['passed'] and compare(fine,old_fine)['passed'] and fine['adaptive_normalizer_gate']
    cdf=ConditionalCDF(r[0],abc[0],'VF');grid=np.linspace(.18,.22,201);values=np.array([cdf(x)for x in grid])
    assert (np.diff(values)>=-1e-12).all() and (values>=0).all() and (values<=1).all()


def test_visual_conditional_closed_normal_matches_original():
    r=np.array([[LOGK_MEAN,LOGK_MEAN]])
    a=integrate(r,[0],[0],[[0,0,0]],0,'V',2049);b=failed_integrate(r,[0],[0],[[0,0,0]],0,'V',2049)
    for k in ('mean','covariance'):np.testing.assert_allclose(a[k],b[k],atol=1e-12)
    np.testing.assert_allclose(a['interval90'],b['interval90'],atol=5e-7)
    assert a['adaptive_normalizer_gate']
