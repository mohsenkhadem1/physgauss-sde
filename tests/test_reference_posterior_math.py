import numpy as np
from scipy.stats import truncnorm
from scripts.compositional_neural_mechanics.reference_posterior_math import integrate,compare,scale_bounds,LOGK_MEAN


def test_exact_conditional_common_scale_gaussian_and_intervals():
    nodes=np.array([[LOGK_MEAN,LOGK_MEAN]])
    a=integrate(nodes,[0.],[0.],[[0,0,0]],0.,'V',1025)
    b=integrate(nodes,[0.],[0.],[[0,0,0]],0.,'V',2049)
    law=truncnorm(-np.sqrt(3),np.sqrt(3),scale=1/np.sqrt(3))
    np.testing.assert_allclose(b['mean'],[0,LOGK_MEAN,LOGK_MEAN],atol=1e-12)
    np.testing.assert_allclose(b['covariance'],np.full((3,3),law.var()),atol=1e-12)
    np.testing.assert_allclose(np.array(b['interval90'])-np.array([0,LOGK_MEAN,LOGK_MEAN])[:,None],np.broadcast_to(law.ppf([.05,.95]),(3,2)),atol=5e-7)
    assert compare(a,b)['passed'] and abs(b['conditional_cdf_total']-1)<1e-12


def test_wrench_likelihood_resolves_scale_without_changing_visual_gauge():
    nodes=np.array([[LOGK_MEAN,LOGK_MEAN]]);target=np.exp(.2)
    v=integrate(nodes,[0.],[0.],[[100,100*target,100*target**2]],0.,'V',2049)
    vf=integrate(nodes,[0.],[0.],[[100,100*target,100*target**2]],0.,'VF',2049)
    assert abs(v['mean'][0])<1e-12 and .15<vf['mean'][0]<.22 and vf['sd'][0]<v['sd'][0]/4
    assert scale_bounds([np.log(20)-.5,LOGK_MEAN])[0]==.5
