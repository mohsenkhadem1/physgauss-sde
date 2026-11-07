import unittest
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from scipy.stats import norm
from scipy.optimize import brentq
from scripts.gap_first_v01.quantiles import gaussian_mixture_quantiles as quantiles,gaussian_mixture_cdf as cdf,marginal_quantiles
from scripts.joint_field_identification.inference import Posterior,marginal_quantiles as historical_quantiles

class MixtureQuantileTests(unittest.TestCase):
    def test_historical_endpoint_regression(self):
        p=Posterior(np.array([.5,.5]),np.zeros((2,1)),np.ones((2,1,1)),0.)
        T=np.zeros((2,1,1));offset=np.array([[.001],[.009]])
        old=historical_quantiles(p,T,offset=offset)
        self.assertLess(old[1,0],offset[1,0])
        new=marginal_quantiles(p,T,offset=offset)
        assert_array_equal(new[:,0],offset[:,0])
        self.assertTrue(new[0,0]<=offset[0,0]<=new[1,0])
        self.assertTrue(new[0,0]<=offset[1,0]<=new[1,0])

    def test_atoms_duplicate_zero_weight_and_plateau(self):
        w=[0,.2,.3,.5];mu=[-99,-.3,-.3,.8]
        got=quantiles(w,mu,[0]*4,[.05,.5,np.nextafter(.5,1),.95])[:,0]
        assert_array_equal(got,[-.3,-.3,.8,.8])
        assert_array_equal(quantiles([2],[.123],[0],[.01,.99]),[[.123],[.123]])

    def test_generalized_inverse_jump(self):
        w=[.5,.5];m=[0.,0.];v=[0.,1.]
        assert_array_equal(quantiles(w,m,v,[.25,.3,.5,.7,.75])[:,0],[0.,0.,0.,0.,0.])
        assert_allclose(quantiles(w,m,v,[.1,.9])[:,0],[norm.ppf(.2),norm.ppf(.8)],atol=1e-14)
        assert_allclose(cdf([0],w,m,v,left_limit=True),[.25])
        assert_allclose(cdf([0],w,m,v),[.75])

    def test_continuous_mixture_independent_root(self):
        w=np.array([.2,.3,.5]);m=np.array([-2.,.5,3.]);v=np.array([.3,2.,.7])
        ps=[.05,.25,.95];want=[brentq(lambda x:sum(w*norm.cdf(x,m,np.sqrt(v)))-p,-20,20,xtol=1e-13) for p in ps]
        assert_allclose(quantiles(w,m,v,ps)[:,0],want,atol=1e-12)

    def test_normal_extreme_tails_not_twelve_sigma_clipped(self):
        ps=[1e-80,1e-20,.5,1-1e-12]
        assert_allclose(quantiles([1],[2.],[9.],ps)[:,0],norm.ppf(ps,2.,3.),atol=3e-5)

    def test_coordinate_order_and_scale_equivariance(self):
        mu=np.array([[-.3,.009],[.8,.001]]);v=np.zeros_like(mu)
        q=quantiles([.5,.5],mu,v)
        assert_array_equal(q,[[-.3,.001],[.8,.009]])
        assert_allclose(quantiles([5,5],mu*100+7,v*10000),q*100+7)

    def test_validation(self):
        for w,m,v,p in [([0],[0],[0],[.5]),([-1],[0],[0],[.5]),([1],[0],[-1],[.5]),([1],[0],[0],[np.nan]),([1],[0],[0],[0])]:
            with self.assertRaises(ValueError):quantiles(w,m,v,p)

if __name__=='__main__':unittest.main()
