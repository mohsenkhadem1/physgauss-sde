import unittest
import numpy as np
from experiments.gaussian_linear_filter import gaussian_linear_update


class LinearFilterTests(unittest.TestCase):
    def test_matches_direct_gaussian_conditioning(self):
        rng=np.random.default_rng(8);h=rng.normal(size=(8,3));b=rng.normal(size=8);y=rng.normal(size=8)
        c=np.diag([.2,.3,.4]);r=np.eye(8)*.1;m=np.array([1.,2.,3.])
        mean,covariance,evidence=gaussian_linear_update(y,b,h,m,c,r)
        s=r+h@c@h.T;gain=np.linalg.solve(s,h@c).T
        np.testing.assert_allclose(mean,m+gain@(y-b-h@m),atol=1e-12)
        np.testing.assert_allclose(covariance,c-gain@h@c,atol=1e-12)
        expected=-.5*(8*np.log(2*np.pi)+np.linalg.slogdet(s)[1]+(y-b-h@m)@np.linalg.solve(s,y-b-h@m))
        self.assertAlmostEqual(evidence,expected,places=10)

    def test_uninformative_measurement_preserves_prior(self):
        mean,covariance,_=gaussian_linear_update(np.ones(2),np.zeros(2),np.zeros((2,1)),np.array([3.]),np.array([[4.]]),np.eye(2))
        np.testing.assert_allclose(mean,[3]);np.testing.assert_allclose(covariance,[[4]])


if __name__=='__main__':unittest.main()
