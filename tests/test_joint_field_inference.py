import unittest
import numpy as np
from numpy.testing import assert_allclose
from scripts.joint_field_identification.inference import (
    Posterior, fit_posterior, risks_after_values, expected_risk_mc,
    expected_risk_gh, marginal_quantiles)


class TestJointFieldInference(unittest.TestCase):
    def setUp(self):
        self.H = np.array([[[1.,.3],[.2,1.]],[[.4,-.2],[.8,1.3]]])
        self.K = np.array([[1.,.2],[.2,.7]])
        self.R = np.array([[.1,.02],[.02,.15]])
        self.y = np.array([.7,-.4])
        self.p = fit_posterior(self.H,self.y,self.K,self.R,[.3,.7])
        self.T = np.array([[[1.,.2],[.3,1.]],[[1.2,.4],[-.5,.7]]])

    def test_explicit_joint_gaussian_identity_and_determinants(self):
        masses=[]
        for j in range(2):
            H=self.H[j];precision=np.linalg.inv(self.K)+H.T@np.linalg.solve(self.R,H)
            cov=np.linalg.inv(precision)
            mean=cov@H.T@np.linalg.solve(self.R,self.y)
            assert_allclose(self.p.means[j],mean,atol=1e-12)
            assert_allclose(self.p.covariances[j],cov,atol=1e-12)
            C=self.R+H@self.K@H.T
            density=np.exp(-.5*self.y@np.linalg.solve(C,self.y))/(2*np.pi*np.sqrt(np.linalg.det(C)))
            masses.append([.3,.7][j]*density)
        assert_allclose(self.p.weights,np.array(masses)/sum(masses),atol=1e-13)
        self.assertAlmostEqual(self.p.log_evidence,np.log(sum(masses)),places=12)

    def test_total_covariance_includes_between_hypothesis_term(self):
        t=self.p.target(self.T)
        mu=self.p.weights@t.component_means
        variance=sum(w*(v+(m-mu)**2) for w,m,v in zip(self.p.weights,t.component_means,t.component_variances))
        assert_allclose(t.variance,variance)
        self.assertGreater(t.variance.sum(),(self.p.weights@t.component_variances).sum())

    def test_known_hypothesis_analytic_risk(self):
        p=Posterior(np.ones(1),self.p.means[:1],self.p.covariances[:1])
        h=np.array([[.4,.7]]);noise=.13;T=self.T[:1]
        cross=T[0]@p.covariances[0]@h[0]
        expected=p.target(T).risk-np.mean(cross**2)/(h[0]@p.covariances[0]@h[0]+noise)
        self.assertAlmostEqual(expected_risk_gh(p,T,h,noise),expected,places=12)
        self.assertAlmostEqual(expected_risk_mc(p,T,h,noise,n_samples=128),expected,places=12)

    def test_probe_preserves_force_conditionals(self):
        p=self.p.update_scalar(np.zeros_like(self.p.means),.8,.05,offset=[0.,1.])
        assert_allclose(p.means,self.p.means,atol=0)
        assert_allclose(p.covariances,self.p.covariances,atol=0)
        self.assertGreater(p.weights[1],self.p.weights[1])

    def test_order_independence_and_batch_equivalence(self):
        h1=np.array([[.2,1.],[.4,.5]]);h2=np.array([[1.,-.3],[.7,.9]])
        p12=self.p.update_scalar(h1,.3,.2).update_scalar(h2,-.2,.1)
        p21=self.p.update_scalar(h2,-.2,.1).update_scalar(h1,.3,.2)
        assert_allclose(p12.weights,p21.weights,atol=1e-13)
        assert_allclose(p12.means,p21.means,atol=1e-13)
        assert_allclose(p12.covariances,p21.covariances,atol=1e-13)
        self.assertAlmostEqual(p12.log_evidence,p21.log_evidence,places=12)
        H=np.concatenate([self.H,h1[:,None],h2[:,None]],axis=1)
        R=np.zeros((4,4));R[:2,:2]=self.R;R[2,2]=.2;R[3,3]=.1
        batch=fit_posterior(H,np.r_[self.y,.3,-.2],self.K,R,[.3,.7])
        assert_allclose(batch.means,p12.means,atol=1e-12)
        assert_allclose(batch.weights,p12.weights,atol=1e-12)

    def test_vectorized_risks_equal_explicit_updates(self):
        h=np.array([[.2,.8],[.7,-.4]]);offset=np.array([.3,-.1]);values=np.linspace(-2,2,17)
        calculated=risks_after_values(self.p,self.T,h,.15,values,offset)
        reference=[self.p.update_scalar(h,y,.15,offset).target(self.T).risk for y in values]
        assert_allclose(calculated,reference,atol=1e-12)

    def test_mc_matches_quadrature_and_reproducible(self):
        h=np.zeros_like(self.p.means);offset=np.array([-.3,.5])
        q=expected_risk_gh(self.p,self.T,h,.2,offset,order=81)
        a=expected_risk_mc(self.p,self.T,h,.2,offset,n_samples=10000,seed=9,return_details=True)
        b=expected_risk_mc(self.p,self.T,h,.2,offset,n_samples=10000,seed=9,return_details=True)
        self.assertEqual(a,b)
        self.assertLess(abs(a['expected_risk']-q),5*a['standard_error']+1e-6)
        self.assertLessEqual(q,self.p.target(self.T).risk+1e-10)

    def test_sample_moments_and_quantiles(self):
        indices,forces=self.p.sample(60000,seed=31)
        q=np.einsum('sdf,sf->sd',self.T[indices],forces)
        moment=self.p.target(self.T)
        assert_allclose(q.mean(0),moment.mean,atol=.012)
        assert_allclose(q.var(0),moment.variance,atol=.012)
        intervals=marginal_quantiles(self.p,self.T,[.05,.95])
        covered=((q>=intervals[0])&(q<=intervals[1])).mean(0)
        assert_allclose(covered,.9,atol=.008)

    def test_bad_noise_and_zero_prior(self):
        with self.assertRaises((ValueError,np.linalg.LinAlgError)):
            fit_posterior(self.H,self.y,self.K,-.1)
        p=fit_posterior(self.H,self.y,self.K,self.R,[1.,0.])
        assert_allclose(p.weights,[1.,0.])
        updated=p.update_scalar(np.ones_like(p.means),0.,.1)
        assert_allclose(updated.weights,[1.,0.])

    def test_high_signal_noise_covariance_remains_positive(self):
        p=fit_posterior(self.H,self.y,self.K,1e-14)
        self.assertGreater(np.linalg.eigvalsh(p.covariances).min(),0)
        p.sample(10,seed=1)

    def test_cached_target_moments_identical(self):
        h=np.array([[.2,.8],[.7,-.4]]);offset=np.array([.3,-.1])
        target_offset=np.array([[3.,-1.],[.1,-.2]])
        cached=self.p.target(self.T,target_offset)
        a=expected_risk_mc(self.p,self.T,h,.15,offset,n_samples=128,seed=38,target_offset=target_offset,return_details=True)
        b=expected_risk_mc(self.p,self.T,h,.15,offset,n_samples=128,seed=38,target_offset=target_offset,return_details=True,target_moments=cached)
        self.assertEqual(a,b)


if __name__=='__main__':
    unittest.main()
