"""Independent finite-dimensional references for continuous marginalized SMC."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.stats import norm
from scripts.joint_field_identification.continuous import (
    SMCConfig,SMCResult,run_smc,reflect_unit,systematic_resample,
    conditional_ess,MarginalGaussianLikelihood)
from scripts.joint_field_identification.inference import fit_posterior


class TestContinuousSMC(unittest.TestCase):
    def test_constant_likelihood_normalization_and_support(self):
        bounds=np.array([[-2,3],[5,7],[-.8,.8]])
        r=run_smc(lambda theta:np.full(len(theta),-2.3),bounds=bounds,
                  config=SMCConfig(particles=4096),seed=16)
        self.assertTrue(r.complete);self.assertAlmostEqual(r.log_evidence,-2.3,places=12)
        self.assertTrue(np.all(r.theta>=bounds[:,0]) and np.all(r.theta<=bounds[:,1]))
        mean,cov=r.moments()
        assert_allclose(mean,bounds.mean(1),atol=.045)
        assert_allclose(cov,np.diag(np.diff(bounds,axis=1).ravel()**2/12),atol=.06)
        self.assertEqual(r.evaluation_count,4096*(1+4))

    def test_truncated_gaussian_against_quadrature_and_evidence(self):
        bounds=np.array([[-2.,3.]])
        def loglike(theta):return norm.logpdf(theta[:,0],loc=.7,scale=.35)
        z=quad(lambda x:np.exp(loglike(np.array([[x]]))[0])/5,-2,3,epsabs=1e-12)[0]
        mean=quad(lambda x:x*np.exp(loglike(np.array([[x]]))[0])/5,-2,3,epsabs=1e-12)[0]/z
        variance=quad(lambda x:(x-mean)**2*np.exp(loglike(np.array([[x]]))[0])/5,-2,3,epsabs=1e-12)[0]/z
        r=run_smc(loglike,bounds=bounds,config=SMCConfig(particles=8192,moves=6),seed=75)
        m,c=r.moments()
        self.assertAlmostEqual(m[0],mean,delta=.018)
        self.assertAlmostEqual(c[0,0],variance,delta=.012)
        self.assertAlmostEqual(r.log_evidence,np.log(z),delta=.045)
        for stage in r.stages:
            if stage['temperature']<1:self.assertAlmostEqual(stage['conditional_ess'],8192*.8,delta=1e-6)
        self.assertGreater(len(r.stages),2)

    def test_checkpoint_resume_bitwise_repeatability(self):
        bounds=np.array([[-2,3],[-1,2]])
        def loglike(t):return -.5*np.sum(((t-[.4,.6])/[.2,.3])**2,axis=1)
        config=SMCConfig(particles=384,moves=4,batch_size=100)
        uninterrupted=run_smc(loglike,bounds=bounds,config=config,seed=901,target_id='synthetic-v1')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.npz'
            partial=run_smc(loglike,bounds=bounds,config=config,seed=901,target_id='synthetic-v1',
                            checkpoint=path,stop_after_stages=1)
            self.assertFalse(partial.complete)
            with self.assertRaises(ValueError):partial.force_posterior(lambda theta:None)
            resumed=run_smc(loglike,bounds=bounds,config=config,seed=901,target_id='synthetic-v1',checkpoint=path,resume=True)
            loaded=SMCResult.load(path)
            for result in (resumed,loaded):
                assert_allclose(result.theta,uninterrupted.theta,atol=0,rtol=0)
                self.assertEqual(result.log_evidence,uninterrupted.log_evidence)
                self.assertEqual(result.stages,uninterrupted.stages)
                self.assertEqual(result.rng_state,uninterrupted.rng_state)
            with self.assertRaises(ValueError):
                run_smc(loglike,bounds=bounds,config=config,seed=901,target_id='different',checkpoint=path,resume=True)

    def test_invalid_particles_are_rejected_not_replaced(self):
        count=[0]
        def loglike(t):
            count[0]+=len(t)
            return np.where(t[:,0]<.7,0.,-np.inf)
        r=run_smc(loglike,bounds=[[0,1]],config=SMCConfig(particles=4096,moves=4),seed=49)
        self.assertTrue(np.all(r.theta[:,0]<.7))
        self.assertAlmostEqual(r.log_evidence,np.log(.7),delta=.04)
        self.assertEqual(count[0],r.evaluation_count)
        self.assertEqual(r.evaluation_count,4096*(1+4))
        self.assertGreater(r.invalid_evaluation_count,0)
        self.assertEqual(r.stages[0]['kind'],'support_conditioning')
        self.assertGreater(r.stages[1]['invalid_proposals'],0)
        self.assertEqual(r.stages[1]['proposal_count'],4096*4)

    def test_all_invalid_and_callback_errors_propagate(self):
        count=[0]
        def invalid(theta):count[0]+=len(theta);return np.full(len(theta),np.nan)
        with self.assertRaises(RuntimeError):run_smc(invalid,bounds=[[0,1]],config=SMCConfig(particles=64))
        self.assertEqual(count[0],64)
        def broken(theta):raise ArithmeticError('explicit physics failure')
        with self.assertRaises(ArithmeticError):run_smc(broken,bounds=[[0,1]])

    def test_independent_rng_and_bounded_adaptation(self):
        np.random.seed(982);old=np.random.get_state()
        f=lambda t:-np.sum((t-.4)**2,axis=1)/.003
        config=SMCConfig(particles=256,minimum_scale=.1,maximum_scale=.8)
        a=run_smc(f,bounds=[[0,1]]*5,config=config,seed=32)
        b=run_smc(f,bounds=[[0,1]]*5,config=config,seed=32)
        new=np.random.get_state()
        assert_allclose(old[1],new[1],atol=0);self.assertEqual(old[2:],new[2:])
        assert_allclose(a.theta,b.theta,atol=0,rtol=0)
        for s in a.stages:
            self.assertTrue(.1<=s['next_proposal_scale']<=.8)
            self.assertTrue(0<=s['acceptance_rate']<=1)

    def test_reflection_and_resampling(self):
        assert_allclose(reflect_unit([-4.2,-1.2,-.2,.2,1.2,2.2,4.2]),[.2,.8,.2,.2,.8,.2,.2],atol=1e-14)
        ids=systematic_resample([0,0,1,0],np.random.default_rng(4))
        assert_allclose(ids,[2,2,2,2])
        self.assertAlmostEqual(conditional_ess([1,1,1],[.1,.2,.7],.8),3)

    def test_marginal_gaussian_callback_and_force_posterior(self):
        def mapping(t):
            H=np.zeros((len(t),2,2));H[:,0,0]=np.exp(t[:,0]);H[:,1,1]=1+t[:,0]**2
            return dict(H=H,offset=np.tile([.1,-.2],(len(t),1)))
        y=np.array([.4,-.7]);K=np.array([[1,.2],[.2,.7]]);R=np.array([[.2,.03],[.03,.1]])
        likelihood=MarginalGaussianLikelihood(mapping,y,K,R)
        theta=np.array([[-.3],[.5]])
        for j in range(2):
            mapped=mapping(theta[j:j+1]);p=fit_posterior(mapped['H'],y,K,R,offset=mapped['offset'])
            self.assertAlmostEqual(likelihood(theta[j:j+1])[0],p.log_evidence,places=12)
            m,c=likelihood.conditional(theta[j:j+1]);assert_allclose(m,p.means,atol=1e-12);assert_allclose(c,p.covariances,atol=1e-12)
        r=run_smc(likelihood,bounds=[[-1,1]],config=SMCConfig(particles=1024),seed=7)
        p=r.force_posterior(likelihood.conditional)
        self.assertEqual(p.means.shape,(1024,2));self.assertEqual(p.covariances.shape,(1024,2,2))
        assert_allclose(p.weights,r.weights,atol=0)
        self.assertTrue(np.all(np.linalg.eigvalsh(p.covariances)>0))
        target=p.target(np.eye(2));self.assertTrue(np.all(target.variance>0))


if __name__=='__main__':unittest.main()
