from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from censored_range_observation import (CensoredRanges,CensoredRangeModel,ranges_from_depth,score_censored_actions,centered_target_gram,gram_target_risk)

class CensoredObservationTests(unittest.TestCase):
    def test_missing_values_are_categorical_and_never_sentinel_ranges(self):
        a=ranges_from_depth(np.array([.8,0,np.inf,np.nan]),np.full(4,.8))
        np.testing.assert_array_equal(a.visible,[True,False,False,False])
        np.testing.assert_array_equal(a.residual_m,np.zeros(4))
        model=CensoredRangeModel(np.eye(4)*1e-6,.1)
        predictions=CensoredRanges(np.zeros((2,4),bool),np.full((2,4),np.nan))
        posterior,_=model.condition(predictions,a,[.2,.8])
        np.testing.assert_allclose(posterior,[.2,.8],atol=1e-14)

    def test_mixed_density_normalizes_including_missing_observation(self):
        model=CensoredRangeModel([[.01**2]],.13,background_std_m=.08,outlier_probability=.17)
        for hit in [True,False]:
            prediction=CensoredRanges([[hit]],[[.035]])
            grid=np.linspace(-.65,.65,2501)
            density=np.array([np.exp(model.log_prob(prediction,CensoredRanges([True],[y]))[0]) for y in grid])
            hitmass=np.trapezoid(density,grid)
            missmass=np.exp(model.log_prob(prediction,CensoredRanges([False],[np.nan]))[0])
            self.assertAlmostEqual(hitmass,.83*(.87 if hit else .13)+.17*.5,places=7)
            self.assertAlmostEqual(hitmass+missmass,1.,places=7)

    def test_submatrix_normalization_and_logdet_are_retained(self):
        covariance=1e-4*np.array([[2,.3,.7],[.3,1,.2],[.7,.2,3.]])
        model=CensoredRangeModel(covariance,.1,outlier_probability=0)
        prediction=CensoredRanges([[True,True,True]],[[.01,.02,-.03]])
        observation=CensoredRanges([True,False,True],[.02,np.nan,-.01])
        indices=[0,2];sub=covariance[np.ix_(indices,indices)];d=np.array([.01,.02])
        expected=2*np.log(.9)+np.log(.1)-.5*(d@np.linalg.solve(sub,d)+2*np.log(2*np.pi)+np.linalg.slogdet(sub)[1])
        self.assertAlmostEqual(model.log_prob(prediction,observation)[0],expected,places=12)

    def test_background_mixture_matches_explicit_formula(self):
        model=CensoredRangeModel([[.002**2]],.08,background_std_m=.1,outlier_probability=.25)
        prediction=CensoredRanges([[True]],[[.01]])
        observation=CensoredRanges([True],[.035])
        normal=lambda d,s:np.exp(-.5*(d/s)**2)/(np.sqrt(2*np.pi)*s)
        expected=np.log(.75*.92*normal(.025,.002)+.25*.5*normal(.035,.1))
        self.assertAlmostEqual(model.log_prob(prediction,observation)[0],expected,places=12)

    def test_fantasies_and_conditioning_share_same_generative_model(self):
        model=CensoredRangeModel([[.0001,.00003],[.00003,.0002]],.12,background_std_m=.08,outlier_probability=.2)
        prediction=CensoredRanges([[True,False],[True,True]],[[-.03,0],[.03,.02]])
        prior=np.array([.3,.7]);fantasies,metadata=model.fantasies(prediction,prior,count=3000,seed=52)
        posterior=np.array([model.condition(prediction,CensoredRanges(v,r),prior)[0] for v,r in zip(fantasies.visible,fantasies.residual_m)])
        np.testing.assert_allclose(posterior.mean(0),prior,atol=.02)
        self.assertAlmostEqual(metadata['outlier_branch'].mean(),.2,delta=.025)
        self.assertTrue(np.isfinite(posterior).all())

    def test_design_prefers_informative_action_and_reports_mc_error(self):
        model=CensoredRangeModel([[.002**2]],.03,background_std_m=.05,outlier_probability=.02)
        target=np.array([[-1.],[1.]])
        null=CensoredRanges([[True],[True]],[[0.],[0.]])
        informative=CensoredRanges([[True],[True]],[[-.02],[.02]])
        scores=score_censored_actions(target,[null,informative],model,fantasies=300,seed=3)
        self.assertEqual(scores['selected_index'],1)
        self.assertAlmostEqual(scores['actions'][0]['expected_target_risk'],1.,places=12)
        self.assertLess(scores['actions'][1]['expected_target_risk'],.2)
        self.assertGreaterEqual(scores['actions'][1]['target_risk_mc_se'],0)
        self.assertGreater(scores['actions'][1]['expected_information_nats'],.4)

    def test_lowrank_subcovariance_matches_dense_gaussian_exactly(self):
        rng=np.random.default_rng(19);factor=rng.normal(0,.003,(12,2))
        covariance=.001**2*np.eye(12)+factor@factor.T
        model=CensoredRangeModel(covariance,.07,outlier_probability=0)
        self.assertEqual(model._shared_factor.shape[1],2)
        h=np.ones((3,12),bool);mu=rng.normal(0,.003,(3,12));v=rng.random(12)>.3;y=rng.normal(0,.003,12)
        result=model.log_prob(CensoredRanges(h,mu),CensoredRanges(v,y))
        sub=covariance[np.ix_(v,v)];difference=y[v][None]-mu[:,v]
        quadratic=np.sum(difference*np.linalg.solve(sub,difference.T).T,-1)
        expected=v.sum()*np.log(.93)+(~v).sum()*np.log(.07)-.5*(quadratic+v.sum()*np.log(2*np.pi)+np.linalg.slogdet(sub)[1])
        np.testing.assert_allclose(result,expected,rtol=1e-12,atol=1e-10)

    def test_gram_risk_matches_direct_mean_and_is_translation_invariant(self):
        rng=np.random.default_rng(57);targets=rng.normal(0,.003,(9,17,5));w=rng.random(9);w/=w.sum()
        gram=centered_target_gram(targets,w)
        for _ in range(6):
            posterior=rng.random(9);posterior/=posterior.sum()
            mean=np.tensordot(posterior,targets,axes=1)
            direct=float(posterior@np.square(targets-mean).reshape(9,-1).mean(-1))
            self.assertAlmostEqual(gram_target_risk(gram,posterior),direct,places=17)
        np.testing.assert_allclose(centered_target_gram(targets+1e3,w),gram,rtol=1e-10,atol=1e-15)
        constant=np.broadcast_to(np.arange(40)*1e10,(9,40))
        np.testing.assert_array_equal(centered_target_gram(constant,w),np.zeros((9,9)))

    def test_constant_targets_and_identical_actions_have_zero_paired_uncertainty(self):
        model=CensoredRangeModel([[.002**2]],.03,outlier_probability=.05)
        candidate=CensoredRanges([[True],[True]],[[-.02],[.02]])
        result=score_censored_actions(np.full((2,31*455*3),.25),[candidate,candidate],model,fantasies=30,seed=15)
        self.assertEqual(result['prior_target_risk'],0.)
        self.assertEqual(result['selected_advantage_standard_error'],[0.,0.])
        self.assertEqual(result['expected_risk_reduction'],[0.,0.])
        a=score_censored_actions(np.array([[-1.],[1.]]),[candidate,candidate],model,fantasies=30,seed=15)
        self.assertEqual(a['selected_advantage_standard_error'],[0.,0.])
        self.assertEqual(a['actions'][0],{**a['actions'][1],'action_index':0})

    def test_non_positive_covariance_and_invalid_probabilities_rejected(self):
        for e in [0,.5,1]:
            with self.assertRaises(ValueError):CensoredRangeModel([[1.]],e)
        with self.assertRaises(np.linalg.LinAlgError):CensoredRangeModel([[0.]],.1)

if __name__=='__main__':unittest.main()
