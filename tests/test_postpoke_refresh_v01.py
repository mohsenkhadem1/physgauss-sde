"""Analytic bridge, likelihood parity and prospective diagnostic controls."""
import numpy as np
import pytest
from scipy.stats import truncnorm
from experiments import campaign_postpoke_refresh_v01 as model
from experiments import run_postpoke_refresh_v01 as run


def test_fixed_pilot_all_objects_one_frozen_noise_no_action_reselection():
 assert run.DESIGN['objects']==list(range(12))and run.DESIGN['noise_index']==0
 assert run.DESIGN['particles']==8192 and run.DESIGN['MH']['retained']==64000
 seeds=[s for key in['fresh_seeds','bridge_seeds','subsample_seeds']for row in run.DESIGN[key]for s in row]+run.DESIGN['MH_seeds']
 assert len(seeds)==len(set(seeds))


def test_exact_independence_hastings_correction_and_prior_cancellation():
 x=np.array([[.2],[.4],[.6]]);y=np.array([[.3],[.5],[.7]]);a=np.array([-2.,-3.,-4.]);na=np.array([-1.,-4.,-2.]);b=np.array([-4.,-2.,-8.]);nb=np.array([-2.,-3.,-4.]);kind=np.array([0,1,2]);mean=np.array([.1]);inverse=np.array([[5.]])
 correction=np.array([model.proposal_logq(x[:1],mean,inverse)[0]-model.proposal_logq(y[:1],mean,inverse)[0],0.,0.])
 np.testing.assert_allclose(model.mh_log_ratio(x,y,a,na,b,nb,.7,kind,mean,inverse),na-a+.7*(nb-b)+correction)
 np.testing.assert_allclose(model.mh_log_ratio(x,y,a,na,b,nb,0.,kind,mean,inverse),na-a+correction)


def test_gaussian_analytic_posterior_bridge_and_fresh_independent_reference():
 """Uniform bounded prior, two Gaussian observations, analytic truncated normal."""
 muA=.35;sdA=.15;muB=.65;sdB=.12;n=8192
 precision=1/sdA**2+1/sdB**2;mu=(muA/sdA**2+muB/sdB**2)/precision;sd=np.sqrt(1/precision)
 analytic=truncnorm(-mu/sd,(1-mu)/sd,loc=mu,scale=sd)
 def likelihoods(x):return -.5*((x[:,0]-muA)/sdA)**2-np.log(sdA*np.sqrt(2*np.pi)),-.5*((x[:,0]-muB)/sdB)**2-np.log(sdB*np.sqrt(2*np.pi))
 initial=truncnorm.rvs(-muA/sdA,(1-muA)/sdA,loc=muA,scale=sdA,size=(n,1),random_state=2026102901)
 mean=initial.mean(0);cov=np.cov(initial,rowvar=False).reshape(1,1)
 bridge=model.bridge_smc(initial,likelihoods,mean,cov,seed=2026102902)
 fresh=run.tempered_smc(model.cohort.old.base.BoxPrior(1),lambda state:sum(likelihoods(state.coordinates)),particles=n,seed=2026102903,ess_fraction=.8,rejuvenation_steps=16,proposal_scale=.08,independence_probability=.25,max_stages=64,periodic_indices=())
 for x in [bridge.unit[:,0],fresh.state.coordinates[:,0]]:
  assert abs(x.mean()-analytic.mean())<.004
  assert abs(x.std()-analytic.std())<.004
  knots=np.linspace(.2,.8,21);assert np.max(abs((x[:,None]<=knots).mean(0)-analytic.cdf(knots)))<.025
 assert bridge.stages[-1]['beta']==1. and bridge.initialization_approximate
 assert all(sum(s['accepted'])>0 for s in bridge.stages)
 from scipy.stats import norm
 prefixZ=norm.cdf((1-muA)/sdA)-norm.cdf(-muA/sdA)
 jointZ=norm.pdf(muA,loc=muB,scale=np.sqrt(sdA**2+sdB**2))*(norm.cdf((1-mu)/sd)-norm.cdf(-mu/sd))
 assert abs(bridge.log_conditional_evidence_estimate-np.log(jointZ/prefixZ))<.05
 assert abs(fresh.log_evidence_estimate-np.log(jointZ))<.05
 assert np.isin(bridge.unit[:,0],initial[:,0]).mean()<.01


def test_invalid_initial_cloud_and_proposal_fail_closed():
 def f(x):return np.zeros(len(x)),np.zeros(len(x))
 for initial,mean,cov in[(np.array([[-.1],[.2]]),np.array([.2]),np.eye(1)),(np.array([[.1],[.2]]),np.array([np.nan]),np.eye(1)),(np.array([[.1],[.2]]),np.array([.2]),np.array([[np.nan]]))]:
  with pytest.raises(ValueError):model.bridge_smc(initial,f,mean,cov,seed=1)


def test_joint_observations_equal_original_scalar_and_independent_numpy():
 """Tiny finite-model parity only; no cohort truths or new posterior calls."""
 import torch
 torch.set_num_threads(1)
 unit=np.array([[.2,.3,.4,.5],[.8,.7,.6,.4]]);theta=model.cohort.physical(unit);poke=list(model.cohort.old.poke_programs().values())[4];a=np.arange(12)*1e-5;b=np.arange(12)[::-1]*1e-5
 joint=model.JointObservationForward(poke,a,b,chunk=1);pred=joint.predictions(unit);scalar=model.cohort.old.ArbitraryForceForward()
 expected=np.stack([model.cohort.old.original.selected_observations(scalar(theta,force))for force in[model.cohort.old.original.known_force('A'),poke]],axis=1)
 np.testing.assert_allclose(pred,expected,atol=1e-13,rtol=1e-12)
 for k,force in enumerate([model.cohort.old.original.known_force('A'),poke]):
  u=model.cohort.old.independent_numpy_forward(theta[0],force);oracle=model.cohort.old.original.selected_observations(u[None])[0];np.testing.assert_allclose(pred[0,k],oracle,atol=1e-13,rtol=1e-12)
 sigma=model.cohort.old.original.SIGMA;normalizer=12*np.log(sigma*np.sqrt(2*np.pi));la,lb=joint.log_components(unit)
 np.testing.assert_allclose(la,-.5*np.sum(((expected[:,0]-a)/sigma)**2,axis=1)-normalizer,atol=1e-9,rtol=1e-12)
 np.testing.assert_allclose(lb,-.5*np.sum(((expected[:,1]-b)/sigma)**2,axis=1)-normalizer,atol=1e-9,rtol=1e-12)
 np.testing.assert_allclose(joint.joint_log_likelihood(unit),la+lb,atol=1e-12)
 assert not np.allclose(joint.joint_log_likelihood(unit),la)


def test_incomplete_diagnostic_cohort_rejected_all_failed_objects_retained():
 rows=[dict(index=i,completed=True,passed=i!=2)for i in range(12)]
 r=run.aggregate(rows);assert r['objects']==12 and r['numerically_passed_objects']==11 and not r['all_numerical_gates_passed']and not r['policy_advantage_claim']
 with pytest.raises(ValueError):run.aggregate(rows[:-1])
