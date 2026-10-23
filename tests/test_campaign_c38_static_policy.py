import copy
import inspect
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from experiments import campaign_c38_static_policy as p
from experiments.campaign_c38_polynomial_likelihood_v02 import GaussianPolynomialLikelihood,PolynomialPosterior
from experiments.campaign_c38_controlled_observations import public_prefix_payload,POINT_IDS
from experiments.campaign_c38_table_validation import POINTS


def design():
 return dict(schema='C38-controlled-static-ideal-inverse-DESIGN-v01',sensor=dict(kind='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD',qualified_point_ids=POINT_IDS.tolist(),reference_points_m=POINTS[POINT_IDS].tolist(),noise=dict(distribution='independent_zero_mean_Gaussian_per_Cartesian_coordinate_per_acquisition',sigma_m=.0002,known_stipulated_not_sensor_calibrated=True)),acquisition=dict(prefix_Fz_N=0.,candidate_probe_Fz_N=[-2.,2.],target_Fz_N=1.,same_latent_across_resets=True,each_probe_reset_from_common_rest=True),noise_repeats_per_configuration=16,independent_noise_seeds=dict(root_seed=2026102301))


def payload():return public_prefix_payload('a'*32,POINTS[POINT_IDS].copy(),design=design())


class WorkModel:
 categories=('both_ends','separated_bottom_patches')
 young_nodes=np.array([75000.,np.sqrt(75000.*150000.),150000.])
 forces_N=(-2.,0.,1.,2.)
 def __init__(self,uninformative=False):self.calls=[];self.uninformative=uninformative
 def observation_coefficients(self,category,F,ids):
  self.calls.append((category,F,tuple(ids)));c=np.zeros((3,len(ids),3))
  if F==1.:c[1,:,2]=.001*(1.+.1*(category==self.categories[1]))
  elif F in (-2.,2.) and not self.uninformative:c[1,:,2]=.0005*(1. if F<0 else 2.)
  return c.reshape(3,-1)


def test_linear_gaussian_oracle_and_total_variance():
 # Positive Hermite cloud is an independent analytic-Gaussian control of math,
 # not the logE GL cloud used in actual policy construction.
 x,w=hermgauss(80);x=np.sqrt(2)*x;w/=np.sqrt(np.pi);y=np.array([[-1.],[0.],[1.]])
 r=p.fantasy_utilities(w,np.log(w),x[:,None],y,np.eye(1),x[:,None])
 np.testing.assert_allclose(r['target_variance_trace'],.5,atol=1e-13)
 np.testing.assert_allclose(r['information_nats'],.5*(np.log(2)-.5+y[:,0]**2/4),atol=1e-13)
 q,gw=hermgauss(24);Y=2*q[:,None];gw/=np.sqrt(np.pi) # predictive Y~N(0,2)
 r=p.fantasy_utilities(w,np.log(w),x[:,None],Y,np.eye(1),x[:,None])
 assert gw@r['information_nats']==pytest.approx(.5*np.log(2),abs=1e-11)
 assert gw@r['target_variance_trace']+gw@(Y[:,0]/2)**2==pytest.approx(1.,abs=1e-11)


def test_uninformative_view_retains_variance_and_weights_without_fake_noise():
 w=np.array([.1,.2,.7]);T=np.array([[0.,0.],[.001,0.],[.004,.002]])
 r=p.fantasy_utilities(w,np.log(w),np.zeros((3,1)),np.array([[-10.],[0.],[10.]]),np.array([[.2**2]]),T)
 mean=w@T;variance=np.sum(w[:,None]*(T-mean)**2)
 np.testing.assert_allclose(r['target_variance_trace'],variance,rtol=0,atol=1e-19)
 np.testing.assert_allclose(r['information_nats'],0.,atol=1e-12)


def test_positive_partition_cloud_logE_measure_and_rare_category():
 m=GaussianPolynomialLikelihood(np.zeros((3,1)),[0.],[[1.]])
 post=PolynomialPosterior({'a':m,'b':m},category_prior=[1e-20,1-1e-20]);T={'a':np.array([[0.],[0.],[0.]]),'b':np.array([[.001],[0.],[0.]])};A=[{'a':np.zeros((3,1)),'b':np.zeros((3,1))}]*2
 c=p.positive_cloud(post,T,A,order=32);reference=post.target_moments(T)
 assert p.check_cloud(c,post,reference)['passed'];assert set(c['category_index'])=={0,1}
 assert np.sum(c['weights'][c['category_index']==0])==pytest.approx(1e-20,rel=1e-13)
 assert c['target_variance_m2'][0]==pytest.approx(1e-26,rel=1e-13)
 T={k:np.array([[0.],[1.],[0.]]) for k in T};c=p.positive_cloud(post,T,A,order=32)
 assert c['target_mean_m'][0]==pytest.approx(.5/np.log(2),abs=1e-14)
 assert np.min(c['young_Pa'])>75000 and np.max(c['young_Pa'])<150000


def test_public_role_barrier_before_candidate_and_no_actual_outcome_API(monkeypatch):
 monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No data files')))
 for key in ['young_Pa','true_support','candidate_probe_positions_m','canonical_target_positions_m']:
  f=WorkModel();bad=dict(payload(),**{key:0});r=p.design_static_policy(f,bad,design=design())
  assert not r['completed'] and f.calls==[]
 assert tuple(inspect.signature(p.design_static_policy).parameters)==('candidate','public','design')


def test_shared_fantasy_prefixes_stable_seed_and_exact_tie_fallback(monkeypatch):
 monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No data files')))
 f=WorkModel(True);r=p.design_static_policy(f,payload(),design=design())
 assert r['completed'] and r['numerical_cloud_controls_passed'],r['failures']
 assert not r['choices_frozen'] and r['per_policy_measurement_budget_coordinates']==27
 assert {F for _,F,_ in f.calls}=={0.,1.,-2.,2.}
 for criterion in ['information','target_risk']:
  assert r[criterion]['selected_action_index']==0 and r[criterion]['fallback_used']
 assert r['fantasy_stream']['same_fantasy_outcomes_all_cloud_orders']
 rng1=np.random.default_rng(p._seed(p.FANTASY_ROOT_SEED,'a'*32));rng2=np.random.default_rng(p._seed(p.FANTASY_ROOT_SEED,'a'*32));np.testing.assert_array_equal(rng1.standard_normal(100),rng2.standard_normal(100))
 assert p._seed(p.FANTASY_ROOT_SEED,'a'*32).entropy!=p._seed(p.FANTASY_ROOT_SEED,'b'*32).entropy


def test_paired_MCSE_and_order_instability_fallback():
 rows={o:dict(information_nats=np.stack([np.zeros(256),np.ones(256)]),target_mean_coordinate_variance_m2=np.stack([np.ones(256),np.zeros(256)])) for o in p.ORDERS}
 for criterion in ['information','risk']:
  r=p.summarize_decision(rows,criterion=criterion);assert r['stable'] and r['selected_action_index']==1
  assert r['budgets']['256']['cloud_orders']['128']['paired_advantage_MCSE']==0
 rows[32]['information_nats'][1]*=-1
 r=p.summarize_decision(rows,criterion='information');assert not r['stable'] and r['selected_action_index']==0


def test_embedding_preserves_full_field_distance_without_truncation():
 rng=np.random.default_rng(8);C={c:rng.normal(size=(3,189))*.001 for c in ['a','b']};x=np.linspace(.5,1,40);T=np.array([np.array([1.,v,v*v])@C[c] for c in C for v in x]);Z,D=p._target_embedding(C,dict(targets_m=T));w=np.ones(len(T))/len(T)
 r=p.fantasy_utilities(w,np.log(w),np.zeros((len(T),1)),np.zeros((1,1)),np.eye(1),Z)
 assert r['target_variance_trace'][0]/D==pytest.approx(np.mean(np.var(T,axis=0)),abs=1e-20)
