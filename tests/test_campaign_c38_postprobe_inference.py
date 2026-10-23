"""Manufactured acquired probes only; no actual data/files/noise/choices."""
import copy,inspect
import numpy as np
import pytest
from experiments import campaign_c38_postprobe_inference as m
from experiments.campaign_c38_controlled_observations import POINT_IDS
from experiments.campaign_c38_table_validation import POINTS
from test_campaign_c38_public_prefix_inference_v02 import design,payload


def probe(public,F=-2.):
 return dict(information_role='PUBLIC_ACQUIRED_IDEAL_STATIC_PROBE',object_id=public['object_id'],observation_kind='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD',acquisition_index=1,force_Fz_N=float(F),qualified_point_ids=POINT_IDS.copy(),reference_points_m=POINTS[POINT_IDS].copy(),observed_positions_m=POINTS[POINT_IDS].copy(),sensor_covariance_m2=np.eye(27)*.0002**2,reset_from_common_rest=True,is_RGBD_perception=False,is_temporal_forecast=False)


class Forward:
 categories=m.CATEGORIES;young_nodes=np.array([75000.,np.sqrt(75000.*150000.),150000.]);forces_N=(-2.,0.,1.,2.)
 def __init__(self,obs=False):self.calls=[];self.obs=obs
 def observation_coefficients(self,c,F,ids):
  self.calls.append((c,F,tuple(ids)));a=np.zeros((3,len(ids),3))
  if F==1.:
   for j,p in enumerate(ids):
    if p==POINT_IDS[0]:a[1,j,2]=.001
  elif self.obs:a[1,:,2]=(.0001+.00002*F)*(1 if c==self.categories[0] else 1.2)
  return a.reshape(3,-1)


def test_exact_single_probe_allowlist_same_id_acquisition_noise_and_no_counterfactual():
 p=payload();b=probe(p);f=Forward()
 for key,value in [('object_id','b'*32),('acquisition_index',0),('force_Fz_N',1.),('reset_from_common_rest',False),('observed_positions_m',np.zeros((2,9,3))),('sensor_covariance_m2',np.eye(27)*.0001**2)]:
  q=copy.deepcopy(b);q[key]=value
  with pytest.raises(ValueError):m.compile_public_models(f,p,q,design=design())
 assert not f.calls
 for key,value in [('young_Pa',90000.),('initial_posterior_weights',np.ones(10)),('counterfactual_other_probe',np.zeros((9,3))),('canonical_target_positions_m',np.zeros((63,3)))]:
  with pytest.raises(ValueError,match='allowlist'):m.compile_public_models(f,p,dict(b,**{key:value}),design=design())
 assert not f.calls
 assert not {'truth','future','weights','posterior','support','young_Pa'}&set(inspect.signature(m.infer_public_probe).parameters)


def test_stacked54_likelihood_equals_independent_raw_gaussian_sum_no_double_prefix():
 p=payload();b=probe(p,2.);f=Forward(True);models,targets=m.compile_public_models(f,p,b,design=design())
 assert {F for _,F,_ in f.calls}=={0.,1.,2.} and len(f.calls)==6
 for category,model in models.items():
  assert model.observed.shape==(54,) and model.noise.shape==(54,54)
  assert np.array_equal(model.noise[:27,27:],np.zeros((27,27)))
  E=110000.;x=75000/E;coeff=np.concatenate([f.observation_coefficients(category,0.,POINT_IDS),f.observation_coefficients(category,2.,POINT_IDS)],axis=1)
  predicted=coeff[0]+x*coeff[1]+x*x*coeff[2];observed=np.r_[(p['observed_positions_m'][0]-p['reference_points_m']).reshape(-1),(b['observed_positions_m']-b['reference_points_m']).reshape(-1)]
  exact=-.5*54*np.log(2*np.pi*.0002**2)-.5*np.sum(((predicted-observed)/.0002)**2)
  assert model.log_likelihood(E)==pytest.approx(exact,abs=2e-13)
  assert targets[category].shape==(3,189)


def test_fresh_uniform_prior_analytic_canonical_intervals_and_no_IO_or_noise(monkeypatch):
 monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('I/O')))
 monkeypatch.setattr(np.random,'default_rng',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Noise draw')))
 p=payload();b=probe(p);f=Forward();r=m.infer_public_probe(f,p,b,design=design())
 assert r['completed'] and r['numerical_controls_passed'],r['failures']
 assert r['fresh_prior'] and not r['initial_posterior_weights_reused'] and r['acquired_coordinate_count']==54
 assert {F for _,F,_ in f.calls}=={0.,1.,-2.} and len(f.calls)==6
 for row in r['methods'].values():
  np.testing.assert_allclose(row['category_probabilities'],[.5,.5],atol=1e-14)
  np.testing.assert_allclose(row['material_equal_tail_Pa'],75000.*2**np.array(m.PROBABILITIES),rtol=1e-7)
  assert row['canonical_mean_displacement_m'][POINT_IDS[0],2]==pytest.approx(.001*.5/np.log(2),abs=1e-14)
  if 'canonical_displacement_equal_tail_m' in row:np.testing.assert_allclose(row['canonical_displacement_equal_tail_m'][0,2],.001*2**(np.array(m.PROBABILITIES)-1),atol=m.ROOT_POSITION_TOL_M,rtol=0)
 assert len(r['GL64_endpoint_CDF_checks'])==162 and not r['noisy_future_measurement_variance_added']


def test_only_supplied_action_and_opaque_id_invariance():
 f=Forward();p=payload('a'*32);q=payload('b'*32)
 a,ta=m.compile_public_models(f,p,probe(p,-2.),design=design());b,tb=m.compile_public_models(f,q,probe(q,-2.),design=design())
 assert all(F in (0.,-2.,1.) for _,F,_ in f.calls)
 for c in m.CATEGORIES:np.testing.assert_array_equal(a[c].observed,b[c].observed);np.testing.assert_array_equal(ta[c],tb[c])
