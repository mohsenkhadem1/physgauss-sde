import copy
import inspect
import numpy as np
import pytest
from experiments import campaign_c38_public_prefix_inference_v02 as worker
from experiments.campaign_c38_controlled_observations import public_prefix_payload,POINT_IDS
from experiments.campaign_c38_table_validation import POINTS
from experiments.campaign_c38_polynomial_likelihood_v02 import GaussianPolynomialLikelihood,PolynomialPosterior

def design():
 return dict(schema='C38-controlled-static-ideal-inverse-DESIGN-v01',sensor=dict(kind='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD',qualified_point_ids=POINT_IDS.tolist(),reference_points_m=POINTS[POINT_IDS].tolist(),noise=dict(distribution='independent_zero_mean_Gaussian_per_Cartesian_coordinate_per_acquisition',sigma_m=.0002,known_stipulated_not_sensor_calibrated=True)),acquisition=dict(prefix_Fz_N=0.,candidate_probe_Fz_N=[-2.,2.],target_Fz_N=1.,same_latent_across_resets=True,each_probe_reset_from_common_rest=True),noise_repeats_per_configuration=16,independent_noise_seeds=dict(root_seed=2026102301))

class FakeGenerator:
 def standard_normal(self,shape):return np.zeros(shape)

def payload(identity='a'*32):return public_prefix_payload(identity,POINTS[POINT_IDS]+.0002*FakeGenerator().standard_normal((9,3)),design=design())

class ManufacturedForward:
 categories=worker.CATEGORIES
 young_nodes=np.array([75000.,np.sqrt(75000.*150000.),150000.])
 forces_N=(-2.,0.,1.,2.)
 def __init__(self,variable=True):self.calls=[];self.variable=variable
 def observation_coefficients(self,category,F,ids):
  self.calls.append((category,F,tuple(ids)));c=np.zeros((3,len(ids),3))
  if F==1. and self.variable:
   for j,p in enumerate(ids):
    if p==POINT_IDS[0]:c[1,j,2]=.001
  if F not in (0.,1.):raise AssertionError('Unacquired force requested')
  return c.reshape(3,-1)

def test_independent_analytic_linear_prior_forecast_and_no_draw_or_IO(monkeypatch):
 monkeypatch.setattr(np.random,'default_rng',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Actual noise draw')))
 monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Archive read')))
 f=ManufacturedForward();r=worker.infer_public_prefix(f,payload(),design=design())
 assert r['completed'] and r['numerical_controls_passed'],r['failures']
 assert set(F for _,F,_ in f.calls)=={0.,1.} and len(f.calls)==4
 for row in r['methods'].values():
  np.testing.assert_allclose(row['category_probabilities'],[.5,.5],atol=1e-14)
  np.testing.assert_allclose(row['material_equal_tail_Pa'],75000.*2**np.array(worker.PROBABILITIES),rtol=1e-7)
  assert row['canonical_mean_displacement_m'][POINT_IDS[0],2]==pytest.approx(.001*.5/np.log(2),abs=1e-14)
  if 'canonical_displacement_equal_tail_m' in row:np.testing.assert_allclose(row['canonical_displacement_equal_tail_m'][0,2],.001*2**(np.array(worker.PROBABILITIES)-1),atol=worker.ROOT_POSITION_TOL_M,rtol=0)
 for method in ['adaptive','GL128']:
  row=r['methods'][method]
  for p,q in zip(worker.PROBABILITIES,row['canonical_quantile_details'][2]['quantiles']):
   exact=.001*2**(p-1);assert q['lower_m']<=exact<=q['upper_m'] and q['upper_m']-q['lower_m']<=worker.ROOT_POSITION_TOL_M
  assert np.all(row['canonical_variance_integration_resolution_estimate_m2']>=0)
 assert not r['noisy_future_measurement_variance_added'] and len(r['GL64_endpoint_CDF_checks'])==162


def test_public_tamper_rejected_before_hypotheses_and_opaque_id_not_feature():
 f=ManufacturedForward(False);p=payload();bad=dict(p,young_Pa=90000.)
 with pytest.raises(ValueError,match='allowlist'):worker.compile_public_models(f,bad,design=design())
 assert not f.calls
 r=worker.infer_public_prefix(f,bad,design=design());assert not r['completed'] and not r['numerical_controls_passed']
 bad=dict(p,canonical_target_positions_m=np.zeros((63,3)));assert not worker.infer_public_prefix(f,bad,design=design())['completed'];assert not f.calls
 a,ta=worker.compile_public_models(f,payload('a'*32),design=design());b,tb=worker.compile_public_models(f,payload('b'*32),design=design())
 for c in worker.CATEGORIES:np.testing.assert_array_equal(a[c].observed,b[c].observed);np.testing.assert_array_equal(ta[c],tb[c])
 assert not {'truth','future','young_Pa','support'}&set(inspect.signature(worker.infer_public_prefix).parameters)


def test_constant_and_mixed_atoms_and_unresolved_quantile_preserved():
 m=GaussianPolynomialLikelihood(np.zeros((3,1)),[0.],[[1.]]);p=PolynomialPosterior({'a':m,'b':m})
 same={'a':np.array([.001,0.,0.]),'b':np.array([.001,0.,0.])}
 for prob in worker.PROBABILITIES:assert worker.coordinate_quantile(p,same,prob)['kind']=='common_constant_atom'
 mixed={'a':np.array([-.001,0.,0.]),'b':np.array([.001,0.,0.])}
 assert worker.coordinate_quantile(p,mixed,.025)['value_m']==-.001
 assert worker.coordinate_quantile(p,mixed,.975)['value_m']==.001
 with pytest.raises(FloatingPointError):worker.coordinate_quantile(p,mixed,.5)


def test_frozen_quantile_ambiguity_is_failure_not_point_estimate():
 with pytest.raises(FloatingPointError,match='unresolved'):
  worker.bounded_quantile(lambda x:dict(cdf=x,lower=max(0.,x-.3),upper=min(1.,x+.3)),0.,1.,.5,tolerance=1e-10)
