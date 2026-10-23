import inspect
import numpy as np
import pytest
from experiments import campaign_c39_typed_static_policy as p
from experiments.campaign_c38_controlled_observations import public_prefix_payload,POINT_IDS
from experiments.campaign_c38_table_validation import POINTS


def design():
 return dict(schema='C38-controlled-static-ideal-inverse-DESIGN-v01',sensor=dict(kind='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD',qualified_point_ids=POINT_IDS.tolist(),reference_points_m=POINTS[POINT_IDS].tolist(),noise=dict(distribution='independent_zero_mean_Gaussian_per_Cartesian_coordinate_per_acquisition',sigma_m=.0002,known_stipulated_not_sensor_calibrated=True)),acquisition=dict(prefix_Fz_N=0.,candidate_probe_Fz_N=[-2.,2.],target_Fz_N=1.,same_latent_across_resets=True,each_probe_reset_from_common_rest=True),noise_repeats_per_configuration=16,independent_noise_seeds=dict(root_seed=2026102301))


class Initial:
 categories=('both_ends','separated_bottom_patches');young_nodes=np.array([75000.,np.sqrt(75000.*150000.),150000.]);forces_N=(-2.,0.,1.,2.)
 def __init__(self):self.calls=[]
 def observation_coefficients(self,c,F,ids):
  self.calls.append((c,F,tuple(ids)));a=np.zeros((3,len(ids),3))
  if F==0.:a[0,:,2]=.02*(c==self.categories[1])
  elif F==1.:a[1,:,2]=.001
  elif F==-2.:a[1,:,2]=.002 if c==self.categories[0]else.0005
  else:raise AssertionError('SIDE must never be scalarFz+2')
  return a.reshape(3,-1)


class Composite:
 def __init__(self,equal=False):self.initial_candidate=Initial();self.categories=self.initial_candidate.categories;self.young_nodes=self.initial_candidate.young_nodes.copy();self.calls=[];self.equal=equal
 def observation_coefficients(self,c,action_id,ids):
  self.calls.append((c,action_id,tuple(ids)))
  if action_id=='middle_down_2N':return self.initial_candidate.observation_coefficients(c,-2.,ids)
  if action_id!='middle_side_2N':raise AssertionError('Typed action required')
  if self.equal:return self.initial_candidate.observation_coefficients(c,-2.,ids)
  a=np.zeros((3,len(ids),3));a[1,:,2]=.001 if c==self.categories[0]else.002;return a.reshape(3,-1)


def payload(category):
 q=POINTS[POINT_IDS].copy();q[:,2]+=.02*category
 return public_prefix_payload('a'*32,q,design=design())


def test_typed_world_actions_manufactured_support_dependent_flip_and_no_Fz_alias(monkeypatch):
 monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No actual archives')))
 for category,expected in [(0,'middle_down_2N'),(1,'middle_side_2N')]:
  f=Composite();r=p.design_typed_policy(f,payload(category),design=design());assert r['completed']and r['numerical_cloud_controls_passed'],r['failures']
  for criterion in ['information','target_risk']:
   assert r[criterion]['stable']and r[criterion]['selected_action']['action_id']==expected
   assert 'selected_force_Fz_N'not in r[criterion]
  assert {F for c,F,ids in f.initial_candidate.calls}=={0.,1.,-2.}
  assert {a for c,a,ids in f.calls}=={'middle_down_2N','middle_side_2N'}
  assert r['actions'][1]==dict(action_id='middle_side_2N',force_world_N=[0.,2.,0.])
  assert r['fantasy_stream']['root_seed']==2026102307 and r['random_root_seed']==2026102306
  assert r['per_policy_extra_coordinates']==27 and r['per_policy_total_coordinates']==54
  assert not r['choices_frozen']and not r['acquired_DOWN_observations_or_posteriors_access']


def test_acquired_DOWN_or_SIDE_payload_or_truth_rejected_before_hypotheses():
 for key in ['acquired_DOWN_observations_m','DOWN_posteriors','SIDE_observations_m','true_young_Pa','canonical_target_positions_m']:
  f=Composite();r=p.design_typed_policy(f,dict(payload(0),**{key:0}),design=design());assert not r['completed'];assert not f.calls and not f.initial_candidate.calls
 assert tuple(inspect.signature(p.design_typed_policy).parameters)==('composite','public','design')


def test_exact_tie_fixed_DOWN_and_design_seeds():
 f=Composite(equal=True);r=p.design_typed_policy(f,payload(0),design=design());assert r['completed']
 for criterion in ['information','target_risk']:assert r[criterion]['fallback_used']and r[criterion]['selected_action']==p.ACTIONS[0].record()
 assert p.DESIGN_RECORD['sha256']=='3555d4ad8235efa72b1c606e626cbab262a1fb7c1729b4f79f6a9fd669f0e321'
 assert np.linalg.norm(p.ACTIONS[0].force_world_N)==np.linalg.norm(p.ACTIONS[1].force_world_N)==2.
