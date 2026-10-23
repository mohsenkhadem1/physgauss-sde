"""Public-prefix-only C39 DOWN vs SIDE design; typed world actions, no Fz alias.

The old gravity-prefix likelihood, material/support prior and canonical+1 target
are compiled by the unchanged C38 function on composite.initial_candidate.
Acquired DOWN measurements/posteriors and SIDE outcomes are NOT accepted inputs.
The frozen experiment is a paired development extension, not a population test.
"""
from __future__ import annotations
from typing import NamedTuple
import hashlib
import numpy as np
from experiments import campaign_c38_static_policy as math
from experiments.campaign_c38_public_prefix_inference_v02 import compile_public_models
from experiments.campaign_c38_polynomial_likelihood_v02 import PolynomialPosterior

DESIGN_RECORD=dict(path='runs/operator-dev/c39-controlled-extension-design-v01/design.json',sha256='3555d4ad8235efa72b1c606e626cbab262a1fb7c1729b4f79f6a9fd669f0e321')
FANTASY_ROOT_SEED=2026102307
RANDOM_ROOT_SEED=2026102306


class Action(NamedTuple):
 action_id:str
 force_world_N:tuple
 def record(self):return dict(action_id=self.action_id,force_world_N=list(self.force_world_N))


ACTIONS=(Action('middle_down_2N',(0.,0.,-2.)),Action('middle_side_2N',(0.,2.,0.)))


def compile_hypotheses(composite,public,*,design):
 """Only the original acquired prefix is validated before hypothesis queries."""
 models,targets=compile_public_models(composite.initial_candidate,public,design=design)
 if tuple(composite.categories)!=tuple(models) or not np.array_equal(composite.young_nodes,composite.initial_candidate.young_nodes):raise ValueError('Same original material/support hypotheses required')
 coefficients=[]
 for action in ACTIONS:
  values={c:np.asarray(composite.observation_coefficients(c,action.action_id,public['qualified_point_ids']),dtype=float)for c in models}
  if any(v.shape!=(3,27)or not np.isfinite(v).all()for v in values.values()):raise ValueError('Typed finite hypothetical27-coordinate probe coefficients required')
  if action.action_id=='middle_down_2N':
   for c,v in values.items():
    if not np.array_equal(v,composite.initial_candidate.observation_coefficients(c,-2.,public['qualified_point_ids'])):raise ValueError('DOWN must reuse the identical original candidate field')
  coefficients.append(values)
 return models,targets,coefficients


def common_fantasies(cloud,noise_covariance,object_id):
 if cloud['order']!=128:raise ValueError('Highest128 cloud generates all common fantasies')
 A=cloud['action_predictions_m'];rng=np.random.default_rng(math._seed(FANTASY_ROOT_SEED,object_id));ancestors=rng.choice(len(cloud['weights']),size=256,p=cloud['weights']);noise=rng.standard_normal((256,27))@np.linalg.cholesky(noise_covariance).T
 return dict(ancestors=ancestors,noise_m=noise,observations_m=A[:,ancestors,:]+noise[None,:,:])


def summarize_typed_decision(rows,*,criterion):
 """Same C38 paired-MC/order rule; metadata contains typed actions only."""
 if criterion not in ('information','risk'):raise ValueError('Fixed information or canonical-risk criterion only')
 key='information_nats'if criterion=='information'else'target_mean_coordinate_variance_m2';budgets={};raw=[];stable=True;advantages={}
 for count in math.FANTASY_COUNTS:
  entries={};advs=[]
  for order in math.ORDERS:
   pair=np.asarray(rows[order][key])[:,:count];delta=pair[1]-pair[0]if criterion=='information'else pair[0]-pair[1];advantage=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(count));index=int(advantage>0);raw.append(index);advs.append(advantage)
   entries[str(order)]=dict(action_scores=pair.mean(axis=1),paired_SIDE_advantage=advantage,paired_advantage_MCSE=se,raw_argbest_action_id=ACTIONS[index].action_id,MC_count=count)
   stable &= advantage!=0 and abs(advantage)>math.DECISION_RULE['paired_MCSE_multiplier']*se
  discrepancy=max(advs)-min(advs);stable &= abs(advs[-1])>math.DECISION_RULE['cloud_advantage_discrepancy_multiplier']*discrepancy
  scores=np.array([entries[str(o)]['action_scores']for o in math.ORDERS]);budgets[str(count)]=dict(cloud_orders=entries,maximum_cloud_advantage_discrepancy=discrepancy,per_action_cloud_score_discrepancy=np.max(scores,axis=0)-np.min(scores,axis=0));advantages[count]=advs[-1]
 stable &= len(set(raw))==1;index=raw[-1]if stable else 0
 return dict(selected_action_index=int(index),selected_action=ACTIONS[index].record(),stable=bool(stable),fallback_used=not bool(stable),fallback_reason=None if stable else'Tie, pairedMC uncertainty, cloud-order discrepancy or budget/order instability; fixedDOWN, no extra fantasies',budgets=budgets,final_advantage_change_128_to_256=advantages[256]-advantages[128],scope='Numerical/MC stability under the stipulated model; not measured action advantage')


def design_typed_policy(composite,public,*,design):
 """Zero-I/O: original PUBLIC gravity prefix plus public candidate hypotheses.

 The legacy prefix planned_Fz list is preserved solely for exact C38 validation;
 it is NEVER relabelled as a SIDE action. This function's separate typed catalog
 defines the C39 probes. Qualification/sealing belongs to the external runtime.
 """
 result=dict(completed=False,numerical_cloud_controls_passed=False,choices_frozen=False,extension_design=DESIGN_RECORD.copy(),information_role='C39_PUBLIC_GRAVITY_PREFIX_ONLY_TYPED_STATIC_POLICY',actions=[a.record()for a in ACTIONS],failures=[],private_E_BC_target_or_probe_outcome_access=False,acquired_DOWN_observations_or_posteriors_access=False,physical_population_calibration_established=False,is_RGBD_perception=False,is_temporal_forecast=False,per_policy_extra_coordinates=27,per_policy_total_coordinates=54,decision_rule=math.DECISION_RULE.copy())
 try:
  models,targets,coefficients=compile_hypotheses(composite,public,design=design);result['object_id']=public['object_id'];post=PolynomialPosterior(models);reference=post.target_moments(targets);clouds={o:math.positive_cloud(post,targets,coefficients,order=o)for o in math.ORDERS};checks={str(o):math.check_cloud(clouds[o],post,reference)for o in math.ORDERS};result['cloud_checks']=checks
  if not all(c['passed']for c in checks.values()):raise FloatingPointError('Positive-cloud controls failed BEFORE any action scoring')
  result['numerical_cloud_controls_passed']=True;fantasies=common_fantasies(clouds[128],public['sensor_covariance_m2'],public['object_id']);rows={}
  for order,cloud in clouds.items():
   T,D=math._target_embedding(targets,cloud);u=[math.fantasy_utilities(cloud['weights'],cloud['log_weights'],cloud['action_predictions_m'][i],fantasies['observations_m'][i],public['sensor_covariance_m2'],T)for i in range(2)]
   rows[order]=dict(information_nats=np.asarray([r['information_nats']for r in u]),target_mean_coordinate_variance_m2=np.asarray([r['target_variance_trace']/D for r in u]))
  random=int(np.random.default_rng(math._seed(RANDOM_ROOT_SEED,public['object_id'])).integers(2));result.update(completed=True,information=summarize_typed_decision(rows,criterion='information'),target_risk=summarize_typed_decision(rows,criterion='risk'),fixed_down=ACTIONS[0].record(),fixed_side=ACTIONS[1].record(),random=ACTIONS[random].record(),fantasy_statistics=rows,fantasy_stream=dict(root_seed=FANTASY_ROOT_SEED,source_cloud_order=128,counts=list(math.FANTASY_COUNTS),same_ancestors_and_noise_both_actions=True,same_outcomes_all_cloud_orders=True,prefix_budgets=True,ancestors=fantasies['ancestors'],pseudo_noise_sha256=hashlib.sha256(np.ascontiguousarray(fantasies['noise_m']).tobytes()).hexdigest()),random_root_seed=RANDOM_ROOT_SEED,prior_canonical_mean_coordinate_variance_m2=float(np.mean(reference['variance_m2'])),scope='Fixed4conditions x16noise repetitions, paireddevelopmentextension. No durations, energycost, perception, population calibration or adaptive-win claim. Canonical+1 latent target excludes sensor noise; actualSIDE observations are not inputs.')
 except Exception as error:result['failures'].append(repr(error))
 return result
