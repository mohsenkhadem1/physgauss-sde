"""Versioned SOURCE-ONLY producer for the frozen ideal static sensor design.

No file, identity entropy or noise is read/drawn on import. Functions below are
for a future reviewed execution AFTER forward and posterior-control gates. The
caller must keep the PRIVATE result away from the prediction/policy API. This
is trusted application separation, not an operating-system security boundary.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,re,secrets
import numpy as np
from experiments.campaign_c38_table_validation import POINTS
from experiments.campaign_c38_nonlinear_table_catalog import catalog

DESIGN_PATH='runs/operator-dev/c38-controlled-inverse-design-v01/design-v01.json'
DESIGN_SHA256='698f99008a588e0df4556e4de5b0e95cd19e15790cc89c9ab1cc24ed27877e85'
POINT_IDS=np.array([11,14,17,29,32,35,47,50,53],dtype=np.int64)
ROOT_SEED=2026102301
PUBLIC_KEYS={'information_role','object_id','observation_kind','acquisition_indices','prefix_force_Fz_N','planned_force_Fz_N','qualified_point_ids','reference_points_m','observed_positions_m','sensor_covariance_m2','known_body_gravity_m_s2','known_density_kg_m3','known_poisson_ratio','is_RGBD_perception','is_temporal_forecast'}


def load_frozen_design(root):
 path=Path(root)/DESIGN_PATH
 if hashlib.sha256(path.read_bytes()).hexdigest()!=DESIGN_SHA256:raise ValueError('Exact frozen sensor/prior design required')
 design=json.loads(path.read_text());validate_design(design);return design


def validate_design(d):
 sensor=d['sensor'];noise=sensor['noise'];a=d['acquisition']
 if d.get('schema')!='C38-controlled-static-ideal-inverse-DESIGN-v01' or sensor['kind']!='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD' or sensor['qualified_point_ids']!=POINT_IDS.tolist() or not np.array_equal(sensor['reference_points_m'],POINTS[POINT_IDS]):raise ValueError('Exact nine reviewed ideal reference points required')
 if noise['distribution']!='independent_zero_mean_Gaussian_per_Cartesian_coordinate_per_acquisition' or noise['sigma_m']!=.0002 or noise['known_stipulated_not_sensor_calibrated'] is not True or d['noise_repeats_per_configuration']!=16 or d['independent_noise_seeds']['root_seed']!=ROOT_SEED:raise ValueError('Frozen200um/16repeat/seed noise recipe required')
 if a['prefix_Fz_N']!=0. or a['candidate_probe_Fz_N']!=[-2.,2.] or a['target_Fz_N']!=1. or a['same_latent_across_resets'] is not True or a['each_probe_reset_from_common_rest'] is not True:raise ValueError('Exact static prefix/reset/probe/canonical-load design required')
 return d


def controlled_configurations():
 result=[]
 for c in catalog():
  key=(c['young_Pa'],c['support'])
  if c['material_use']=='WITHHELD_INTERPOLATION_VALIDATION' and key not in result:result.append(key)
 if len(result)!=4:raise ValueError('Exactly four prospectively withheld controlled configurations required')
 return tuple(result)


def mint_opaque_ids(count,*,token_hex=None):
 """Independent identity entropy only; no E, BC, response or index argument.

 Do NOT call before the eventual execution is reviewed. Persist minted tokens
 in the private execution mapping for deterministic recovery; never regenerate
 tokens to hide a failure. Public collections must be sorted by these opaque
 tokens, not by private configuration/repeat order.
 """
 if type(count) is not int or count<1:raise ValueError('Positive identity count required')
 generate=secrets.token_hex if token_hex is None else token_hex
 ids=[generate(16) for _ in range(count)]
 if len(set(ids))!=count or any(not isinstance(s,str) or re.fullmatch('[0-9a-f]{32}',s) is None for s in ids):raise ValueError('Independent unique opaque128-bit identity tokens required; no redraw')
 return ids


def _public_id(value):
 if not isinstance(value,str) or re.fullmatch('[0-9a-f]{32}',value) is None:raise ValueError('Opaque independently minted public object ID required')
 return value


def public_prefix_payload(public_id,observed_positions_m,*,design):
 """Exact B feature allowlist; accepts no latent label or future response."""
 validate_design(design);q=np.asarray(observed_positions_m,dtype=np.float64)
 if q.shape!=(9,3) or not np.isfinite(q).all():raise ValueError('Only the acquired nine ideal-point positions required')
 payload=dict(information_role='PUBLIC_ACQUIRED_IDEAL_STATIC_POINTS_B',object_id=_public_id(public_id),observation_kind='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD',acquisition_indices=np.array([0],dtype=np.int64),prefix_force_Fz_N=0.,planned_force_Fz_N=np.array([-2.,2.,1.]),qualified_point_ids=POINT_IDS.copy(),reference_points_m=POINTS[POINT_IDS].copy(),observed_positions_m=q[None].copy(),sensor_covariance_m2=np.eye(27)*.0002**2,known_body_gravity_m_s2=9.81,known_density_kg_m3=625.,known_poisson_ratio=.35,is_RGBD_perception=False,is_temporal_forecast=False)
 validate_public_prefix(payload,design=design)
 for v in payload.values():
  if isinstance(v,np.ndarray):v.flags.writeable=False
 return payload


def validate_public_prefix(payload,*,design):
 validate_design(design)
 if not isinstance(payload,dict) or set(payload)!=PUBLIC_KEYS:raise ValueError('Exact public prefix allowlist; no E/BC/private-index/future arrays')
 _public_id(payload['object_id'])
 if payload['information_role']!='PUBLIC_ACQUIRED_IDEAL_STATIC_POINTS_B' or payload['observation_kind']!='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD' or payload['is_RGBD_perception'] is not False or payload['is_temporal_forecast'] is not False:raise ValueError('Ideal static observation role required')
 if not np.array_equal(payload['acquisition_indices'],[0]) or payload['prefix_force_Fz_N']!=0. or not np.array_equal(payload['planned_force_Fz_N'],[-2.,2.,1.]) or not np.array_equal(payload['qualified_point_ids'],POINT_IDS) or not np.array_equal(payload['reference_points_m'],POINTS[POINT_IDS]):raise ValueError('Exact acquired/static planned-force/point scope required')
 q=np.asarray(payload['observed_positions_m']);C=np.asarray(payload['sensor_covariance_m2'])
 if q.shape!=(1,9,3) or not np.isfinite(q).all() or not np.array_equal(C,np.eye(27)*.0002**2) or payload['known_body_gravity_m_s2']!=9.81 or payload['known_density_kg_m3']!=625. or payload['known_poisson_ratio']!=.35:raise ValueError('Exact ideal sensor covariance and known physics required')
 return payload


def measurement_error(private_configuration_index,repeat_index,channel_index,*,rng_factory=None):
 """Frozen independent actual stream, separate from policy-design fantasies.

 Production calls create Gaussian draws and are NOT authorized by importing
 this source. Tests inject fake generators and do not consume production noise.
 """
 for value,bound in ((private_configuration_index,4),(repeat_index,16),(channel_index,3)):
  if type(value) is not int or not 0<=value<bound:raise ValueError('Exact4configuration/16repeat/3channel indices required')
 factory=np.random.default_rng if rng_factory is None else rng_factory
 seed=np.random.SeedSequence([ROOT_SEED,private_configuration_index,repeat_index,channel_index])
 z=np.asarray(factory(seed).standard_normal((9,3)),dtype=np.float64)
 if z.shape!=(9,3) or not np.isfinite(z).all():raise ValueError('Finite nine-point independent Gaussian error realization required')
 return .0002*z


@dataclass(frozen=True)
class BoundPrivateResponses:
 private_configuration_index:int
 responses_by_Fz_N:dict
 source_records:dict
 information_role:str='PRIVATE_QUALIFIED_Q63_GENERATOR_INPUT_NOT_PUBLIC_FEATURES'


def load_qualified_private_responses(root,archive_record,validation_record,posterior_controls_record,*,private_configuration_index):
 """Future lightweight source/hash/point-scope guard; never called on import.

 Runtime-protocol review is still required. A table archive by itself is not
 admission, and the separate numerical posterior-control receipt must match
 THIS frozen sensor/prior design. This loader accesses only PRIVATE q63 data.
 """
 from experiments.prepare_campaign_c38_point_table import checked
 root=Path(root).resolve();validation=json.loads(checked(validation_record,root).read_text());controls=json.loads(checked(posterior_controls_record,root).read_text())
 if validation.get('completed') is not True or validation.get('validation',{}).get('finite_checks_passed') is not True or validation['validation'].get('tolerance_m')!=1e-6 or validation.get('archives',{}).get('WITHHELD_INTERPOLATION_VALIDATION')!=archive_record:raise ValueError('Qualified exact point-archive/1um validation binding required')
 if controls.get('completed') is not True or controls.get('all_numerical_controls_passed') is not True or controls.get('design_sha256')!=DESIGN_SHA256:raise ValueError('Separate specialised posterior controls for this design required before data/noise')
 configs=controlled_configurations()
 if type(private_configuration_index) is not int or not 0<=private_configuration_index<4:raise ValueError('One of four frozen withheld configurations required')
 keys={'source_index','case_ids','material_use','information_role','cells','support','young_Pa','Fz_N','material_probes_m','probe_positions_m'}
 with np.load(checked(archive_record,root),allow_pickle=False) as z:
  if set(z.files)!=keys:raise ValueError('Exact PRIVATE direct-point archive allowlist required')
  if str(z['information_role'])!='PRIVATE_DIRECT_WITHHELD_INTERPOLATION_VALIDATION' or not np.array_equal(z['material_probes_m'],POINTS) or z['probe_positions_m'].shape!=(32,63,3):raise ValueError('Private exact63point archive scope required')
  data={k:z[k].copy() for k in keys}
 expected={(c['id'],nx):c for c in catalog() if c['material_use']=='WITHHELD_INTERPOLATION_VALIDATION' for nx in (32,40)};seen=set()
 for i in range(32):
  cell=str(data['case_ids'][i]);cells=data['cells'][i];key=(cell,int(cells[0]));c=expected.get(key)
  if c is None or key in seen or not np.array_equal(cells,[key[1],key[1]//2,key[1]//2]) or data['material_use'][i]!='WITHHELD_INTERPOLATION_VALIDATION' or data['young_Pa'][i]!=c['young_Pa'] or data['support'][i]!=c['support'] or data['Fz_N'][i]!=c['Fz_N']:raise ValueError('Exact withheld configuration/force/source-index identity required')
  expected_index=2*next(j for j,v in enumerate(catalog()) if v['id']==cell)+(0 if key[1]==32 else 1)
  if data['source_index'][i]!=expected_index:raise ValueError('Private source-index changed')
  seen.add(key)
 if seen!=set(expected) or not np.isfinite(data['probe_positions_m']).all():raise ValueError('All32 direct withheld point fields required')
 E,bc=configs[private_configuration_index];selection=(data['young_Pa']==E)&(data['support']==bc)&(data['cells'][:,0]==40)
 responses={float(data['Fz_N'][i]):data['probe_positions_m'][i].copy() for i in np.flatnonzero(selection)}
 if len(responses)!=4:raise ValueError('Four exact same-object finest force states required')
 return BoundPrivateResponses(private_configuration_index,responses,dict(archive=archive_record,validation=validation_record,posterior_controls=posterior_controls_record,design_sha256=DESIGN_SHA256))


def private_repeat(private_configuration_index,repeat_index,public_id,responses_by_Fz_N,*,design,rng_factory=None):
 """PRIVATE generator returns (public acquired prefix, PRIVATE future bundle).

 ``responses_by_Fz_N`` must come from a separately admitted exact63-point
 direct-withheld archive for the same physical configuration/finest mesh. The
 future execution driver must prove that binding plus qualified candidate
 forward and specialised posterior controls BEFORE invoking this producer.
 Neither source import nor the generic table archive grants observation draws.
 Each action observation is generated once and reused by every policy; a policy
 argument is intentionally absent. Future data must be released only after the
 appropriate prefix/choice/prediction seals in the caller's evaluator.
 """
 validate_design(design);configs=controlled_configurations()
 if type(private_configuration_index) is not int or not 0<=private_configuration_index<4 or type(repeat_index) is not int or not 0<=repeat_index<16:raise ValueError('Frozen controlled configuration/repeat required')
 if not isinstance(responses_by_Fz_N,BoundPrivateResponses) or responses_by_Fz_N.private_configuration_index!=private_configuration_index:raise ValueError('Hash/source/point-scope bound private configuration required before noise')
 bound=responses_by_Fz_N;responses_by_Fz_N=bound.responses_by_Fz_N
 if set(responses_by_Fz_N)!={-2.,0.,1.,2.}:raise ValueError('Exact same-object known gravity/down/up/canonical-load responses required')
 q={F:np.asarray(v,dtype=np.float64) for F,v in responses_by_Fz_N.items()}
 if any(v.shape!=(63,3) or not np.isfinite(v).all() for v in q.values()):raise ValueError('Exact finite qualified63-point responses only; no full-node fields')
 prefix=q[0.][POINT_IDS]+measurement_error(private_configuration_index,repeat_index,0,rng_factory=rng_factory)
 public=public_prefix_payload(public_id,prefix,design=design)
 actions=np.stack([q[F][POINT_IDS]+measurement_error(private_configuration_index,repeat_index,channel,rng_factory=rng_factory) for F,channel in ((-2.,1),(2.,2))])
 E,support=configs[private_configuration_index]
 private=dict(source_records=bound.source_records,information_role='PRIVATE_GENERATOR_EVALUATOR_FUTURE_AND_LATENT_NOT_MODEL_FEATURES',object_id=_public_id(public_id),private_configuration_index=private_configuration_index,repeat_index=repeat_index,young_Pa=E,support_category=support,candidate_probe_Fz_N=np.array([-2.,2.]),candidate_probe_observations_m=actions,canonical_target_Fz_N=1.,canonical_target_point_ids=np.arange(63,dtype=np.int64),canonical_target_positions_m=q[1.].copy(),noise_seed_recipe=[ROOT_SEED,private_configuration_index,repeat_index],common_actual_action_observations_across_policies=True,same_latent_reset_from_rest=True,independent_population_object=False)
 return public,private


def order_public_prefixes(payloads,*,design):
 """Whole64 noise-repeat records, ordered by opaque IDs, never private labels."""
 if not isinstance(payloads,list) or len(payloads)!=64:raise ValueError('All64 controlled noise-repeat prefixes required; no filtering')
 for p in payloads:validate_public_prefix(p,design=design)
 if len({p['object_id'] for p in payloads})!=64:raise ValueError('Unique independently minted IDs required')
 return sorted(payloads,key=lambda p:p['object_id'])
