"""C39 public-only forward/choice barriers; no observation generator or truth API.

Metadata/hash verification precedes candidate array extraction. PRIVATE_SIDE
archive is neither opened nor hashed here. Forward labels are hypotheses only.
"""
from pathlib import Path
import numpy as np
from experiments import run_campaign_c38_initial_inference_v03 as io
from experiments import run_campaign_c38_static_policy_v02 as initial_policy
from experiments import campaign_c39_side_point_forward as side
from experiments import campaign_c39_typed_static_policy as policy
DESIGN=policy.DESIGN_RECORD
POLICIES={'information','target_risk','fixed_down','fixed_side','random'}
LINEAR_KEYS={'information_role','support','cells','young_Pa','action_id','force_vector_N','material_probes_m','displacement63_m'}


def side_plan(root,validation_record):
 """Read SOURCE/qualification metadata only, before any model/response arrays."""
 io.checked(root,DESIGN);v=io.read_json(root,validation_record);check=v.get('validation',{})
 if v.get('completed')is not True or check.get('completed')is not True or check.get('finite_checks_passed')is not True or check.get('tolerance_m')!=1e-6 or check.get('continuous_material_certificate')is not False:raise ValueError('Completed finiteSIDEpoint qualification required')
 rows=check.get('checks',[]);expected={(c['id'],tuple(mesh))for c in side.side_catalog()if c['material_use']=='WITHHELD_INTERPOLATION_VALIDATION'for mesh in side.MESHES}
 if len(rows)!=8 or {(r.get('case_id'),tuple(r.get('cells',[])))for r in rows}!=expected:raise ValueError('All8 unique prospective withheldSIDE checks required')
 for r in rows:
  if r.get('passed')is not True or not np.isfinite(r.get('maximum_point_error_m',float('nan')))or not 0<=r['maximum_point_error_m']<=1e-6:raise ValueError('Fixed1um SIDEforward gate failed')
 vp=io.read_json(root,v['protocol'])
 if vp.get('schema')!='C39-SIDE-point-export-v01'or vp.get('status')!='FROZEN_BEFORE_Q63_EXTRACTION'or vp.get('design')!=DESIGN or vp.get('withheld_interpolation_tolerance_m')!=1e-6 or vp.get('no_observations_or_draws')is not True:raise ValueError('Exact frozen forward-export provenance')
 if not isinstance(vp.get('source_sha256'),dict)or not vp['source_sha256']:raise ValueError('Forward source closure required')
 for name,digest in vp['source_sha256'].items():io.checked(root,dict(path=name,sha256=digest))
 archives=v.get('archives',{})
 if set(archives)!={'CANDIDATE_SIDE','PRIVATE_WITHHELD_SIDE','LINEAR_SIDE'}:raise ValueError('Separate candidate/withheld/linear roles required')
 for key in ('CANDIDATE_SIDE','LINEAR_SIDE'):io.checked(root,archives[key])
 return dict(forward_point_validation=validation_record,candidate_side_archive=archives['CANDIDATE_SIDE'],linear_side_archive=archives['LINEAR_SIDE'],forward_export_protocol=v['protocol'])


def load_composites(root,initial_inputs,side_inputs):
 from experiments.campaign_c38_candidate_point_forward import CandidatePointForward,ARCHIVE_KEYS
 with np.load(io.checked(root,initial_inputs['candidate']),allow_pickle=False)as z:
  if set(z.files)!=ARCHIVE_KEYS:raise ValueError('Exact original candidate-only archive')
  original=CandidatePointForward({k:z[k].copy()for k in ARCHIVE_KEYS})
 linear=io.load_linear_library(root,initial_inputs['linear_library'])
 with np.load(io.checked(root,side_inputs['candidate_side_archive']),allow_pickle=False)as z:
  if set(z.files)!=side.ARCHIVE_KEYS:raise ValueError('OnlySIDE candidateTABLE archive')
  candidate_side=side.SidePointForward({k:z[k].copy()for k in side.ARCHIVE_KEYS})
 with np.load(io.checked(root,side_inputs['linear_side_archive']),allow_pickle=False)as z:
  if set(z.files)!=LINEAR_KEYS or str(z['information_role'])!='C39_SIDE_LINEAR_HYPOTHESES_NOT_OBSERVED_LABELS'or str(z['action_id'])!=side.SIDE or not np.array_equal(z['force_vector_N'],[0.,2.,0.])or not np.array_equal(z['material_probes_m'],side.POINTS):raise ValueError('Exact publicLINEAR SIDEhypotheses archive')
  if z['support'].shape!=(4,)or z['cells'].shape!=(4,3)or z['young_Pa'].shape!=(4,)or z['displacement63_m'].shape!=(4,63,3):raise ValueError('Exact4 linearSIDE rows')
  rows=[dict(support=str(z['support'][i]),cells=z['cells'][i].tolist(),young_Pa=float(z['young_Pa'][i]),action_id=side.SIDE,force_vector_N=[0.,2.,0.],material_probes_m=z['material_probes_m'].copy(),displacement63_m=z['displacement63_m'][i].copy())for i in range(4)]
 linear_side=side.LinearSidePointForward(rows)
 return dict(zip(io.MODELS,(side.CompositePointForward(original,candidate_side),side.CompositePointForward(linear,linear_side))))


def choices_from_result(result,object_id):
 numeric=result.get('completed')is True and result.get('numerical_cloud_controls_passed')is True
 random=int(np.random.default_rng(policy.math._seed(policy.RANDOM_ROOT_SEED,object_id)).integers(2));choices=dict(information='middle_down_2N',target_risk='middle_down_2N',fixed_down='middle_down_2N',fixed_side=side.SIDE,random=policy.ACTIONS[random].action_id)
 if numeric:
  for name in ('information','target_risk'):
   a=result[name]['selected_action'];catalog={x.action_id:x.record()for x in policy.ACTIONS}
   if a.get('action_id')not in catalog or a!=catalog[a['action_id']]:raise ValueError('Exact typed selectedaction/vector required')
   choices[name]=a['action_id']
 if set(choices)!=POLICIES or any(x not in ('middle_down_2N',side.SIDE)for x in choices.values()):raise ValueError('Onlytwo typed actions')
 return choices,numeric


def validate_choices(root,record,*,ids,initial_seal,side_inputs):
 s=io.read_json(root,record)
 if s.get('schema')!='C39-typed-static-policy-choices-seal-v01'or s.get('completed')is not True or s.get('count')!=64 or s.get('model_executions')!=128 or s.get('model')!='nonlinear_table'or s.get('extension_design')!=DESIGN or s.get('initial_prediction_seal')!=initial_seal:raise ValueError('Complete typed64choices/initial/design binding')
 for key in ('private_SIDE_outcomes_access','acquired_DOWN_observations_or_posteriors_access','private_target_or_latent_access'):
  if s.get(key)is not False:raise ValueError('Choice information barrier '+key)
 if s.get('all_choices_sealed_before_SIDE_noise')is not True or s.get('action_catalog')!=[a.record()for a in policy.ACTIONS]:raise ValueError('PreSIDEtyped choicecatalog')
 if s.get('forward_point_validation')!=side_inputs['forward_point_validation']or s.get('candidate_side_archive_sha256')!=side_inputs['candidate_side_archive']['sha256']or s.get('linear_side_archive_sha256')!=side_inputs['linear_side_archive']['sha256']:raise ValueError('Choiceforwardlibrary mismatch')
 rows=s.get('choices',[]);modelrows=s.get('model_choices',{})
 if len(ids)!=64 or len(set(ids))!=64 or [r.get('object_id')for r in rows]!=ids or set(modelrows)!=set(io.MODELS):raise ValueError('Exactall64 identity join')
 for name in io.MODELS:
  if [r.get('object_id')for r in modelrows[name]]!=ids:raise ValueError('Bothmodels64choices retained')
 for i,row in enumerate(rows):
  state=io.read_json(root,row['result'])
  if state.get('completed')is not True or state.get('object_id')!=row['object_id']or set(state.get('models',{}))!=set(io.MODELS)or any(state.get(k)is not False for k in ('private_SIDE_outcomes_access','acquired_DOWN_observations_or_posteriors_access','private_target_or_latent_access')):raise ValueError('Bound completed2modelpolicy result')
  for name in io.MODELS:
   entry=state['models'][name];r=io.read_json(root,entry['result']);choices,numeric=choices_from_result(r,row['object_id'])
   if entry.get('choices')!=choices or entry.get('numerical_controls_passed')!=numeric or modelrows[name][i].get('choices')!=choices or modelrows[name][i].get('result')!=entry['result']:raise ValueError('Choice/result/action recomputation mismatch')
  if row['choices']!=state['models']['nonlinear_table']['choices']:raise ValueError('Primarychoice mismatch')
 return s
