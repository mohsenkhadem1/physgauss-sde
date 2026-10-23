"""Typed C39 public probe payload and one new SIDE-noise stream only.

No field files, old DOWN outcomes, identity entropy, private labels or targets are
read here. The trusted producer calls noise only after all new choices seal.
Old PUBLIC prefix bytes/plannedFz remain unchanged legacy C38 observations.
"""
import numpy as np
from experiments import campaign_c38_controlled_observations as prefix
from experiments.campaign_c39_side_point_forward import ACTIONS,SIDE
DESIGN_PATH='runs/operator-dev/c39-controlled-extension-design-v01/design.json'
DESIGN_SHA256='3555d4ad8235efa72b1c606e626cbab262a1fb7c1729b4f79f6a9fd669f0e321'
PUBLIC_KEYS={'information_role','object_id','observation_kind','acquisition_index','action_id','force_vector_N','qualified_point_ids','reference_points_m','observed_positions_m','sensor_covariance_m2','reset_from_common_rest','is_RGBD_perception','is_temporal_forecast'}
ROLE='C39_PUBLIC_ACQUIRED_IDEAL_STATIC_PROBE'
PROBE_ACTIONS=('middle_down_2N',SIDE)


def side_measurement_error(private_configuration_index,repeat_index,*,rng_factory=np.random.default_rng):
 if type(private_configuration_index)is not int or not 0<=private_configuration_index<4 or type(repeat_index)is not int or not 0<=repeat_index<16:raise ValueError('Exact existing4config/16repeat indices, no redraw')
 seed=np.random.SeedSequence([2026102305,private_configuration_index,repeat_index,0])
 return np.asarray(rng_factory(seed).standard_normal((9,3)),dtype=np.float64)*.0002


def public_probe(public_prefix,action_id,observed_positions_m,*,design):
 prefix.validate_public_prefix(public_prefix,design=design)
 if action_id not in PROBE_ACTIONS:raise ValueError('TypedDOWN/SIDE known action only')
 y=np.asarray(observed_positions_m)
 if y.shape!=(9,3)or y.dtype!=np.float64 or not np.isfinite(y).all():raise ValueError('Exact finite27-coordinate acquisition')
 return dict(information_role=ROLE,object_id=public_prefix['object_id'],observation_kind='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD',acquisition_index=1,action_id=action_id,force_vector_N=np.array(ACTIONS[action_id]),qualified_point_ids=public_prefix['qualified_point_ids'].copy(),reference_points_m=public_prefix['reference_points_m'].copy(),observed_positions_m=y.copy(),sensor_covariance_m2=public_prefix['sensor_covariance_m2'].copy(),reset_from_common_rest=True,is_RGBD_perception=False,is_temporal_forecast=False)


def validate_public_probe(public_prefix,probe,*,design):
 prefix.validate_public_prefix(public_prefix,design=design)
 if not isinstance(probe,dict)or set(probe)!=PUBLIC_KEYS:raise ValueError('Exacttypedprobe allowlist; no labels/counterfactual/Fz alias')
 if probe['information_role']!=ROLE or probe['object_id']!=public_prefix['object_id']or probe['observation_kind']!='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD'or type(probe['acquisition_index'])is not int or probe['acquisition_index']!=1 or probe['action_id']not in PROBE_ACTIONS:raise ValueError('SameID/acq1/typed action/publicrole')
 if probe['reset_from_common_rest']is not True or probe['is_RGBD_perception']is not False or probe['is_temporal_forecast']is not False:raise ValueError('Staticreset/idealpoint flags')
 for key,shape,dtype in [('force_vector_N',(3,),np.float64),('qualified_point_ids',(9,),np.int64),('reference_points_m',(9,3),np.float64),('observed_positions_m',(9,3),np.float64),('sensor_covariance_m2',(27,27),np.float64)]:
  a=np.asarray(probe[key])
  if a.shape!=shape or a.dtype!=np.dtype(dtype)or not np.isfinite(a).all():raise ValueError('Exacttypedprobe shape/dtype '+key)
 if not np.array_equal(probe['force_vector_N'],np.array(ACTIONS[probe['action_id']])):raise ValueError('Typedaction/worldvector mismatch; SIDE is not scalarFz')
 for key in ('qualified_point_ids','reference_points_m','sensor_covariance_m2'):
  if not np.array_equal(probe[key],public_prefix[key]):raise ValueError('Samequalified9points/R200um')
 return probe


def encode(public_prefix,probe,*,design):
 validate_public_probe(public_prefix,probe,design=design);out={}
 for k,v in probe.items():
  if k in ('information_role','object_id','observation_kind','action_id'):out[k]=np.asarray(v)
  elif k in ('reset_from_common_rest','is_RGBD_perception','is_temporal_forecast'):out[k]=np.asarray(v,dtype=np.bool_)
  elif k=='acquisition_index':out[k]=np.asarray(v,dtype=np.int64)
  else:out[k]=np.asarray(v).copy()
 return out


def decode(public_prefix,members,*,design):
 if set(members)!=PUBLIC_KEYS:raise ValueError('Exacttyped publicNPZ allowlist')
 out={}
 for k,v in members.items():
  a=np.asarray(v)
  if k in ('information_role','object_id','observation_kind','action_id'):
   if a.shape!=()or a.dtype.kind!='U':raise ValueError('ScalarUnicode typedprobe')
   out[k]=str(a.item())
  elif k in ('reset_from_common_rest','is_RGBD_perception','is_temporal_forecast'):
   if a.shape!=()or a.dtype!=np.bool_:raise ValueError('Scalarbool typedprobe')
   out[k]=bool(a.item())
  elif k=='acquisition_index':
   if a.shape!=()or a.dtype!=np.int64:raise ValueError('Scalarint64 acquisition')
   out[k]=int(a.item())
  else:out[k]=a.copy()
 return validate_public_probe(public_prefix,out,design=design)
