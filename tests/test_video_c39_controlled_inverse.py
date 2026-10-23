"""Manufactured static fields and pure gates only; no actual C39 values."""
import copy
import numpy as np
import pytest
from experiments import video_c39_controlled_inverse as v


def fixtures():
 ids=[format(i,'032x')for i in range(64)];I=v.INPUTS['initial'];C=v.INPUTS['choices'];P=v.INPUTS['post'];R=v.INPUTS['reference'];action=dict(information=v.full.metrics.SIDE,target_risk=v.full.metrics.SIDE,fixed_down=v.full.metrics.DOWN,fixed_side=v.full.metrics.SIDE,random=v.full.metrics.DOWN)
 rows=[dict(object_id=i,choices=action.copy())for i in ids]
 initial=dict(completed=True,count=64,model_executions=128,private_truth_access=False,objects=[dict(object_id=i)for i in ids],protocol_sha256=v.PROTOCOLS['initial']['sha256'])
 choices=dict(schema='C39-typed-static-policy-choices-seal-v01',completed=True,count=64,model_executions=128,extension_design=v.full.DESIGN,initial_prediction_seal=I,all_choices_sealed_before_SIDE_noise=True,private_SIDE_outcomes_access=False,acquired_DOWN_observations_or_posteriors_access=False,private_target_or_latent_access=False,choices=copy.deepcopy(rows),model_choices={m:copy.deepcopy(rows)for m in v.io.MODELS},protocol=v.PROTOCOLS['choices'])
 post=dict(schema='C39-public-SIDE-postprobe-prediction-seal-v01',completed=True,count=64,model_executions=128,private_target_or_latent_access=False,initial_prediction_seal=I,choices_seal=C,all_postpredictions_sealed_before_private_evaluation=True,objects=[dict(object_id=i,action_id=v.full.metrics.SIDE)for i in ids],protocol=v.PROTOCOLS['post'])
 summary=dict(completed=True,all_SIDE_predictions_sealed_before_SIDE_dependent_scoring=True,design=v.full.DESIGN,choices_seal=C,SIDE_prediction_seal=P,frozen_reference=R)
 return initial,choices,post,summary,I,C,P,R


def test_complete_both_model_typed_seals_and_eval_before_values(monkeypatch):
 monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('No arrays in puregates')))
 i,c,p,s,I,C,P,R=fixtures();assert len(v.validate_gate(i,c,p,s,initial_record=I,choices_record=C,post_record=P,reference_record=R))==64
 for key,value in [('completed',False),('count',63),('model_executions',127),('private_target_or_latent_access',True)]:
  with pytest.raises(ValueError):v.validate_gate(i,c,dict(p,**{key:value}),s,initial_record=I,choices_record=C,post_record=P,reference_record=R)
 bad=copy.deepcopy(c);bad['model_choices']['linear_rest_tangent'].pop()
 with pytest.raises(ValueError):v.validate_gate(i,bad,p,s,initial_record=I,choices_record=C,post_record=P,reference_record=R)
 for bad in [dict(s,completed=False),dict(s,SIDE_prediction_seal={'path':'other','sha256':'f'*64})]:
  with pytest.raises(ValueError):v.validate_gate(i,c,p,bad,initial_record=I,choices_record=C,post_record=P,reference_record=R)


def test_missing_last_saved_bundle_blocks_summary_selection_and_targets(tmp_path,monkeypatch):
 i,c,p,s,I,C,P,R=fixtures();recipe=dict(path='recipe',sha256='e'*64);seen=[]
 monkeypatch.setattr(v,'sources',lambda:[]);monkeypatch.setattr(v.io,'runtime',lambda:{});monkeypatch.setattr(v,'verify_protocol_sources',lambda*r:None);monkeypatch.setattr(v,'guard',lambda*r:None);monkeypatch.setattr(v.full,'source_records',lambda*r:{})
 def read(root,r):
  seen.append(r['path'])
  if r==recipe:return dict(recipe=v.RECIPE,inputs=v.INPUTS,protocols=v.PROTOCOLS,source_sha256={},runtime={})
  if r==I:return i
  if r==C:return c
  if r==P:return p
  if r==R:return dict(schema='C39-fixed-metric-reference-v01',design=v.full.DESIGN,source_sha256={},initial_prediction_seal=I)
  raise AssertionError('Premature summary/privatevalue read')
 monkeypatch.setattr(v.io,'read_json',read);monkeypatch.setattr(v.full,'barrier',lambda*a:(_ for _ in()).throw(ValueError('MissingLASTSIDEhash')));monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('No arrayreads')))
 with pytest.raises(ValueError,match='MissingLAST'):v.render(recipe,root=tmp_path)
 assert seen==['recipe',I['path'],C['path'],P['path'],R['path']]


def test_same_four_configs_repeat_zero_no_error_selection():
 rows=[dict(object_id=f'{c}-{r}',private_configuration_index=c,repeat_index=r,error=999 if r==0 else 0)for c in range(4)for r in range(16)]
 assert [r['object_id']for r in v.select_records(rows[::-1])]==['0-0','1-0','2-0','3-0']
 with pytest.raises(ValueError):v.select_records(rows[:-1])
 assert v.RECIPE['geometry_scale']==1. and v.RECIPE['motion']=='STATIC_FIELDS_CAMERA_MOTION_ONLY'and not v.RECIPE['surface_claim']


def test_typed_DOWN_SIDE_labels_true_scale_z_profiles_and_interval_limit():
 assert v.action_label('middle_side_2N')=='SIDE +2 N along y'and v.action_label('middle_down_2N')=='DOWN -2 N along z'
 for bad in (2.,-2.,'middle_up_2N'):
  with pytest.raises(ValueError):v.action_label(bad)
 a=42.;d=np.array([np.cos(np.deg2rad(a)),-np.sin(np.deg2rad(a)),0.])*.001
 np.testing.assert_allclose(v.project(v.POINTS+d,a,origin_pixels=(0,0))-v.project(v.POINTS,a,origin_pixels=(0,0)),np.tile([1.5,0.],(63,1)),atol=1e-13)
 assert v.lattice_edges().shape==(138,2)and '10nm'in v.RECIPE['interval_scope']
 q=v.POINTS.copy();mean=q.copy();q[:,2]+=.0002;mean[:,2]+=.0001;interval=np.repeat(mean[v.POINT_IDS,:,None],3,axis=2);interval[:,2,0]-=.00005;interval[:,2,2]+=.00005
 p=v.profile_values(q,mean,interval);np.testing.assert_allclose(p['actual_z_um'],200.,atol=1e-10);np.testing.assert_allclose(p['lower_z_um'],50.,atol=1e-10)


def test_manufactured_both_action_frames_and_hash_guard(tmp_path):
 q=v.POINTS.copy();mean=q.copy();mean[:,2]-=.0001;interval=np.repeat(mean[v.POINT_IDS,:,None],3,axis=2);interval[:,2,0]-=.0002;interval[:,2,2]+=.0002
 for action in ('middle_down_2N','middle_side_2N'):
  image=v.frame(dict(configuration=0,action_id=action,actual=q,before=dict(mean=mean,intervals=interval),after=dict(mean=mean,intervals=interval)),40.);assert image.size==(1536,960)and image.mode=='RGB'
 p=tmp_path/'fixture';p.write_text('source');r=v.io.record(tmp_path,p);p.write_text('changed')
 with pytest.raises(ValueError):v.io.checked(tmp_path,r)
