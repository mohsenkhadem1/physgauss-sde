import copy,os
import numpy as np
import pytest
from experiments import video_c38_controlled_inverse as v


def fixtures():
 ids=[format(i,'032x') for i in range(64)];I=dict(path='initial.json',sha256='a'*64);C=dict(path='choices.json',sha256='b'*64);P=dict(path='post.json',sha256='c'*64)
 initial=dict(completed=True,count=64,model_executions=128,private_truth_access=False,objects=[dict(object_id=i) for i in ids]);choices=dict(completed=True,count=64,private_probe_outcomes_access=False,initial_prediction_seal=I,choices=[dict(object_id=i,choices=dict(information=-2.,target_risk=-2.,fixed_down=-2.,fixed_up=2.,random=2.))for i in ids]);post=dict(completed=True,count=128,model_executions=256,private_target_or_latent_access=False,initial_prediction_seal=I,choices_seal=C,objects=[dict(object_id=i,force_Fz_N=f)for i in ids for f in [-2.,2.]])
 summary=dict(completed=True,all_initial_choices_and_postpredictions_sealed_before_private_target_or_label_access=True,initial_prediction_seal=I,choices_seal=C,postprobe_prediction_seal=P)
 return initial,choices,post,summary,I,C,P


def test_full_postseal_and_eval_gate_before_selection_or_arrays(monkeypatch):
 monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No arrays in metadata gates')))
 a,c,p,s,I,C,P=fixtures();assert len(v.validate_gate(a,c,p,s,initial_record=I,choices_record=C,post_record=P)[0])==64
 for field,value in [('completed',False),('count',127),('model_executions',255),('private_target_or_latent_access',True)]:
  with pytest.raises(ValueError):v.validate_gate(a,c,dict(p,**{field:value}),s,initial_record=I,choices_record=C,post_record=P)
 with pytest.raises(ValueError):v.validate_gate(a,c,p,dict(s,completed=False),initial_record=I,choices_record=C,post_record=P)
 with pytest.raises(ValueError):v.validate_gate(a,c,p,s,initial_record=dict(I,sha256='f'*64),choices_record=C,post_record=P)


def test_incomplete_public_gate_never_opens_summary_or_private_values(monkeypatch,tmp_path):
 a,c,p,s,I,C,P=fixtures();a['completed']=False;R=dict(path='recipe.json',sha256='d'*64);S=dict(path='summary.json',sha256='e'*64)
 monkeypatch.setattr(v,'sources',lambda:[]);monkeypatch.setattr(v.io,'runtime',lambda:{});monkeypatch.setattr(os,'sched_getaffinity',lambda pid:{30,31});monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
 seen=[]
 def read(root,r):
  seen.append(r['path'])
  if r==R:return dict(recipe=v.RECIPE,source_sha256={},runtime={})
  if r==I:return a
  if r==C:return c
  if r==P:return p
  raise AssertionError('Premature private summary/data read')
 monkeypatch.setattr(v.io,'read_json',read)
 with pytest.raises(ValueError):v.render(R,I,C,P,S,dict(path='private-execution.json',sha256='f'*64),root=tmp_path)
 assert seen==['recipe.json','initial.json','choices.json','post.json']


def test_deterministic_four_cases_not_bestcase_filter():
 rows=[dict(object_id=f'{c}-{r}',private_configuration_index=c,repeat_index=r,model_error=10000 if r==0 else 0)for c in range(4)for r in range(16)]
 selected=v.select_records(rows[::-1]);assert [r['object_id']for r in selected]==['0-0','1-0','2-0','3-0']
 with pytest.raises(ValueError):v.select_records(rows[:-1])
 assert v.RECIPE['geometry_scale']==1. and v.RECIPE['motion']=='STATIC_FIELDS_CAMERA_MOTION_ONLY' and not v.RECIPE['surface_claim']


def test_true_world_scale_lattice_and_explicit_profile_units():
 a=42.;d=np.array([np.cos(np.deg2rad(a)),-np.sin(np.deg2rad(a)),0.])*.001
 first=v.project(v.POINTS,a,origin_pixels=(0,0));second=v.project(v.POINTS+d,a,origin_pixels=(0,0));np.testing.assert_allclose(second-first,np.tile([1.5,0.],(63,1)),atol=1e-13)
 edges=v.lattice_edges();assert edges.shape==(138,2);assert len(set(map(tuple,edges)))==138
 q=v.POINTS.copy();mean=q.copy();q[:,2]+=.0002;mean[:,2]+=.0001;interval=np.repeat(mean[v.POINT_IDS,:,None],3,axis=2);interval[:,2,0]-=.00005;interval[:,2,2]+=.00005
 p=v.profile_values(q,mean,interval)
 np.testing.assert_allclose(p['actual_z_um'],200.,atol=1e-10);np.testing.assert_allclose(p['mean_z_um'],100.,atol=1e-10);np.testing.assert_allclose(p['lower_z_um'],50.,atol=1e-10);np.testing.assert_allclose(p['upper_z_um'],150.,atol=1e-10)


def test_manufactured_preview_and_hash_guard(tmp_path):
 q=v.POINTS.copy();mean=q.copy();mean[:,2]-=.0001;interval=np.repeat(mean[v.POINT_IDS,:,None],3,axis=2);interval[:,2,0]-=.0002;interval[:,2,2]+=.0002
 c=dict(configuration=0,force=-2.,actual=q,before=dict(mean=mean,intervals=interval),after=dict(mean=mean,intervals=interval));im=v.frame(c,40.)
 assert im.size==(1536,960) and im.mode=='RGB'
 p=tmp_path/'fixture';p.write_text('source');r=v.io.record(tmp_path,p);assert v.io.checked(tmp_path,r)==p;p.write_text('changed')
 with pytest.raises(ValueError):v.io.checked(tmp_path,r)
