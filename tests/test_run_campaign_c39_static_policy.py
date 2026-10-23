import json,copy
import numpy as np
import pytest
from experiments import run_campaign_c39_static_policy as m
from test_campaign_c38_public_prefix_inference_v02 import design,payload


def test_full64_two_model_metadata_seal_and_retained_numeric_fallback(tmp_path,monkeypatch):
 ids=[format(i+1,'032x')for i in range(64)];objects=[dict(object_id=i,prefix={})for i in ids];side=dict(forward_point_validation={'path':'val','sha256':'a'*64},candidate_side_archive={'path':'c','sha256':'b'*64},linear_side_archive={'path':'l','sha256':'c'*64});p=dict(output='out',objects=objects,side_inputs=side,source_sha256={'fake':'d'*64});initial=dict(path='initial',sha256='e'*64)
 monkeypatch.setattr(m,'validate',lambda*a,**k:(p,{}));monkeypatch.setattr(m.base,'guard',lambda*a:None);monkeypatch.setattr(m.base,'initial_barrier',lambda*a:(initial,{}));monkeypatch.setattr(m.common,'load_composites',lambda*a:dict(zip(m.MODELS,(object(),object()))));monkeypatch.setattr(m.base.obs,'load_frozen_design',lambda*a:design())
 # Prefix identity is supplied by an exact row hash binding, never actual I/O.
 for row in objects:row['prefix']={'id':row['object_id']}
 monkeypatch.setattr(m.base.codec,'load_public_prefix',lambda root,r,**k:payload(r['id']))
 calls=[]
 def policy(model,public,*,design):
  calls.append(public['object_id']);ok=len(calls)!=1
  return dict(completed=ok,numerical_cloud_controls_passed=ok,information=dict(selected_action=m.policy.ACTIONS[1].record()),target_risk=dict(selected_action=m.policy.ACTIONS[0].record()),failures=[]if ok else['manufactured cloudfailure'])
 monkeypatch.setattr(m.policy,'design_typed_policy',policy);monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('No arrays')))
 out=tmp_path/'out';out.mkdir();(out/'objects').mkdir();m.worker('p','f'*64,root=tmp_path);s=json.loads((out/'choices-seal.json').read_text())
 assert s['count']==64 and s['model_executions']==128 and len(calls)==128 and len(s['model_choices'])==2
 assert not s['all_policy_numerical_controls_passed']and s['choices'][0]['choices']['information']=='middle_down_2N'and s['choices'][1]['choices']['information']=='middle_side_2N'
 assert s['candidate_side_archive_sha256']==side['candidate_side_archive']['sha256']and s['forward_point_validation']==side['forward_point_validation']
 assert not s['private_SIDE_outcomes_access']and not s['acquired_DOWN_observations_or_posteriors_access']and s['all_choices_sealed_before_SIDE_noise']
 bad=copy.deepcopy(s);bad['candidate_side_archive_sha256']='0'*64;q=out/'bad.json';q.write_text(json.dumps(bad))
 with pytest.raises(ValueError):m.common.validate_choices(tmp_path,m.io.record(tmp_path,q),ids=[r['object_id']for r in objects],initial_seal=initial,side_inputs=side)


def test_policy_has_bounded_owned_cpu2_stage_no_gpu():
 assert m.RESOURCES['global_wall_s']==300 and m.RESOURCES['cores']==[30,31]and m.RESOURCES['RSS_bytes']==2*1024**3
 assert m.RESOURCES['no_CUDA']and m.RESOURCES['disk_free_reserve_bytes']==30*1024**3
