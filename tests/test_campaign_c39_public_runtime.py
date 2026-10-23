"""Manufactured metadata/archives only; no actual C39 files or responses."""
import copy,json
import numpy as np
import pytest
from experiments import campaign_c39_public_runtime as m


def put(root,name,value=None,raw=None):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)if raw is not None else p.write_text(json.dumps(value));return m.io.record(root,p)


def fixture(root,monkeypatch):
 monkeypatch.setattr(m,'DESIGN',put(root,'design.json',dict(manufactured=True)))
 source=put(root,'source.py',raw=b'SOURCE FIXTURE')
 proto=put(root,'export.json',dict(schema='C39-SIDE-point-export-v01',status='FROZEN_BEFORE_Q63_EXTRACTION',design=m.DESIGN,withheld_interpolation_tolerance_m=1e-6,no_observations_or_draws=True,source_sha256={source['path']:source['sha256']}))
 checks=[dict(case_id=c['id'],cells=list(n),maximum_point_error_m=1e-8,passed=True)for c in m.side.side_catalog()if c['material_use']=='WITHHELD_INTERPOLATION_VALIDATION'for n in m.side.MESHES]
 v=dict(completed=True,protocol=proto,validation=dict(completed=True,finite_checks_passed=True,tolerance_m=1e-6,checks=checks,continuous_material_certificate=False),archives={k:put(root,k+'.npz',raw=b'NOT ARRAY')for k in ('CANDIDATE_SIDE','LINEAR_SIDE')})
 # PRIVATE file deliberately absent: no policy path may open/hash it.
 v['archives']['PRIVATE_WITHHELD_SIDE']=dict(path='PRIVATE/missing.npz',sha256='f'*64)
 return put(root,'validation.json',v),v


def test_forward_gate_all8_and_source_before_array_reads_no_private(tmp_path,monkeypatch):
 r,v=fixture(tmp_path,monkeypatch);monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('No arrays during gate')))
 p=m.side_plan(tmp_path,r);assert set(p)=={'forward_point_validation','candidate_side_archive','linear_side_archive','forward_export_protocol'}
 for bad in [dict(v,completed=False),dict(v,validation={**v['validation'],'checks':v['validation']['checks'][:-1]}),dict(v,validation={**v['validation'],'checks':[v['validation']['checks'][0]]*8})]:
  b=put(tmp_path,'bad.json',bad)
  with pytest.raises(ValueError):m.side_plan(tmp_path,b)
 bad=copy.deepcopy(v);bad['validation']['checks'][0]['maximum_point_error_m']=1.01e-6
 with pytest.raises(ValueError):m.side_plan(tmp_path,put(tmp_path,'bad.json',bad))
 (tmp_path/'source.py').write_text('changed')
 with pytest.raises(ValueError):m.side_plan(tmp_path,r)


def test_typed_choice_fallback_and_malformed_vector_is_not_silently_remapped():
 c,ok=m.choices_from_result(dict(completed=False),'a'*32);assert not ok and c['information']==c['target_risk']=='middle_down_2N'and c['fixed_side']=='middle_side_2N'
 r=dict(completed=True,numerical_cloud_controls_passed=True,information=dict(selected_action=m.policy.ACTIONS[1].record()),target_risk=dict(selected_action=m.policy.ACTIONS[0].record()))
 d,ok=m.choices_from_result(r,'a'*32);assert ok and d['information']=='middle_side_2N'and d['random']==c['random']
 bad=copy.deepcopy(r);bad['information']['selected_action']['force_world_N']=[0,0,2]
 with pytest.raises(ValueError):m.choices_from_result(bad,'a'*32)
