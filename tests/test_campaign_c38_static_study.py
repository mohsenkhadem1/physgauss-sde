"""Manufactured metadata/API fixtures only; no physical dataset is admitted."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import pytest
from experiments import campaign_c38_static_study as m
from experiments.campaign_static_p2_block import enrich
from experiments.campaign_c38_study import verify_admission as dynamic_admission


def record(path,**extra):return dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**extra)
def geometry():
    X,E=enrich(np.array([[0.,0,0],[.2,0,0],[0,.1,0],[0,0,.08]]),np.array([[0,1,2,3]]))
    return dict(rest_m=X,elements=E,tets=E[:,:4],faces=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]]))
def save(tmp,name,**data):
    p=tmp/name;np.savez(p,**data);return record(p)


def gate_fixture(tmp):
    source=tmp/m.SOURCE;source.parent.mkdir(parents=True);source.write_bytes(Path(m.__file__).read_bytes())
    physical=tmp/'physical.py';physical.write_text('# synthetic physical metadata source\n')
    evidence=[]
    for kind in sorted(m.EVIDENCE_KINDS):
        p=tmp/(kind+'.json');p.write_text('{"passed":true,"manufactured_metadata_only":true}')
        evidence.append(dict(kind=kind,**record(p)))
    domain='d'*64;g=geometry();a=save(tmp,'target.npz',**g,positions_m=g['rest_m']+.0001,constraint_mask=np.zeros_like(g['rest_m'],dtype=bool));a.update(object_id='objectA',load_state_index=7,domain_sha256=domain)
    review=tmp/'review.json';review.write_text(json.dumps(dict(schema='C38-static-domain-review-v01',approved_static_prediction=True,domain_sha256=domain,reviewed_evidence=evidence)))
    gate=dict(schema='C38-reviewed-static-admission-v01',task_kind=m.TASK,domain_sha256=domain,finite_screen_is_uniform_continuous_domain_certificate=False,source_sha256={m.SOURCE:hashlib.sha256(source.read_bytes()).hexdigest()},physical_source_sha256={'physical.py':hashlib.sha256(physical.read_bytes()).hexdigest()},evidence=evidence,review=record(review),admitted_static_fields=[a],temporal_qualified=False,contact_qualified=False,**{k:True for k in m.STATIC_FLAGS})
    manifest=dict(schema='C38-static-study-v01',task_kind=m.TASK,domain_sha256=domain)
    def commit(value):
        p=tmp/'gate.json';p.write_text(json.dumps(value));manifest['static_admission']=record(p)
    commit(gate);return gate,manifest,a,commit


def test_static_specific_flags_and_explicit_review_are_required_without_temporal_fabrication(tmp_path):
    gate,manifest,a,commit=gate_fixture(tmp_path)
    assert m.verify_static_admission(manifest,root=tmp_path)['temporal_qualified'] is False
    for name in m.STATIC_FLAGS:
        bad=copy.deepcopy(gate);bad[name]=False;commit(bad)
        with pytest.raises(ValueError,match='unqualified'):m.verify_static_admission(manifest,root=tmp_path)
    bad=copy.deepcopy(gate);bad['finite_screen_is_uniform_continuous_domain_certificate']=True;commit(bad)
    with pytest.raises(ValueError,match='uniform'):m.verify_static_admission(manifest,root=tmp_path)
    bad=copy.deepcopy(gate);bad['evidence']=[r for r in bad['evidence'] if r['kind']!='force_compiler'];commit(bad)
    with pytest.raises(ValueError,match='force'):m.verify_static_admission(manifest,root=tmp_path)
    bad=copy.deepcopy(gate);bad['review']={};commit(bad)
    with pytest.raises(ValueError):m.verify_static_admission(manifest,root=tmp_path)


def test_dynamic_temporal_gate_stays_distinct(tmp_path):
    gate,manifest,a,commit=gate_fixture(tmp_path)
    gate['qualified_for_prediction']=True;commit(gate)
    assert m.verify_static_admission(manifest,root=tmp_path)['qualified_for_static_prediction']
    dynamic={'forward_admission':manifest['static_admission'],'domain_sha256':manifest['domain_sha256']}
    with pytest.raises(ValueError,match='temporal'):dynamic_admission(dynamic,root=tmp_path)
    bad=dict(manifest,task_kind='causal_dynamic_forecast')
    with pytest.raises(ValueError,match='static manifest'):m.verify_static_admission(bad,root=tmp_path)


def test_exact_static_field_binding_before_private_arrays(tmp_path,monkeypatch):
    gate,manifest,a,commit=gate_fixture(tmp_path);result=m.load_admitted_static_targets(tmp_path,manifest,a)
    assert result.positions_m.shape==(10,3) and not result.positions_m.flags.writeable
    assert 'PRIVATE_STATIC' in result.information_role
    b=save(tmp_path,'unqualified.npz',**geometry(),positions_m=np.zeros((10,3)),constraint_mask=np.zeros((10,3),bool));b.update(object_id='objectA',load_state_index=7,domain_sha256='d'*64)
    monkeypatch.setattr(np,'load',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Private arrays opened before binding')))
    with pytest.raises(ValueError,match='not bound'):m.load_admitted_static_targets(tmp_path,manifest,b)
    with pytest.raises(ValueError,match='load/domain'):m.load_admitted_static_targets(tmp_path,manifest,dict(a,load_state_index=8))
    (tmp_path/'physical.py').write_text('changed')
    with pytest.raises(ValueError,match='Changed'):m.load_admitted_static_targets(tmp_path,manifest,a)


def test_static_target_rejects_clock_velocity_payload_even_with_valid_record(tmp_path):
    gate,manifest,a,commit=gate_fixture(tmp_path);g=geometry();r=save(tmp_path,'timed.npz',**g,positions_m=g['rest_m'],constraint_mask=np.zeros((10,3),bool),times_s=np.array([0.]))
    r.update(object_id='objectA',load_state_index=7,domain_sha256='d'*64);gate['admitted_static_fields']=[r];commit(gate)
    with pytest.raises(ValueError,match='allowlist'):m.load_admitted_static_targets(tmp_path,manifest,r)


def test_public_RGBD_adapter_reuses_exact_allowlist_and_never_accepts_private_context(tmp_path):
    g=geometry();gr=save(tmp_path,'geometry.npz',**g);controls=save(tmp_path,'controls.npz',load_state_indices=np.array([7]),load_vectors_N=np.zeros((1,10,3)))
    rgb=dict(acquisition_indices=np.array([0]),load_state_indices=np.array([7]),rgb=np.zeros((1,1,2,2,3)),depth_m=np.ones((1,1,2,2)),visible=np.ones((1,1,2,2),bool),camera_poses=np.eye(4)[None,None],intrinsics=np.eye(3)[None,None]);obs=save(tmp_path,'rgbd.npz',**rgb)
    p=m.public_static_rgbd_inputs(tmp_path,geometry_record=gr,controls_record=controls,observation_record=obs,last_acquisition=0)
    assert p['observation_kind']=='rgbd' and not p['is_temporal_forecast']
    assert not {'young_Pa','constraint_mask','positions_m','times_s'}&set(p)
    with pytest.raises(TypeError):m.public_static_rgbd_inputs(tmp_path,geometry_record=gr,controls_record=controls,observation_record=obs,last_acquisition=0,young_Pa=100000.)
    bad=save(tmp_path,'leaked.npz',**rgb,constraint_mask=np.ones((10,3),bool))
    with pytest.raises(ValueError,match='allowlist'):m.public_static_rgbd_inputs(tmp_path,geometry_record=gr,controls_record=controls,observation_record=bad,last_acquisition=0)


def test_ideal_points_are_not_RGBD_and_unacquired_point_values_not_accessed(tmp_path,monkeypatch):
    data=dict(acquisition_indices=np.array([0,1]),load_state_indices=np.array([7,8]),material_points_m=np.zeros((1,3)),observed_positions_m=np.ones((2,1,3)),sensor_covariance_m2=np.eye(3)*.0002**2)
    r=save(tmp_path,'points.npz',**data);p=m.load_public_ideal_static_points(tmp_path,r,last_acquisition=1,known_load_state_indices=[7,8])
    assert p['observation_kind']=='ideal_tracked_points' and p['is_RGBD_perception'] is False
    calls=[];original=np.lib.npyio.NpzFile.__getitem__
    def tracked(self,key):calls.append(key);return original(self,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',tracked)
    with pytest.raises(ValueError,match='Only acquired'):m.load_public_ideal_static_points(tmp_path,r,last_acquisition=0,known_load_state_indices=[7,8])
    assert calls==['acquisition_indices']


def catalog():
    rows=[];expected={}
    for i,role in enumerate(m.ROLES):
        obj=f'o{i}';ids=[obj+'prefix',obj+'query'];expected[obj]=ids
        rows.append(dict(object_id=obj,episode_id=f'e{i}',role=role,support_category='both_ends',young_Pa=80000.+i*1000,observation_kind='ideal_tracked_points',complete_acquisition_ids=ids,acquisition_ids=ids))
    return rows,expected


def test_repeated_known_support_categories_allowed_but_objects_materials_acquisitions_not_shared():
    rows,expected=catalog();r=m.validate_static_catalog(rows,expected_acquisitions_by_object=expected,expected_object_counts={r:1 for r in m.ROLES})
    assert r['known_support_categories_may_repeat'] and not r['unseen_support_layout_generalization']
    extra=copy.deepcopy(rows[0]);extra.update(episode_id='counterfactual',role=m.ROLES[1]);bad=rows+[extra]
    with pytest.raises(ValueError,match='Object/counterfactual'):m.validate_static_catalog(bad,expected_acquisitions_by_object=expected)
    bad=copy.deepcopy(rows);bad[1]['young_Pa']=bad[0]['young_Pa']
    with pytest.raises(ValueError,match='material draw'):m.validate_static_catalog(bad,expected_acquisitions_by_object=expected)
    bad=copy.deepcopy(rows);plans=copy.deepcopy(expected);plans['o1']=plans['o0'];bad[1]['complete_acquisition_ids']=plans['o0'];bad[1]['acquisition_ids']=plans['o0']
    with pytest.raises(ValueError,match='Acquisition ledger'):m.validate_static_catalog(bad,expected_acquisitions_by_object=plans)


def test_complete_ledger_and_separate_sensor_tracks_required():
    rows,expected=catalog();bad=copy.deepcopy(rows);bad[0]['acquisition_ids']=bad[0]['acquisition_ids'][:1]
    with pytest.raises(ValueError,match='All declared acquisitions'):m.validate_static_catalog(bad,expected_acquisitions_by_object=expected)
    with pytest.raises(ValueError,match='All declared objects'):m.validate_static_catalog(rows[:-1],expected_acquisitions_by_object=expected)
    bad=copy.deepcopy(rows);bad[0]['observation_kind']='rgbd'
    with pytest.raises(ValueError,match='separate declared'):m.validate_static_catalog(bad,expected_acquisitions_by_object=expected)
