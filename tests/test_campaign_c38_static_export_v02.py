"""Manufactured metadata controls only; no actual physical scope admitted."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import pytest
from test_campaign_c38_static_export import fixture,compare_public
from experiments import campaign_c38_static_export_v02 as m


def setup_scope(tmp):
    gate,manifest,args,commit=fixture(tmp)
    source=tmp/m.SOURCE;source.parent.mkdir(exist_ok=True,parents=True);source.write_bytes(Path(m.__file__).read_bytes())
    gate['source_sha256'][m.SOURCE]=hashlib.sha256(source.read_bytes()).hexdigest()
    geometry=m.load_public_geometry(tmp,args['geometry_record'])
    _,_,X=m.original._points(geometry,args['element_indices'],args['barycentric'])
    p=tmp/'qualified_points.npz';np.savez(p,material_points_m=X)
    scope=dict(kind='finite_reference_material_points_only',points={'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()},spatial_evidence=[r for r in gate['evidence'] if r['kind']=='spatial_mesh'],admitted_fields=[{k:r[k] for k in ('path','sha256')} for r in gate['admitted_static_fields']])
    gate['material_point_scope']=scope
    def approve():
        review_path=tmp/'review.json';review=json.loads(review_path.read_text());review.update(approved_material_point_scope=True,reviewed_material_point_scope=copy.deepcopy(gate['material_point_scope']));review_path.write_text(json.dumps(review));gate['review']['sha256']=hashlib.sha256(review_path.read_bytes()).hexdigest();commit(gate)
    approve();return gate,manifest,args,commit,approve


def test_finite_scope_subset_matches_old_public_payload_and_roundoff_only(tmp_path):
    gate,manifest,args,_,_=setup_scope(tmp_path)
    old,_=m.original.export_ideal_static_prefix(tmp_path,manifest,**args)
    public,private=m.export_reviewed_ideal_static_prefix(tmp_path,manifest,qualified_point_indices=np.array([0,1]),**args)
    compare_public(old,public)
    assert private['qualified_point_indices']==[0,1]
    assert private['coordinate_roundoff_tolerance_m']<1e-14
    assert 'no whole-field' in private['spatial_scope']


def test_arbitrary_new_point_rejected_before_private_q_read(tmp_path,monkeypatch):
    _,manifest,args,_,_=setup_scope(tmp_path)
    args['barycentric'][0]=[.2,.3,.25,.25]
    monkeypatch.setattr(m.original,'export_ideal_static_prefix',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('private export entered')))
    with pytest.raises(ValueError,match='outside reviewed'):m.export_reviewed_ideal_static_prefix(tmp_path,manifest,qualified_point_indices=np.array([0,1]),**args)


def test_scope_must_be_explicit_and_reviewed_not_just_in_gate(tmp_path,monkeypatch):
    gate,manifest,args,commit,_=setup_scope(tmp_path)
    monkeypatch.setattr(m.original,'export_ideal_static_prefix',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('private export entered')))
    gate['material_point_scope']['kind']='whole_field';commit(gate)
    with pytest.raises(ValueError,match='finite material-point'):m.export_reviewed_ideal_static_prefix(tmp_path,manifest,qualified_point_indices=np.array([0,1]),**args)
    gate['material_point_scope']['kind']='finite_reference_material_points_only';gate['material_point_scope']['points']['sha256']='0'*64;commit(gate)
    with pytest.raises(ValueError,match='independently reviewed'):m.export_reviewed_ideal_static_prefix(tmp_path,manifest,qualified_point_indices=np.array([0,1]),**args)


def test_scope_needs_actual_spatial_evidence_and_correct_qualified_indices(tmp_path):
    gate,manifest,args,commit,approve=setup_scope(tmp_path)
    with pytest.raises(ValueError,match='outside reviewed'):m.export_reviewed_ideal_static_prefix(tmp_path,manifest,qualified_point_indices=np.array([1,0]),**args)
    gate['material_point_scope']['spatial_evidence']=[r for r in gate['evidence'] if r['kind']=='force_compiler'];approve()
    with pytest.raises(ValueError,match='spatial-mesh'):m.export_reviewed_ideal_static_prefix(tmp_path,manifest,qualified_point_indices=np.array([0,1]),**args)


def test_source_pin_required_and_qualified_scope_archive_tamper_rejected(tmp_path):
    gate,manifest,args,commit,_=setup_scope(tmp_path)
    del gate['source_sha256'][m.SOURCE];commit(gate)
    with pytest.raises(ValueError,match='source must be pinned'):m.export_reviewed_ideal_static_prefix(tmp_path,manifest,qualified_point_indices=np.array([0,1]),**args)
    gate['source_sha256'][m.SOURCE]=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest();commit(gate)
    (tmp_path/'qualified_points.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed'):m.export_reviewed_ideal_static_prefix(tmp_path,manifest,qualified_point_indices=np.array([0,1]),**args)


def test_point_scope_cannot_be_reused_for_another_admitted_field(tmp_path,monkeypatch):
    gate,manifest,args,_,approve=setup_scope(tmp_path)
    gate['material_point_scope']['admitted_fields']=gate['material_point_scope']['admitted_fields'][1:];approve()
    monkeypatch.setattr(m.original,'export_ideal_static_prefix',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('private export entered')))
    with pytest.raises(ValueError,match='lacks reviewed'):m.export_reviewed_ideal_static_prefix(tmp_path,manifest,qualified_point_indices=np.array([0,1]),**args)
