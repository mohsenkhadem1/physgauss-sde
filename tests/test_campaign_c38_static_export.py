"""Tiny manufactured metadata and affine P2 fields; no physical data admitted."""
import copy
from pathlib import Path
import numpy as np
import pytest
from test_campaign_c38_static_study import gate_fixture,geometry,save
from experiments import campaign_c38_static_export as m


def fixture(tmp):
    gate,manifest,field,commit=gate_fixture(tmp);g=geometry()
    gr=save(tmp,'public_geometry.npz',**g)
    forces=np.zeros((2,10,3));forces[0,:,2]=-.1;forces[1,:,2]=.1
    cr=save(tmp,'controls.npz',load_state_indices=np.array([7,8]),load_vectors_N=forces)
    future=save(tmp,'future.npz',**g,positions_m=g['rest_m']+100,constraint_mask=np.ones((10,3),bool))
    future.update(object_id='objectA',load_state_index=8,domain_sha256='d'*64)
    for i,r in enumerate((field,future)):
        r['public_controls']=cr;r['load_vector_sha256']=m.force_row_sha256(forces[i])
    gate['admitted_static_fields']=[field,future];commit(gate)
    args=dict(object_id='objectA',geometry_record=gr,controls_record=cr,
              acquisitions=[dict(acquisition_index=0,load_state_index=7,field=field),dict(acquisition_index=2,load_state_index=8,field=future)],
              last_acquisition=0,element_indices=np.array([0,0]),barycentric=np.array([[.25]*4,[.1,.2,.3,.4]]),
              measurement_errors_m=np.array([[[1e-5,0,0],[0,2e-5,0]]]),sensor_covariance_m2=np.eye(6)*.0002**2)
    return gate,manifest,args,commit


def compare_public(a,b):
    assert a.keys()==b.keys()
    for k in a:
        if isinstance(a[k],dict):compare_public(a[k],b[k])
        elif isinstance(a[k],np.ndarray):assert np.array_equal(a[k],b[k])
        else:assert a[k]==b[k]


def test_compatible_affine_p2_interpolation_and_exact_public_allowlist(tmp_path):
    gate,manifest,args,_=fixture(tmp_path);public,private=m.export_ideal_static_prefix(tmp_path,manifest,**args)
    o=public['observations'];assert np.allclose(o['observed_positions_m'],o['material_points_m'][None]+.0001+args['measurement_errors_m'],rtol=0,atol=1e-16)
    assert set(public)=={'information_role','geometry','known_controls','observations','observation_kind','is_RGBD_perception','is_temporal_forecast'}
    assert o['acquisition_indices'].tolist()==[0] and o['load_state_indices'].tolist()==[7]
    assert not public['is_RGBD_perception'] and not public['is_temporal_forecast']
    assert set(public['geometry'])=={'rest_m','elements','tets','faces'}
    assert set(public['known_controls'])=={'information_role','load_state_indices','load_vectors_N'}
    assert not {'young_Pa','support_category','constraint_mask','positions_m','times_s','field','domain_sha256'}&set(o)
    assert 'PRIVATE_' in private['information_role'] and len(private['acquired_fields'])==1


def test_future_private_arrays_never_opened_and_changes_do_not_change_prefix(tmp_path,monkeypatch):
    gate,manifest,args,commit=fixture(tmp_path);before,_=m.export_ideal_static_prefix(tmp_path,manifest,**args)
    future=args['acquisitions'][1]['field'];g=geometry();changed=save(tmp_path,'different_future.npz',**g,positions_m=g['rest_m']-999,constraint_mask=np.zeros((10,3),bool))
    future.update({k:changed[k] for k in ('path','sha256')});gate['admitted_static_fields'][1]=future;commit(gate)
    opened=[];original=np.load
    def tracked(path,*a,**kw):
        opened.append(str(path));assert 'future' not in Path(path).name;return original(path,*a,**kw)
    monkeypatch.setattr(np,'load',tracked)
    after,_=m.export_ideal_static_prefix(tmp_path,manifest,**args);compare_public(before,after)
    assert not any('future' in Path(p).name for p in opened)


def test_wrong_object_load_or_unreviewed_force_binding_rejected_before_private_field(tmp_path,monkeypatch):
    gate,manifest,args,commit=fixture(tmp_path)
    monkeypatch.setattr(m,'load_admitted_static_targets',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Private field opened')))
    bad=copy.deepcopy(args);bad['object_id']='other'
    with pytest.raises(ValueError,match='object/load'):m.export_ideal_static_prefix(tmp_path,manifest,**bad)
    bad=copy.deepcopy(args);bad['acquisitions'][0]['load_state_index']=8
    with pytest.raises(ValueError,match='object/load'):m.export_ideal_static_prefix(tmp_path,manifest,**bad)
    gate['admitted_static_fields'][0]['load_vector_sha256']='0'*64;commit(gate)
    with pytest.raises(ValueError,match='force binding'):m.export_ideal_static_prefix(tmp_path,manifest,**args)


def test_same_load_id_with_different_control_bytes_cannot_relabel_a_field(tmp_path):
    _,manifest,args,_=fixture(tmp_path)
    args['controls_record']=save(tmp_path,'wrong_forces.npz',load_state_indices=np.array([7,8]),load_vectors_N=np.zeros((2,10,3)))
    with pytest.raises(ValueError,match='force binding'):m.export_ideal_static_prefix(tmp_path,manifest,**args)


def test_geometry_substitution_rejected_even_if_both_records_hash_valid(tmp_path):
    _,manifest,args,_=fixture(tmp_path);g=geometry();g['rest_m']=g['rest_m']+1
    args['geometry_record']=save(tmp_path,'wrong_geometry.npz',**g)
    with pytest.raises(ValueError,match='rest geometry'):m.export_ideal_static_prefix(tmp_path,manifest,**args)


def test_errors_for_future_rows_extra_metadata_and_outside_points_rejected(tmp_path):
    _,manifest,args,_=fixture(tmp_path)
    bad=copy.deepcopy(args);bad['measurement_errors_m']=np.zeros((2,2,3))
    with pytest.raises(ValueError,match='acquired entries only'):m.export_ideal_static_prefix(tmp_path,manifest,**bad)
    bad=copy.deepcopy(args);bad['acquisitions'][0]['young_Pa']=100000
    with pytest.raises(ValueError,match='allowlist'):m.export_ideal_static_prefix(tmp_path,manifest,**bad)
    bad=copy.deepcopy(args);bad['barycentric'][0]=[-.1,.3,.3,.5]
    with pytest.raises(ValueError,match='barycentric'):m.export_ideal_static_prefix(tmp_path,manifest,**bad)
    bad=copy.deepcopy(args);bad['sensor_covariance_m2'][0,0]=-1
    with pytest.raises(ValueError,match='Positive definite'):m.export_ideal_static_prefix(tmp_path,manifest,**bad)


def test_p2_weights_match_independent_existing_shape_implementation():
    from experiments.campaign_static_p2_block import shapes
    g=geometry();b=np.array([[.25]*4,[0.,.2,.3,.5],[1.,0,0,0]])
    nodes,N,X=m._points(g,np.array([0,0,0]),b)
    assert np.array_equal(N,shapes(b));assert np.allclose(X,b@g['rest_m'][:4],rtol=0,atol=1e-17)


def test_constraint_metadata_changes_never_become_public_features(tmp_path):
    gate,manifest,args,commit=fixture(tmp_path);before,_=m.export_ideal_static_prefix(tmp_path,manifest,**args)
    g=geometry();changed=save(tmp_path,'changed_private_mask.npz',**g,positions_m=g['rest_m']+.0001,constraint_mask=np.ones((10,3),bool))
    field=args['acquisitions'][0]['field'];field.update({k:changed[k] for k in ('path','sha256')})
    gate['admitted_static_fields'][0]=field;commit(gate)
    after,_=m.export_ideal_static_prefix(tmp_path,manifest,**args);compare_public(before,after)
