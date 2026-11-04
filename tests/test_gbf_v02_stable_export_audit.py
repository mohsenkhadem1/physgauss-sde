"""Stable predictor package tests; no benchmark target access."""
import json
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02 import export_inputs as export
from scripts.gaussian_boundary_flow_v02 import benchmark_inputs as adapter


def payload():
    nodes=np.array([[x,y,z]for x in (0.,.5,1.)for y in (0.,.5,1.)for z in (0.,.5,1.)])
    T=8;end=4;times=np.arange(T)*.05
    controls=nodes[None,None]+times[:,None,None,None]*np.array([.01,.02,.03])
    return dict(name='synthetic-contract-only',dataset='SoftVTBench',nodes=nodes,
                observed=nodes[None]+times[:end,None,None]*np.array([.01,.02,.03]),
                observed_initial=nodes.copy(),observed_mask=np.ones((end,len(nodes)),bool),
                observation_matrix=np.eye(len(nodes)),controller_targets=controls,
                attachment_weights=np.ones((1,len(nodes))),attachment_active=np.ones((T,1)),
                floor=0.,train_end=end,times=times,controls=np.zeros((T,1,3)),
                source='synthetic-test-only',units='m,s')


def test_round_trip_preserves_exact_arrays_and_digest(tmp_path):
    original=payload();path=tmp_path/'package'
    manifest=export.write_package(original,path)
    loaded=export.load_package(path)
    assert adapter.predictor_digest(loaded)==adapter.predictor_digest(original)==manifest['predictor_digest']
    for key,value in original.items():
        if isinstance(value,np.ndarray):
            assert loaded[key].dtype==value.dtype
            assert np.array_equal(loaded[key],value)
        else:assert loaded[key]==value
    assert manifest['evaluation_exported']is False


def test_reduction_keeps_initial_support_and_complete_commands():
    data=payload();original_nodes=data['nodes'].copy();targets=data['controller_targets'].copy()
    reduced=export.reduce_geometry(data,8);ids=reduced['input_geometry_reduction']['source_node_ids']
    assert reduced['nodes'].shape==(8,3)
    assert np.array_equal(reduced['support_geometry'],original_nodes)
    assert np.array_equal(reduced['controller_targets'],targets[:,:,ids])
    assert np.array_equal(reduced['attachment_weights'],data['attachment_weights'][:,ids])
    assert reduced['observation_matrix'].shape==(27,8)
    assert np.allclose(reduced['observation_matrix'].sum(-1),1.)
    assert np.array_equal(reduced['observed'],data['observed'])
    assert np.array_equal(data['nodes'],original_nodes)
    assert np.array_equal(data['controller_targets'],targets)
    assert 'support_geometry'not in data
    with pytest.raises(ValueError):export.reduce_geometry(data,3)


def test_existing_support_geometry_is_not_replaced_by_coarse_nodes():
    data=payload();data['support_geometry']=np.r_[data['nodes'],[[.3,.4,.5],[.2,.6,.7]]]
    reduced=export.reduce_geometry(data,8)
    assert np.array_equal(reduced['support_geometry'],data['support_geometry'])


def test_loading_does_not_reexport_or_retriangulate(tmp_path,monkeypatch):
    path=tmp_path/'package';export.write_package(export.reduce_geometry(payload(),8),path)
    def forbidden(*a,**k):raise AssertionError('Raw export/triangulation called during reload')
    monkeypatch.setattr(adapter,'load_phystwin_predictor',forbidden)
    monkeypatch.setattr(adapter,'load_softvt_predictor',forbidden)
    from scripts.external_benchmark import data
    monkeypatch.setattr(data,'fps',forbidden);monkeypatch.setattr(data,'interpolate_matrix',forbidden)
    assert export.load_package(path)['nodes'].shape==(8,3)


def test_file_hash_and_semantic_digest_are_verified(tmp_path):
    path=tmp_path/'package';export.write_package(payload(),path)
    manifest=path/export.MANIFEST_FILE;raw=manifest.read_text();m=json.loads(raw);m['predictor_digest']='0'*64;manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='semantic digest'):export.load_package(path)
    manifest.write_text(raw)
    with(path/export.ARRAY_FILE).open('ab')as f:f.write(b'changed')
    with pytest.raises(ValueError,match='file hash'):export.load_package(path)


def test_immutable_packages_refuse_changed_input(tmp_path):
    path=tmp_path/'package';data=payload();first=export.write_package(data,path)
    assert export.write_package(data,path)['predictor_digest']==first['predictor_digest']
    changed=payload();changed['observed']=changed['observed']+.001
    with pytest.raises(ValueError,match='immutable'):export.write_package(changed,path)
    assert adapter.predictor_digest(export.load_package(path))==first['predictor_digest']


def test_pickle_object_arrays_and_evaluation_fields_rejected(tmp_path):
    data=payload();data['unsafe']=np.array([{'x':1}],dtype=object)
    with pytest.raises(ValueError,match='pickle'):export.write_package(data,tmp_path/'unsafe')
    data=payload();data['evaluation']={'future':1}
    with pytest.raises(ValueError):export.write_package(data,tmp_path/'evaluation')
