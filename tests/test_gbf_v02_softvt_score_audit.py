import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02 import softvt_score as score
from scripts.gaussian_boundary_flow_v02.export_inputs import write_package
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import predictor_digest


@pytest.fixture
def setup(tmp_path,monkeypatch):
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    initial=nodes[[2,0,3]];observed=np.broadcast_to(initial,(3,3,3)).copy()
    source=tmp_path/'raw.hdf5';source.touch()
    data=dict(name='object-soft-task1-demo0',dataset='SoftVTBench',nodes=nodes,support_geometry=nodes.copy(),observed_initial=initial,
              observed=observed,observed_mask=np.ones((3,3),bool),observation_matrix=np.eye(4)[[2,0,3]],
              train_end=3,times=np.arange(5)*.1,controller_targets=np.broadcast_to(nodes,(5,1,4,3)).copy(),
              attachment_weights=np.ones((1,4)),source=str(source))
    package=tmp_path/'inputs';write_package(data,package)
    plan=dict(case=data['name'],args={'nodes':4},input_package=str(package),input_package_manifest_sha256=score.shared.sha(package/'manifest.json'),
              input_digest=predictor_digest(data),prediction_methods=['oracle_raw'],prefix_frames=[0,2],point_ids=[0,1,2])
    seal={'train_end':3};complete={'fit_end':3,'frames':5}
    pred={k:data[k]for k in ['nodes','observation_matrix','observed_initial']};pred['oracle_raw']=np.broadcast_to(nodes,(5,4,3)).copy()
    evaluation={'points':np.broadcast_to(nodes,(5,4,3)).copy(),'mask':np.ones((5,4),bool)}
    monkeypatch.setattr(score,'_softvt_path',lambda *a:source)
    monkeypatch.setattr(score.shared,'load_sealed',lambda *a:(plan,seal,complete,pred))
    calls=[]
    def evaluate(*a,**kw):calls.append(True);return evaluation
    monkeypatch.setattr(score,'load_softvt_evaluation',evaluate)
    return data,plan,seal,complete,pred,evaluation,calls,tmp_path


def test_identity_and_order_come_only_from_initial_geometry(setup):
    data,plan,seal,complete,pred,evaluation,calls,tmp=setup
    loaded,_,names,_=score.verified_inputs(plan,seal,complete,pred)
    assert names==['oracle_raw']
    assert score.target_observation_ids(loaded,evaluation).tolist()==[2,0,3]
    assert not calls
    evaluation['points'][4]+=100 # suffix does not change matching
    assert score.target_observation_ids(loaded,evaluation).tolist()==[2,0,3]
    evaluation['points'][1,2,0]+=1
    with pytest.raises(ValueError,match='prefix differs'):score.target_observation_ids(loaded,evaluation)


@pytest.mark.parametrize('kind',['digest','manifest','geometry','method','fresh','clock'])
def test_pre_evaluation_guards(setup,kind):
    data,plan,seal,complete,pred,evaluation,calls,tmp=setup
    if kind=='digest':plan['input_digest']='bad'
    elif kind=='manifest':plan['input_package_manifest_sha256']='bad'
    elif kind=='geometry':pred['nodes']=pred['nodes']+1
    elif kind=='method':plan['prediction_methods']=['missing']
    elif kind=='fresh':plan['case']='object-soft-task1-demo50'
    else:complete['frames']=4
    with pytest.raises((ValueError,PermissionError)):score.score_case(tmp)
    assert not calls


def test_scoring_sparse_same_ids_and_no_fake_intervals(setup,monkeypatch):
    data,plan,seal,complete,pred,evaluation,calls,tmp=setup
    monkeypatch.setattr(score.shared,'sha',lambda p:'testdigest')
    plan['input_package_manifest_sha256']='testdigest'
    result=score.score_case(tmp)
    assert calls==[True] and result['interval']==[3,5]
    assert result['observed_node_ids']==[2,0,3] and not result['full_mesh_metric']
    assert result['methods']['oracle_raw']['summary']['coordinate_rmse_mm']==0
    assert not result['posterior_intervals']['oracle_raw']['available']


def test_changed_initial_order_rejected(setup):
    data,_,_,_,_,evaluation,_,_=setup
    evaluation['points']=evaluation['points'][:,[1,0,2,3]]
    with pytest.raises(ValueError,match='initial node order'):score.target_observation_ids(data,evaluation)
