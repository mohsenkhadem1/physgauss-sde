"""Independent scoring checks; no fresh data or external target reads."""
import hashlib,json
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02 import object_score as score


def test_point_expansion_preserves_geometry_and_arbitrary_joint_samples():
    nodes=np.array([[0.,0.,0.],[1.,0.,0.]])
    A=np.array([[.75,.25],[.5,.5]])
    initial=np.array([[.25,.2,0.],[.5,.3,0.]])
    prediction=np.broadcast_to(nodes,(3,4,2,3)).copy()+np.array([.1,.2,.3])
    expanded=score.expand_points(prediction,nodes,A,initial)
    assert expanded.shape==(3,4,2,3)
    assert np.allclose(expanded,initial+np.array([.1,.2,.3]))


def test_geometry_score_separates_rigid_motion_and_shape():
    target=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    pred=target@rotation+np.array([.2,-.3,.5])
    result=score.geometry_score(np.broadcast_to(pred,(5,4,3)),np.broadcast_to(target,(5,4,3)),np.ones((5,4),bool),3,5)
    assert result['summary']['coordinate_rmse_mm']>100
    assert result['summary']['aligned_shape_rmse_mm']<1e-10
    assert [r['frame']for r in result['frames']]==[3,4]


def test_nonfinite_failures_and_empty_reference_frames_remain_visible():
    p=np.zeros((4,4,3));y=p.copy();mask=np.ones((4,4),bool)
    p[2,0,0]=np.nan;mask[3]=False
    r=score.geometry_score(p,y,mask,1,4)['summary']
    assert r['total_frames']==3 and r['valid_frames']==1
    assert r['failed_frames']==1 and r['no_reference_frames']==1


def test_matched_prefix_velocity_has_no_future_input():
    initial=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    times=np.arange(6)*.1;velocity=np.array([.02,-.01,.03])
    observed=initial[None]+times[:3,None,None]*velocity
    payload=dict(train_end=3,times=times,observed_initial=initial,observed=observed,observed_mask=np.ones((3,4),bool))
    result=score.prefix_baselines(payload,[0,2],np.arange(4))
    expected=initial[None]+times[:,None,None]*velocity
    assert np.allclose(result['prefix_velocity'],expected)
    assert np.allclose(result['prefix_persistence'][3:],observed[-1])
    with pytest.raises(ValueError):score.prefix_baselines(payload,[0,3],np.arange(4))


def test_nodewise_bounds_are_not_used_as_pointwise_posterior():
    nodes=np.array([[0.,0.,0.],[1.,0.,0.]])
    A=np.array([[.5,.5]]);initial=np.array([[.5,0.,0.]])
    target=np.broadcast_to(initial,(3,1,3));mask=np.ones((3,1),bool)
    assert score.interval_score(None,nodes,A,initial,target,mask,1,3)['available']is False
    samples=np.broadcast_to(nodes,(2,3,2,3)).copy()
    samples[0,:,0,0]+=1;samples[0,:,1,0]-=1
    samples[1,:,0,0]-=1;samples[1,:,1,0]+=1
    exact=score.interval_score(samples,nodes,A,initial,target,mask,1,3)
    assert exact['available'] and exact['coordinate_coverage']==1.
    assert exact['average_width_mm']==0.
    marginal_lo=np.quantile(samples,.05,axis=0)
    assert not np.allclose(score.expand_points(marginal_lo,nodes,A,initial),target)


def test_seal_failure_precedes_target_reader(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(score,'load_phystwin_evaluation',lambda *a,**k:calls.append(True))
    with pytest.raises(ValueError):score.score_case(tmp_path)
    assert not calls


def test_sealed_prediction_digest_is_checked(tmp_path):
    plan={'case':'single_lift_sloth'};planpath=tmp_path/'plan.json';planpath.write_text(json.dumps(plan))
    np.savez(tmp_path/'prediction.npz',flow=np.zeros((3,4,3)))
    seal={'train_end':2,'forecast_target_access':False,'plan_sha256':hashlib.sha256(planpath.read_bytes()).hexdigest(),'bank_sha256':'unused'}
    complete={'completed':True,'held_target_access':False,'fit_end':2,'prediction_sha256':hashlib.sha256((tmp_path/'prediction.npz').read_bytes()).hexdigest()}
    (tmp_path/'inference-sealed.json').write_text(json.dumps(seal));(tmp_path/'complete.json').write_text(json.dumps(complete))
    assert score.load_sealed(tmp_path)[0]==plan
    with (tmp_path/'prediction.npz').open('ab')as f:f.write(b'changed')
    with pytest.raises(ValueError):score.load_sealed(tmp_path)


def test_scorer_prefers_persisted_inputs_and_checks_their_hash(tmp_path,monkeypatch):
    from scripts.gaussian_boundary_flow_v02.export_inputs import write_package
    from scripts.gaussian_boundary_flow_v02.benchmark_inputs import predictor_digest
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    data=dict(name='single_lift_mock',dataset='PhysTwin',nodes=nodes,observed_initial=nodes.copy(),
              observed=np.broadcast_to(nodes,(3,4,3)).copy(),observed_mask=np.ones((3,4),bool),
              observation_matrix=np.eye(4),train_end=3,times=np.arange(5)*.1,
              controller_targets=np.broadcast_to(nodes,(5,1,4,3)).copy(),attachment_weights=np.ones((1,4)))
    path=tmp_path/'inputs';write_package(data,path)
    plan=dict(case=data['name'],args={'nodes':4},input_package=str(path),input_digest=predictor_digest(data))
    seal={'train_end':3}
    def forbidden(*a,**k):raise AssertionError('Scorer re-exported persisted inputs')
    monkeypatch.setattr(score,'load_phystwin_predictor',forbidden)
    assert np.array_equal(score.load_inference_payload(plan,seal)['nodes'],nodes)
    with pytest.raises(ValueError,match='sealed inference'):
        score.load_inference_payload(dict(plan,input_digest='bad'),seal)
    metadata=path/'predictor-meta.json';metadata.write_text(metadata.read_text()+' ')
    with pytest.raises(ValueError,match='file hash'):
        score.load_inference_payload(plan,seal)


def test_scorer_old_pilot_reexport_remains_supported(monkeypatch):
    from scripts.gaussian_boundary_flow_v02.benchmark_inputs import predictor_digest
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    data=dict(name='single_lift_mock',dataset='PhysTwin',nodes=nodes,observed_initial=nodes.copy(),
              observed=np.broadcast_to(nodes,(3,4,3)).copy(),observed_mask=np.ones((3,4),bool),
              observation_matrix=np.eye(4),train_end=3,times=np.arange(5)*.1,
              controller_targets=np.broadcast_to(nodes,(5,1,4,3)).copy())
    calls=[]
    def legacy(*args,**kwargs):calls.append((args,kwargs));return data
    monkeypatch.setattr(score,'load_phystwin_predictor',legacy)
    plan=dict(case=data['name'],args={'nodes':4},input_digest=predictor_digest(data))
    assert score.load_inference_payload(plan,{'train_end':3})is data
    assert len(calls)==1 and calls[0][1]['max_nodes']==4


def test_generic_method_names_preserve_oracle_identity_and_default():
    pred={'nodes':np.zeros((4,3)),'oracle_raw':np.zeros((5,4,3)),
          'oracle_tracking':np.zeros((5,4,3))}
    names=['oracle_raw','oracle_tracking']
    assert score.prediction_methods({'prediction_methods':names},pred,5)==names
    assert score.prediction_methods({},dict(pred,flow=pred['oracle_raw'],mixture=pred['oracle_raw']),5)==['flow','mixture']
    with pytest.raises(ValueError,match='Missing declared'):
        score.prediction_methods({'prediction_methods':['missing']},pred,5)
    for invalid in ('nodes','prefix_persistence','oracle_raw_samples','../raw'):
        with pytest.raises(ValueError,match='Reserved or invalid'):
            score.prediction_methods({'prediction_methods':[invalid]},pred,5)
    with pytest.raises(ValueError,match='Duplicate'):
        score.prediction_methods({'prediction_methods':['oracle_raw']*2},pred,5)
    with pytest.raises(ValueError,match='numeric full node'):
        score.prediction_methods({'prediction_methods':['oracle_raw']},dict(pred,oracle_raw=np.zeros((4,4,3))),5)
    with pytest.raises(ValueError,match='complete trajectory'):
        score.prediction_methods({'prediction_methods':['oracle_raw']},dict(pred,oracle_raw_samples=np.zeros((2,4,4,3))),5)


def test_interval_samples_require_complete_clock_not_broadcasting():
    nodes=np.zeros((4,3));A=np.eye(4);target=np.zeros((5,4,3));mask=np.ones((5,4),bool)
    with pytest.raises(ValueError,match='complete target clock'):
        score.interval_score(np.zeros((2,1,4,3)),nodes,A,nodes,target,mask,0,5)


@pytest.mark.parametrize('changed',['inference-sealed.npz','transform.pt','flow.pt','bank.npz','partial','unsafe_name','missing'])
def test_optional_v03_provenance_tamper_fails_before_targets(tmp_path,monkeypatch,changed):
    planpath=tmp_path/'plan.json';planpath.write_text('{}')
    np.savez(tmp_path/'prediction.npz',flow=np.zeros((3,4,3)))
    for name in ['inference-sealed.npz','transform.pt','flow.pt','bank.npz']:(tmp_path/name).write_bytes(b'bound checkpoint')
    seal=dict(schema=1,forecast_target_access=False,train_end=2,plan_sha256=score.sha(planpath),
              inference_sha256=score.sha(tmp_path/'inference-sealed.npz'),transform_sha256=score.sha(tmp_path/'transform.pt'),
              models_sha256={'flow':score.sha(tmp_path/'flow.pt')},bank_sha256=score.sha(tmp_path/'bank.npz'))
    complete=dict(completed=True,held_target_access=False,fit_end=2,prediction_sha256=score.sha(tmp_path/'prediction.npz'))
    (tmp_path/'complete.json').write_text(json.dumps(complete));(tmp_path/'inference-sealed.json').write_text(json.dumps(seal))
    assert score.load_sealed(tmp_path)[1]==seal
    if changed=='missing':(tmp_path/'transform.pt').unlink()
    elif changed=='partial':del seal['transform_sha256']
    elif changed=='unsafe_name':seal['models_sha256']={'../outside':seal['models_sha256']['flow']}
    else:(tmp_path/changed).write_bytes(b'tampered')
    (tmp_path/'inference-sealed.json').write_text(json.dumps(seal))
    calls=[];monkeypatch.setattr(score,'load_phystwin_evaluation',lambda *a,**k:calls.append(True))
    with pytest.raises(ValueError):score.score_case(tmp_path)
    assert calls==[]


def test_support_flag_uses_explicit_decision_without_rewriting_historical_evidence():
    diagnostic={'out_of_bank_warning':True,'real_standardized_context_norm':1.,'bank_context_norm_max':5.,'feature_support':{'discarded_space':True}}
    assert score.support_screen(diagnostic)['out_of_bank_warning']is True
    legacy={key:value for key,value in diagnostic.items()if key!='out_of_bank_warning'}
    result=score.support_screen(legacy)
    assert result['out_of_bank_warning']is False and 'incomplete screening' in result['support_screening']
    assert score.support_screen(None)['out_of_bank_warning']is None
