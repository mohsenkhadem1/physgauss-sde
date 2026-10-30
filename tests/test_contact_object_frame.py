import copy
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.contact_benchmark_v12.object_frame import decompose,project_deformation,adapt,select_adapter


def fixture():
    rng=np.random.default_rng(42);nodes=rng.normal(size=(20,3))*.04
    T=20;end=12;shift=np.arange(T)[:,None,None]*np.array([.002,0.,.001])
    base=nodes[None]+shift
    obs=base[:end]+np.array([.01,-.003,.002])
    return dict(nodes=nodes,observed=obs,observed_initial=nodes,observed_mask=np.ones((end,len(nodes)),bool),
                observation_matrix=np.eye(len(nodes)),train_end=end,controller_targets=base[:,None],
                attachment_active=np.ones((T,1))),base


def test_decomposition_proper_and_reconstructs():
    d,base=fixture();X=d['nodes'];R=Rotation.from_rotvec([.4,-.1,.7]).as_matrix();b=np.array([1.,2.,3.])
    got_R,got_b,u=decompose(X,X@R+b)
    assert np.linalg.det(got_R)>0
    np.testing.assert_allclose((X+u)@got_R+got_b,X@R+b,atol=1e-12)
    np.testing.assert_allclose(u,0,atol=1e-12)


def test_pose_transport_preserves_shape():
    d,base=fixture();pred,meta=adapt(d,base)
    np.testing.assert_allclose(pred[d['train_end']:],base[d['train_end']:]+[.01,-.003,.002],atol=1e-12)
    for a,b in zip(pred,base):
        np.testing.assert_allclose(np.linalg.norm(a[:,None]-a[None],axis=-1),np.linalg.norm(b[:,None]-b[None],axis=-1),atol=1e-12)
    np.testing.assert_array_equal(pred[0],base[0])


def test_pose_equivariant_under_global_rigid_transform():
    d,base=fixture();p,_=adapt(d,base)
    R=Rotation.from_rotvec([.4,.2,-.1]).as_matrix();b=np.array([.1,-.3,1.])
    trans=copy.deepcopy(d)
    for key in ('nodes','observed','observed_initial','controller_targets'):trans[key]=trans[key]@R+b
    q,_=adapt(trans,base@R+b)
    np.testing.assert_allclose(q,p@R+b,atol=1e-12)


def test_residual_has_no_rigid_mode():
    d,base=fixture();X=d['nodes'];u=np.random.default_rng(3).normal(size=X.shape)*.001
    correction=project_deformation(X,u)
    R,b,_=decompose(X,X+correction)
    np.testing.assert_allclose(R,np.eye(3),atol=1e-12)
    np.testing.assert_allclose(b,0,atol=1e-12)


def test_future_truth_never_read_and_input_not_mutated():
    d,base=fixture();before=copy.deepcopy(d);p,meta=select_adapter(d,base)
    d['evaluation']={'points':np.full_like(base,np.nan)}
    q,other=select_adapter(d,base)
    np.testing.assert_array_equal(p,q);assert meta==other
    for key in before:
        if isinstance(before[key],np.ndarray):np.testing.assert_array_equal(before[key],d[key])


def test_inactive_release_left_to_physics():
    d,base=fixture();d['attachment_active'][14:]=0
    for method in ('pose','residual'):
        p,_=adapt(d,base,method)
        np.testing.assert_array_equal(p[14:],base[14:])


def test_selector_can_retain_unchanged_model():
    d,base=fixture();d['observed']=base[:d['train_end']].copy()
    p,meta=select_adapter(d,base)
    assert meta['method']=='none'
    np.testing.assert_array_equal(p,base)


def test_attachment_diagnostic_identifies_stationary_detachment():
    from scripts.contact_benchmark_v12.object_frame import estimate_attachment_state
    d,base=fixture();d['observed']=np.broadcast_to(d['nodes'],d['observed'].shape).copy()
    assert estimate_attachment_state(d)['state']=='detached_stationary'
    d['observed']=base[:d['train_end']].copy()
    assert estimate_attachment_state(d)['state']=='kinematically_attached'


def test_attachment_diagnostic_uninformative_static_tool():
    from scripts.contact_benchmark_v12.object_frame import estimate_attachment_state
    d,base=fixture();d['controller_targets'][:]=d['controller_targets'][0]
    assert estimate_attachment_state(d)['state']=='unknown'


def test_independent_selector_candidates_receive_only_shortened_payload(monkeypatch):
    import scripts.contact_benchmark_v12.object_frame as module
    d,base=fixture();cut=7;short=copy.deepcopy(d)
    short['observed']=short['observed'][:cut].copy();short['observed_mask']=short['observed_mask'][:cut].copy();short['train_end']=cut
    shortbase=base.copy();shortbase[cut:]+=[.002,0,0]
    original=module.adapt;calls=[]
    def checked(data,pred,method='pose',fit_end=None,strength=1.,ridge=1.):
        calls.append((data,pred,fit_end))
        return original(data,pred,method,fit_end,strength,ridge)
    monkeypatch.setattr(module,'adapt',checked)
    out,meta=module.select_adapter(d,base,validation_prediction=shortbase,validation_data=short)
    assert meta['independent_validation'] is True
    assert len(calls)==8
    for payload,pred,end in calls[:-1]:
        assert payload is short and pred is shortbase and end==cut
        assert len(payload['observed'])==cut
    assert calls[-1][0] is d and calls[-1][1] is base


def test_independent_selector_rejects_untrimmed_observations():
    import pytest
    d,base=fixture();short=copy.deepcopy(d);short['train_end']=7
    with pytest.raises(ValueError,match='only subprefix'):
        select_adapter(d,base,validation_prediction=base,validation_data=short)
    with pytest.raises(ValueError,match='together'):
        select_adapter(d,base,validation_prediction=base)


def test_validation_targets_change_scores_not_candidate_predictions(monkeypatch):
    import scripts.contact_benchmark_v12.object_frame as module
    d,base=fixture();cut=7;short=copy.deepcopy(d)
    short['train_end']=cut;short['observed']=short['observed'][:cut].copy();short['observed_mask']=short['observed_mask'][:cut].copy()
    original=module.adapt;generated=[]
    def capture(*args,**kwargs):
        out,meta=original(*args,**kwargs)
        if args[0] is short:generated.append(out.copy())
        return out,meta
    monkeypatch.setattr(module,'adapt',capture)
    module.select_adapter(d,base,validation_prediction=base,validation_data=short)
    first=generated.copy();generated.clear();d['observed'][cut:]+=100
    module.select_adapter(d,base,validation_prediction=base,validation_data=short)
    for a,b in zip(first,generated):np.testing.assert_array_equal(a,b)


def test_inactive_failures_remain_explicit_and_active_failures_rejected():
    import pytest
    d,base=fixture();d['attachment_active'][14:]=0;base[16]=np.nan
    for method in ('none','pose','residual'):
        p,meta=adapt(d,base,method)
        assert np.isnan(p[16]).all()
        assert meta['inherited_failed_frames']==[16]
        assert np.isfinite(p[:14]).all()
    base[5]=np.nan
    with pytest.raises(ValueError,match='Active base'):
        adapt(d,base,'pose')


def test_selector_does_not_silently_exclude_failed_validation_frames():
    d,base=fixture();d['attachment_active'][10:]=0;base[11]=np.nan
    p,meta=select_adapter(d,base)
    assert meta['method']=='none'
    assert all(row['failed_validation_frames']==[11] for row in meta['validation'])
    assert all(np.isinf(row['validation_rmse_m']) for row in meta['validation'])
