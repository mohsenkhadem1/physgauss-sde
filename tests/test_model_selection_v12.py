import copy
import numpy as np
import pytest
import scripts.contact_benchmark_v12.model_selection as module


def fixture():
    nodes=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    T=7;end=5;cut=3;base=np.tile(nodes,(T,1,1));observed=base[:end].copy();observed[cut:,:,0]+=.1
    data=dict(dataset='PhysTwin',nodes=nodes,observed=observed,observed_initial=nodes,observed_mask=np.ones((end,4),bool),train_end=end,
      times=np.arange(T)*.1,controller_targets=base[:,None],attachment_active=np.ones((T,1)),attachment_weights=np.ones((1,4)),controls=base[:,:1],observation_matrix=np.eye(4),floor=0.)
    short=copy.deepcopy(data);short['train_end']=cut;short['observed']=observed[:cut].copy();short['observed_mask']=data['observed_mask'][:cut].copy()
    field=base.copy();field[:,:,0]+=.1
    families={'current':{'full':base.copy(),'validation':base.copy()},'field_free':{'full':field.copy(),'validation':field.copy()}}
    return data,short,families,base.copy()


def mocks(monkeypatch):
    calls=[]
    monkeypatch.setattr(module,'inherited_elements',lambda d:np.array([[0,1,2,3]]))
    def adapt(d,p,method,strength):
        calls.append((d['train_end'],len(d['observed']),len(d['times']),len(p)))
        if not np.isfinite(p).all():raise ValueError('base future active failure')
        return p.copy(),{}
    monkeypatch.setattr(module,'adapt',adapt)
    monkeypatch.setattr(module,'repair',lambda d,a,p,elements:(p.copy(),{}))
    monkeypatch.setattr(module,'rigid_nodal',lambda d:np.tile(d['nodes'],(len(d['times']),1,1)))
    return calls


def test_evaluator_and_extra_observations_rejected(monkeypatch):
    mocks(monkeypatch);d,s,f,b=fixture();d['evaluation']={}
    with pytest.raises(ValueError,match='Evaluator'):module.select_model(d,s,f,{'method':'none'},b)
    del d['evaluation'];s['observed']=d['observed'].copy()
    with pytest.raises(ValueError,match='exactly'):module.select_model(d,s,f,{'method':'none'},b)


def test_validation_uses_only_short_payload_and_short_fits(monkeypatch):
    calls=mocks(monkeypatch);d,s,f,b=fixture()
    out,meta=module.select_model(d,s,f,{'method':'none'},b)
    assert meta['selected_configuration']['family']=='field_free'
    assert all(c==(3,3,5,5) for c in calls[:14])
    assert set(out)=={'selected','current','field_free','rigid'}
    assert meta['switched_from_previous']
    changed=copy.deepcopy(f)
    for family in changed:changed[family]['full']*=20
    _,other=module.select_model(d,s,changed,{'method':'none'},b)
    assert meta['validation']==other['validation']


def test_postvalidation_solver_failures_cannot_affect_selection(monkeypatch):
    mocks(monkeypatch);d,s,f,b=fixture()
    _,first=module.select_model(d,s,f,{'method':'none'},b)
    for family in f:f[family]['validation'][d['train_end']:]=np.nan
    _,second=module.select_model(d,s,f,{'method':'none'},b)
    assert first['validation']==second['validation']


def test_default_ties_retain_exact_saved_prediction(monkeypatch):
    mocks(monkeypatch);d,s,f,b=fixture();f['field_free']=copy.deepcopy(f['current']);b+=.004
    out,meta=module.select_model(d,s,f,{'method':'pose','strength':.5},b)
    assert not meta['switched_from_previous'];np.testing.assert_array_equal(out['selected'],b)
    assert meta['validation'][0]['method']=='pose' and meta['validation'][0]['strength']==.5


def test_allfailed_retain_saved_default(monkeypatch):
    mocks(monkeypatch);d,s,f,b=fixture()
    monkeypatch.setattr(module,'_predict',lambda *a:(_ for _ in ()).throw(ValueError('fail')))
    out,meta=module.select_model(d,s,f,{'method':'none'},b)
    np.testing.assert_array_equal(out['selected'],b);assert not meta['switched_from_previous']
    assert all(r['validation_rmse_m'] is None for r in meta['validation'])
    assert np.isnan(out['field_free']).all()


def test_selected_full_failure_is_not_replaced(monkeypatch):
    mocks(monkeypatch);d,s,f,b=fixture();f['field_free']['full'][:]=np.nan
    out,meta=module.select_model(d,s,f,{'method':'none'},b)
    assert meta['selected_configuration']['family']=='field_free'
    assert np.isnan(out['selected']).all()
    assert meta['final_predictions']['selected']['failed_full_prediction']


def test_any_failed_validation_frame_invalidates_candidate():
    d,s,f,b=fixture();p=f['field_free']['validation'].copy();p[3,0]=np.nan
    value,failed=module._score(d,p,s,3,5)
    assert np.isinf(value) and failed==[3]


def test_rigid_uses_nodes_and_observations_without_evaluator():
    d,s,_,_=fixture();p=module.rigid_nodal(s)
    assert p.shape==(7,4,3);np.testing.assert_array_equal(p[0],s['nodes'])
    for frame in p:np.testing.assert_allclose(np.linalg.norm(frame-frame[0],axis=1),np.linalg.norm(s['nodes']-s['nodes'][0],axis=1),atol=1e-12)


def test_final_topology_invalidates_validation_and_full_prediction(monkeypatch):
    mocks(monkeypatch);d,s,f,b=fixture()
    def bad_prediction(data,anchor,config,scene,elements):
        result=anchor.copy();result[:,[0,1]]=result[:,[1,0]]
        return result,{}
    monkeypatch.setattr(module,'_predict',bad_prediction)
    out,meta=module.select_model(d,s,f,{'method':'none'},b)
    assert not meta['switched_from_previous']
    assert all(row['failed_topology_frames']==[3,4] for row in meta['validation'])
    np.testing.assert_array_equal(out['selected'],b)
