import copy
import numpy as np
from scipy.spatial import Delaunay
from scripts.prefix_ridge_switch_v13.inputs import phystwin_payload,softvt_payload
from scripts.prefix_ridge_switch_v13.selector import ridge,select,guarded
from scripts.external_benchmark.evaluate import baselines
from scripts.contact_benchmark_v12.scene import StaticScene

def data_raw():
    points=np.array([[x,y,z] for x in (0.,.02) for y in (0.,.02) for z in (0.,.02)])
    T=20;motion=np.arange(T)[:,None,None]*np.array([.0001,0.,0.])[None,None,:]
    return dict(object_points=points[None]+motion,object_visibilities=np.ones((T,8),bool),object_motions_valid=np.ones((T,8),bool),surface_points=np.empty((0,3)),interior_points=np.empty((0,3)),controller_points=np.repeat(points[None,:3],T,axis=0)+motion)

def elements(nodes):
    e=Delaunay(nodes).simplices.copy()
    for row in e:
        if np.linalg.det((nodes[row[1:]]-nodes[row[0]]).T)<0:row[1],row[2]=row[2],row[1]
    return e

def equal(a,b):
    assert set(a)==set(b)
    for k in a:
        if isinstance(a[k],np.ndarray):np.testing.assert_array_equal(a[k],b[k])
        else:assert a[k]==b[k]

def test_same_location_ridge_matches_original():
    d=phystwin_payload(data_raw(),'single_synthetic',12)
    original=dict(d,evaluation={'points':np.broadcast_to(d['support_geometry'],(20,8,3))})
    expected=baselines(original,12)['position_ridge'];actual,_=ridge(d,d['support_geometry'])
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-15)

def test_phystwin_forbidden_future_mutation_through_selection():
    raw=data_raw();changed=copy.deepcopy(raw)
    for k in ('object_points','object_visibilities','object_motions_valid'):changed[k][12:]=0
    changed['manual']=np.ones((20,8,3))*1e9
    a=phystwin_payload(raw,'single_synthetic',12);b=phystwin_payload(changed,'single_synthetic',12);equal(a,b)
    short=phystwin_payload(raw,'single_synthetic',8);short_changed=copy.deepcopy(raw)
    for k in ('object_points','object_visibilities','object_motions_valid'):short_changed[k][8:]=0
    equal(short,phystwin_payload(short_changed,'single_synthetic',8))
    base=np.broadcast_to(a['nodes'],(20,8,3)).copy();meta={'selected_configuration':{'family':'current','method':'none','strength':0.}}
    args=(base,base,base,meta,base)
    x,m=select(a,short,*args,elements=elements(a['nodes']));y,n=select(b,short,*args,elements=elements(a['nodes']))
    np.testing.assert_array_equal(x,y);assert m==n;assert 'validation_error' not in m and m['ridge_validation_rmse_m'] is not None;assert m['switched'] and np.isfinite(x).all()

def soft_raw():
    r=data_raw();T=20;local=np.zeros((T,7));local[:,:3]=[.01,.01,.01];local[:,3]=1
    return dict(object_prefix=r['object_points'],eef_pose=local,robot_root=np.array([0,0,0,1,0,0,0]),width=np.full(T,.04),command=np.full(T,-1.))

def test_softvt_future_and_short_boundary_mutations():
    raw=soft_raw();full=softvt_payload(raw,'object-soft',0,0,end=12)
    changed=copy.deepcopy(raw);changed['object_prefix'][12:]=1e9
    equal(full,softvt_payload(changed,'object-soft',0,0,end=12))
    short=softvt_payload(raw,'object-soft',0,0,end=8);changed['object_prefix'][8:]=1e9
    equal(short,softvt_payload(changed,'object-soft',0,0,end=8))
    old=np.broadcast_to(full['nodes'],(20,8,3)).copy();meta={'selected_configuration':{'family':'current','method':'none','strength':0.}}
    scene=StaticScene(floor=-1.);a,m=select(full,short,old,old,old,meta,old,scene,elements(full['nodes']))
    changed=copy.deepcopy(raw);changed['object_prefix'][12:]=-1e9;b,n=select(softvt_payload(changed,'object-soft',0,0,end=12),short,old,old,old,meta,old,scene,elements(full['nodes']))
    np.testing.assert_array_equal(a,b);assert m==n;assert 'validation_error' not in m and m['ridge_validation_rmse_m'] is not None

def test_invalid_validation_fallback_is_exact():
    d=phystwin_payload(data_raw(),'single_synthetic',12);old=np.arange(20*8*3,dtype=float).reshape(20,8,3);old[4,1,2]=np.nan
    out,meta=select(d,d,old,old,old,{},old)
    assert out.tobytes()==old.tobytes() and not meta['switched']

def test_invalid_anchor_never_raw_ridge_bypass():
    d=phystwin_payload(data_raw(),'single_synthetic',12);anchor=np.full((20,8,3),np.nan)
    out,_=guarded(d,anchor,None,elements(d['nodes']));assert np.isnan(out).all()


def test_switch_full_failure_and_strict_tie(monkeypatch):
    import scripts.prefix_ridge_switch_v13.selector as module
    d=phystwin_payload(data_raw(),'single_synthetic',12);short=phystwin_payload(data_raw(),'single_synthetic',8)
    old=np.broadcast_to(d['nodes'],(20,8,3)).copy();meta={'selected_configuration':{'family':'current','method':'none','strength':0.}}
    for improvement,switched in [(.98,False),(.5,True)]:
        scores=iter([(1.,[]),(improvement,[])])
        monkeypatch.setattr(module,'_score',lambda *args:next(scores))
        def fake_guarded(data,*args):return (np.full_like(old,np.nan) if data['train_end']==12 else old[:12].copy()),{}
        monkeypatch.setattr(module,'guarded',fake_guarded)
        out,a=module.select(d,short,old,old,old,meta,old,elements=elements(d['nodes']))
        assert a['switched']==switched and 'validation_error' not in a
        if switched:assert np.isnan(out).all() and len(a['failed_full_frames'])==20
        else:assert out.tobytes()==old.tobytes()

def test_scene_export_replay_never_reopens_raw(tmp_path,monkeypatch):
    from scripts.prefix_ridge_switch_v13.inputs import save_scene,load_scene_payload
    import scripts.contact_benchmark_v12.scene as scene_module
    v=np.array([[0.,0.,0.],[.02,0.,0.],[0.,.02,0.]])
    mesh=dict(name='fixture',vertices=v,triangles=np.array([[0,1,2]]),pose=[],scale=[],provenance={})
    original=scene_module.StaticScene([mesh],floor=-1.)
    monkeypatch.setattr(scene_module,'load_scene',lambda d:original)
    save_scene(tmp_path/'scene.npz',{})
    def forbidden(*a,**k):raise AssertionError('Raw scene loader accessed by forecast')
    monkeypatch.setattr(scene_module,'load_scene',forbidden)
    rebuilt=load_scene_payload(tmp_path/'scene.npz');query=np.array([[.002,.002,.01]])
    for key,value in original.query_surface(query).items():np.testing.assert_array_equal(value,rebuilt.query_surface(query)[key])
