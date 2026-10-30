import numpy as np
from scripts.contact_benchmark_v12.scene import StaticScene


def plane(z,lo=-1,hi=1):
    return dict(name='plane',vertices=np.array([[lo,lo,z],[hi,lo,z],[hi,hi,z],[lo,hi,z]]),triangles=np.array([[0,1,2],[0,2,3]]),pose=[0,0,0,1,0,0,0],scale=[1,1,1],provenance={})


def test_finite_support_and_floor():
    scene=StaticScene([plane(.2)],floor=0,spacing=.1)
    q=scene.query_surface([[0,0,.3],[2,0,.3]])
    np.testing.assert_allclose(q['gap'],[.1,.3]);np.testing.assert_allclose(q['normal'],[[0,0,1],[0,0,1]])
    np.testing.assert_array_equal(q['object_index'],[0,-1])

def test_floor_cannot_be_lost_by_inherited_penetration():
    q=StaticScene(floor=0).query_surface([[0,0,-.01]],max_height=[-.008])
    np.testing.assert_allclose(q['height'],[0]);np.testing.assert_allclose(q['gap'],[-.01])


def test_handle_not_selected_above_previous_point():
    scene=StaticScene([plane(.1),plane(.4)],floor=0,spacing=.1)
    q=scene.query_surface([[0,0,.05],[0,0,.5]],max_height=np.array([.2,.6]))
    np.testing.assert_allclose(q['height'],[.1,.4]);np.testing.assert_allclose(q['gap'],[-.05,.1])


def test_tilted_surface_normal():
    obj=plane(.2);obj['vertices'][:,2]+=.2*obj['vertices'][:,0]
    scene=StaticScene([obj],floor=-1,spacing=.1);q=scene.query_surface([[.5,0,.5]])
    np.testing.assert_allclose(q['height'],[.3]);np.testing.assert_allclose(q['normal'][0],np.array([-.2,0,1])/np.sqrt(1.04))


def test_loader_reads_initial_pose_only(tmp_path,monkeypatch):
    import h5py
    import scripts.contact_benchmark_v12.scene as module
    def mesh(*args,**kwargs):
        p=plane(.1);return p['vertices'],p['triangles'],{}
    monkeypatch.setattr(module,'asset_mesh',mesh)
    path=tmp_path/'scene.hdf5'
    with h5py.File(path,'w') as h:
        h['data/demo_0/initial_state/rigid_object/plate_1/root_pose']=np.array([[0,0,.2,1,0,0,0.]])
        h['data/demo_0/states/rigid_object/plate_1/root_pose']=np.full((5,7),np.nan)
    data=dict(dataset='SoftVTBench',name='spatial-soft-task0-demo0',source=str(path),floor=0)
    a=module.load_scene(data,spacing=.1).query_surface([[0,0,1.]])['height']
    with h5py.File(path,'a') as h:h['data/demo_0/states/rigid_object/plate_1/root_pose'][:]=1e8
    b=module.load_scene(data,spacing=.1).query_surface([[0,0,1.]])['height']
    np.testing.assert_array_equal(a,b)
