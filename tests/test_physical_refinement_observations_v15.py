import numpy as np
from scripts.physical_refinement_v15.observations import Camera,box_mesh,render_scene,compare_rgbd

def test_occluded_fixture_and_metric_depth():
    block=box_mesh([0,0,0],[.08,.06,.05],[.2,.5,.7]);fixtures=[box_mesh([x,.04,0],[.012,.02,.016],[.5,.5,.5])for x in [-.02,.02]]
    front=Camera((0,-.3,0),(0,0,0));rear=Camera((0,.3,0),(0,0,0))
    a,b=[render_scene([block,f],front)for f in fixtures]
    np.testing.assert_array_equal(a['rgb'],b['rgb']);np.testing.assert_array_equal(a['depth'],b['depth'])
    assert np.isclose(a['depth'][60,80],.27)
    c,d=[render_scene([block,f],rear)for f in fixtures]
    assert compare_rgbd(c,d)['depth_rmse_m']>.001
    assert set(a)=={'rgb','depth','camera','depth_convention'}

def test_rigid_scene_camera_translation_invariance():
    box=box_mesh([0,0,0],[.08,.06,.05],[.2,.5,.7]);cam=Camera((0,-.3,0),(0,0,0));a=render_scene([box],cam)
    shift=np.array([.1,.2,-.3]);m=dict(box,vertices=box['vertices']+shift);c=Camera(tuple(np.array(cam.eye)+shift),tuple(np.array(cam.target)+shift));b=render_scene([m],c)
    np.testing.assert_allclose(a['rgb'],b['rgb'],atol=1e-12);np.testing.assert_allclose(a['depth'],b['depth'],atol=1e-12)

def test_fixture_geometry_uses_mechanical_area_density():
    from scripts.physical_refinement_v15.fixture_geometry import graded_fixture
    from scripts.position_ood.smooth_fixture import footprint
    mesh,meta=graded_fixture('upper',False)
    yz=meta['yz_m'];w=footprint(yz[:,0],yz[:,1],'upper');np.testing.assert_allclose(meta['normalized_area'],w/w.sum())
    assert np.isclose(meta['cross_section_area_m2'].sum(),.0008)
    assert np.isclose(mesh['vertices'][:,0].min(),.24)

def test_fixture_endpoints_use_same_pose_and_state():
    from scripts.physical_refinement_v15.state_scene import scene_from_state
    class Model:
        def query_field(self,u,p):return np.broadcast_to(u,p.shape)
    poses=np.zeros((2,6));poses[1,:3]=[.001,.002,.003];u=np.array([.0001,.0002,.0003])
    meshes=scene_from_state(Model(),u,'parallel_lower',poses)
    right=meshes[2]['vertices'];assert np.isclose(right[:,0].min(),.2401)
    assert np.isclose(right[:,0].max(),.253)

def test_actual_polygon_cross_sections_match_declared_area():
    from scripts.physical_refinement_v15.fixture_geometry import graded_fixture
    mesh,meta=graded_fixture('lower',True)
    vertices=mesh['vertices'].reshape(-1,16,3)
    cross=vertices[:,:8,1:]
    areas=.5*np.abs(np.sum(cross[:,:,0]*np.roll(cross[:,:,1],-1,axis=1)-cross[:,:,1]*np.roll(cross[:,:,0],-1,axis=1),axis=1))
    np.testing.assert_allclose(areas,meta['cross_section_area_m2'],rtol=1e-11,atol=1e-18)
    assert np.isclose(areas.sum(),.0008)
    offsets=cross-meta['yz_m'][:,None]
    assert np.linalg.norm(offsets,axis=2).max()<.004
