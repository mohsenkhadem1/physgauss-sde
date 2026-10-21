import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from campaign_ray_sensor import (batched_ray_depth, intersect_mesh_rays, pixel_rays,
                                 TorchMeshRaySensor, predict_censored_ranges)
from validate_isaac_dataset import ray_depth
from censored_range_observation import predict_censored_ranges as numpy_censored


def scene():
    # Front and back squares, plus a degenerate triangle: nearest two-sided hit.
    front = np.array([[-1,-1,-2],[1,-1,-2],[1,1,-2],[-1,1,-2]], dtype=float)
    q = np.concatenate([front,front*np.array([1,1,2])])
    faces = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,0,0]])
    k = np.array([[10.,0,10.],[0,10.,10.],[0,0,1.]])
    xs,ys = np.array([9.,10.,12.,18.,-1.]), np.array([9.,10.,8.,10.,3.])
    return q,faces,np.eye(4),k,xs,ys


def compare(q, faces, pose, k, xs, ys, **kwargs):
    expected = ray_depth(q,faces,pose,k,xs,ys)
    actual = batched_ray_depth(q,faces,pose,k,xs,ys,**kwargs)
    np.testing.assert_array_equal(actual.visible.numpy(),np.isfinite(expected))
    np.testing.assert_allclose(actual.depth_m.numpy(),expected,rtol=1e-13,atol=2e-14)
    return actual


def test_camera_axis_depth_not_normalized_range_and_misses():
    q,f,p,k,x,y=scene();actual=compare(q,f,p,k,x,y)
    np.testing.assert_array_equal(actual.visible,[True,True,True,False,False])
    np.testing.assert_allclose(actual.depth_m[:3],2.,atol=0,rtol=0)
    assert torch.isinf(actual.depth_m[3:]).all()
    assert actual.depth_m.dtype==torch.float64


def test_usd_row_pose_rotated_translated_and_half_pixel():
    q,f,_,k,x,y=scene()
    a=.63;rotation=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    p=np.eye(4);p[:3,:3]=rotation;p[3,:3]=[.2,-.7,1.3]
    world=q@rotation+p[3,:3]
    compare(world,f,p,k,x,y)
    o,d=pixel_rays(p,k,[9.5],[9.5])
    np.testing.assert_allclose(o,p[3,:3],atol=0)
    np.testing.assert_allclose(d,[[0,0,-1]]@rotation,atol=1e-15)


def test_chunk_boundaries_and_shared_mesh_are_equivalent():
    q,f,p,k,x,y=scene()
    positions=np.stack([q,q+[.07,.02,-.13],q+[.03,-.1,.1]])
    actual=batched_ray_depth(positions,f,p,k,x,y,particle_chunk=2,triangle_chunk=2,ray_chunk=2)
    expected=np.stack([ray_depth(v,f,p,k,x,y) for v in positions])
    np.testing.assert_array_equal(actual.visible,np.isfinite(expected))
    np.testing.assert_allclose(actual.depth_m,expected,atol=2e-14,rtol=1e-13)
    shared=batched_ray_depth(q,f,np.stack([p,p]),k,x,y)
    np.testing.assert_array_equal(shared.depth_m[0],shared.depth_m[1])


def test_particle_time_and_per_time_rays_broadcast_like_explicit_loops():
    q,f,p,k,x,y=scene()
    positions=np.broadcast_to(q,(2,3,*q.shape)).copy()
    positions+=np.arange(6).reshape(2,3,1,1)*.008
    poses=np.stack([p.copy() for _ in range(3)]);poses[:,3,0]=[.0,.1,-.1]
    ks=np.stack([k,k,k]);ks[:,0,0]=[10.,11.,9.]
    xs=np.stack([x,x+.3,x-.2]);ys=np.broadcast_to(y,xs.shape).copy()
    actual=batched_ray_depth(positions,f,poses,ks,xs,ys,particle_chunk=2,triangle_chunk=1)
    expected=np.stack([[ray_depth(positions[i,j],f,poses[j],ks[j],xs[j],ys[j]) for j in range(3)] for i in range(2)])
    np.testing.assert_array_equal(actual.visible,np.isfinite(expected))
    np.testing.assert_allclose(actual.depth_m,expected,atol=2e-14,rtol=1e-13)


def test_censored_conversion_omits_missing_residual_and_matches_reference():
    q,f,p,k,x,y=scene();rest=ray_depth(q,f,p,k,x,y)
    # Censored residual reference must be finite even for ray missing the mesh.
    rest=np.where(np.isfinite(rest),rest,2.)
    query={'pose':p,'K':k,'xs':x,'ys':y,'rest_depth_m':rest}
    reference={'surface_faces':f,'rest_points_m':q}
    trajectories=np.stack([np.stack([q,q+[.01,0,-.1],q+[.3,0,.03]]),
                            np.stack([q,q+[0,.03,-.2],q+[.4,0,-.1]])])
    actual=predict_censored_ranges(trajectories,reference,query)
    expected=numpy_censored(trajectories,reference,query)
    np.testing.assert_array_equal(actual.visible,expected.visible)
    np.testing.assert_allclose(actual.residual_m,expected.residual_m,atol=2e-14)
    np.testing.assert_array_equal(actual.residual_m[~actual.visible],0.)
    assert not np.any(actual.residual_m[~actual.visible]==.05)


def test_cached_per_time_sensor_selects_same_frames_for_cameras_and_rest():
    q,f,p,k,x,y=scene();poses=np.stack([p.copy() for _ in range(3)])
    poses[:,3,0]=[0,.1,.2];query={'pose':poses,'K':k,'xs':x,'ys':y,'rest_depth_m':np.full((3,len(x)),2.)}
    sensor=TorchMeshRaySensor(f,query,particle_chunk=1,triangle_chunk=2)
    trajectories=np.broadcast_to(q,(2,3,*q.shape)).copy()
    actual=sensor.predict_censored(trajectories,[2,0])
    expected=np.stack([np.concatenate([ray_depth(v[t],f,poses[t],k,x,y) for t in [2,0]]) for v in trajectories])
    np.testing.assert_array_equal(actual.visible,np.isfinite(expected))
    np.testing.assert_allclose(actual.residual_m[actual.visible],(expected-2.)[actual.visible],atol=2e-14)


def test_no_faces_and_all_behind_camera_are_misses():
    q,f,p,k,x,y=scene()
    assert not batched_ray_depth(q,np.empty((0,3),int),p,k,x,y).visible.any()
    assert not batched_ray_depth(q*-1,f,p,k,x,y).visible.any()


def test_absolute_parallel_determinant_cutoff_matches_reference():
    # Unit-area triangle facing -z. The reference explicitly rejects |det|<=1e-10.
    points=np.array([[0.,0.,-1.],[1.,0.,-1.],[0.,1.,-1.]])
    directions=np.array([[0.,0.,-1e-12],[0.,0.,-1e-9]])
    result=intersect_mesh_rays(points,[[0,1,2]],[.2,.2,0.],directions)
    np.testing.assert_array_equal(result.visible,[False,True])
    np.testing.assert_allclose(result.depth_m[1],1e9,rtol=1e-15)


def test_depth_has_piecewise_finite_gradient():
    q,f,p,k,x,y=scene();q=torch.tensor(q,requires_grad=True)
    result=batched_ray_depth(q,f,p,k,x[:3],y[:3])
    gradient=torch.autograd.grad(result.depth_m.sum(),q)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0


@pytest.mark.parametrize('changes',[{'triangle_chunk':0},{'dtype':torch.float16}])
def test_invalid_runtime_arguments_rejected(changes):
    q,f,p,k,x,y=scene()
    with pytest.raises(ValueError):batched_ray_depth(q,f,p,k,x,y,**changes)


def test_nonfinite_positions_bad_indices_and_bad_camera_rejected():
    q,f,p,k,x,y=scene();q[0,0]=np.nan
    with pytest.raises(ValueError):batched_ray_depth(q,f,p,k,x,y)
    q[0,0]=0
    with pytest.raises(ValueError):batched_ray_depth(q,[[0,1,99]],p,k,x,y)
    k[0,0]=0
    with pytest.raises(ValueError):batched_ray_depth(q,f,p,k,x,y)


def test_allowed_train_mesh_and_deformed_states_match_reference():
    import json
    path=Path(__file__).resolve().parents[1]/'runs/gaussian-boundary-neural-v01/episodes/t0000'
    if not path.exists():pytest.skip('Optional local TRAIN dataset is absent')
    assert json.loads((path/'summary.json').read_text())['split']=='train'
    with np.load(path/'reference.npz') as raw:
        reference={k:raw[k] for k in raw.files}
    with np.load(path/'observations/cameras.npz') as raw:
        poses=raw['camera_to_world_usd_row'][[0,4,9]];k=raw['K']
    # Only explicitly allowed TRAIN states; no test/control/calibration paths.
    with np.load(path/'truth/trajectory.npz') as raw:
        positions=raw['positions_m'][[0,20,60]]
    from gaussian_acquisition_sensor import public_ray_queries
    query=public_ray_queries(reference,poses[0],k,count=24)
    actual=batched_ray_depth(positions,reference['surface_faces'],poses,k,
                             query['xs'],query['ys'],particle_chunk=2,triangle_chunk=37,ray_chunk=7)
    expected=np.stack([ray_depth(q,reference['surface_faces'],p,k,query['xs'],query['ys']) for q,p in zip(positions,poses)])
    np.testing.assert_array_equal(actual.visible,np.isfinite(expected))
    np.testing.assert_allclose(actual.depth_m,expected,atol=2e-13,rtol=1e-12)
