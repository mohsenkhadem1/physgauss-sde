import numpy as np
from .transport import valid_anchors,setup,warp

def test_incoming_motion_mask_at_last_frame():
    raw=dict(object_points=np.zeros((3,20,3)),object_visibilities=np.ones((3,20),bool),
             object_motions_valid=np.array([[True]*20,[True]*20,[False]*20]))
    assert valid_anchors(raw,2).sum()==20
    raw['object_motions_valid'][1,4]=False
    assert not valid_anchors(raw,2)[4]

def test_identity_affine_and_rigid_preserve_gaussians():
    rng=np.random.default_rng(21);initial=rng.normal(size=(40,3));xyz=rng.normal(size=(10,3))
    scales=np.full((10,3),.01);quats=np.tile([1.,0,0,0],(10,1));binding=setup(initial,xyz)
    for affine,covariance in ((False,False),(True,False),(True,True)):
        x,s,q,_=warp(initial,initial,xyz,scales,quats,binding,affine,covariance)
        np.testing.assert_allclose(x,xyz,atol=1e-10);np.testing.assert_allclose(s,scales,atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(q,axis=1),1.,atol=1e-10)

def test_rigid_transform_does_not_stretch_footprints():
    rng=np.random.default_rng(4);initial=rng.normal(size=(40,3));xyz=rng.normal(size=(10,3))
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    scales=np.full((10,3),.01);quats=np.tile([1.,0,0,0],(10,1));binding=setup(initial,xyz)
    x,s,q,_=warp(initial,initial@R+2,xyz,scales,quats,binding,True,True)
    np.testing.assert_allclose(x,xyz@R+2,atol=1e-10);np.testing.assert_allclose(s,scales,atol=1e-10)

def test_stretch_changes_covariance_only_when_enabled():
    rng=np.random.default_rng(8);initial=rng.normal(size=(60,3));xyz=initial[:10].copy()
    scales=np.full((10,3),.01);quats=np.tile([1.,0,0,0],(10,1));binding=setup(initial,xyz)
    current=initial*np.array([1.8,1,1])
    _,fixed,_,_=warp(initial,current,xyz,scales,quats,binding,True,False)
    _,transported,_,_=warp(initial,current,xyz,scales,quats,binding,True,True)
    np.testing.assert_array_equal(fixed,scales)
    assert np.all(transported[:,0]>.015)
