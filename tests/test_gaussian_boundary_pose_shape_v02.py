import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.pose_shape import predict


def fixture():
    nodes=np.array([[0.,0.,0.],[1,0,0],[0,1,0],[0,0,1]])*.02
    T=8; end=4; shifts=np.arange(T)[:,None]*np.array([[.001,0,0]])
    observed=nodes[None]+shifts[:end,None]+np.array([.03,.02,.1])
    data=dict(nodes=nodes,times=np.arange(T)*.1,train_end=end,observed=observed,
              observed_initial=nodes.copy(),observed_mask=np.ones((end,4),bool),
              controller_targets=(nodes[None]+shifts[:,None])[:,None],attachment_active=np.ones((T,1)))
    raw=np.repeat((nodes[None]+shifts[:,None])[None],2,axis=0)
    return data,raw


def test_rigid_shape_invariance_and_shared_sample_transform():
    data,raw=fixture(); out,meta=predict(data,raw,prefix_frames=[0,2,3],point_ids=[0,1,2,3])
    assert np.allclose(out[:,3:],raw[:,3:]+[.03,.02,.1])
    assert np.array_equal(out[:,:3],raw[:,:3])
    for t in range(3,8):
        assert np.allclose(np.linalg.norm(out[0,t,:,None]-out[0,t,None,:],axis=-1),
                           np.linalg.norm(raw[0,t,:,None]-raw[0,t,None,:],axis=-1))
    assert meta['prescribed_release_frame'] is None


def test_unselected_prefix_no_effect_and_suffix_rejected():
    data,raw=fixture(); args=dict(prefix_frames=[0,2,3],point_ids=[0,1,2])
    first,_=predict(data,raw,**args)
    data['observed'][1]+=200; data['observed'][:,3]+=100
    second,_=predict(data,raw,**args)
    assert np.array_equal(first,second)
    data['observed']=np.concatenate([data['observed'],data['observed'][-1:]])
    with pytest.raises(ValueError,match='Only prefix'):predict(data,raw,**args)


def test_prescribed_release_ballistic_no_inferred_contact():
    data,raw=fixture(); data['attachment_active'][5:]=0
    out,meta=predict(data,raw,prefix_frames=[0,2,3],point_ids=[0,1,2,3])
    assert meta['prescribed_release_frame']==5 and meta['inferred_release_frame'] is None
    assert np.allclose(out[0,5].mean(0)-out[0,4].mean(0),[.001,0,-.5*9.81*.1**2])
    assert np.allclose(out[0,6].mean(0)-out[0,4].mean(0),[.002,0,-.5*9.81*.2**2])


def test_reject_privileged_fields():
    data,raw=fixture(); data['evaluation']={}
    with pytest.raises(ValueError,match='forbidden'):predict(data,raw,prefix_frames=[0,2,3],point_ids=[0,1,2])


def test_held_prediction_equivariant_to_world_rigid_change():
    from scipy.spatial.transform import Rotation
    data,raw=fixture(); args=dict(prefix_frames=[0,2,3],point_ids=[0,1,2,3])
    first,_=predict(data,raw,**args)
    R=Rotation.from_rotvec([.3,-.2,.4]).as_matrix(); shift=np.array([2.,-1.,.5])
    for key in ['nodes','observed','observed_initial','controller_targets']:
        data[key]=data[key]@R+shift
    second,_=predict(data,raw@R+shift,**args)
    assert np.allclose(second,first@R+shift,atol=1e-10)
