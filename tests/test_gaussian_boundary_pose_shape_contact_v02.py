import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.pose_shape_contact import predict


def fixture():
    nodes=np.array([[0.,0.,0.],[1,0,0],[0,1,0],[0,0,1]])*.02
    T=12;end=4;shift=np.arange(T)[:,None]*np.array([[.001,0,0]])
    obs=nodes[None]+shift[:end,None]+[0,0,.05]
    data=dict(nodes=nodes,times=np.arange(T)*.05,train_end=end,observed=obs,observed_initial=nodes,
              observed_mask=np.ones((end,4),bool),controller_targets=(nodes[None]+shift[:,None])[:,None],
              attachment_active=np.ones((T,1)),floor=0.)
    raw=np.repeat((nodes[None]+shift[:,None])[None],2,axis=0)
    return data,raw


def test_release_floor_nonpenetration_and_shape_preserved():
    data,raw=fixture();data['attachment_active'][5:]=0
    out,meta=predict(data,raw,prefix_frames=[0,2,3],point_ids=[0,1,2,3])
    assert out[:,:, :,2].min()>=-1e-12
    assert meta['prescribed_release_frame']==5 and meta['inferred_release_frame'] is None
    assert meta['contact_diagnostics'][0]['contact_substeps']>0
    distances=lambda x:np.linalg.norm(x[:,None]-x[None,:],axis=-1)
    assert np.allclose(distances(out[0,-1]),distances(out[0,4]))


def test_held_projection_and_no_extra_observation_access():
    data,raw=fixture();data['floor']=.08
    args=dict(prefix_frames=[0,2,3],point_ids=[0,1,2])
    first,meta=predict(data,raw,**args)
    assert first[:,3:,:,2].min()>=.08-1e-12
    assert meta['contact_diagnostics'][0]['maximum_held_vertical_projection_m']>0
    data['observed'][1]+=100;data['observed'][:,3]+=20
    second,_=predict(data,raw,**args)
    assert np.array_equal(first,second)
    data['evaluation']={}
    with pytest.raises(ValueError,match='forbidden'):predict(data,raw,**args)
