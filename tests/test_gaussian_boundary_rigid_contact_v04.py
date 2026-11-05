import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.rigid_contact_baseline import predict_initial_shape


def fixture():
    nodes=np.array([[0.,0.,0.],[.02,0,0],[0,.02,0],[0,0,.02]])
    T=9;end=4;shift=np.arange(T)[:,None]*np.array([[.001,0,0]])
    data=dict(nodes=nodes,times=np.arange(T)*.05,train_end=end,observed=nodes[None]+shift[:end,None]+[0,0,.04],
              observed_initial=nodes,observed_mask=np.ones((end,4),bool),
              controller_targets=(nodes[None]+shift[:,None])[:,None],attachment_active=np.ones((T,1)),floor=0.)
    return data,dict(prefix_frames=[0,2,3],point_ids=[0,1,2])


def test_rigid_initial_shape_preserved_and_no_posterior_axis():
    data,plan=fixture();data['attachment_active'][5:]=0
    out,meta=predict_initial_shape(data,plan)
    assert out.shape==(9,4,3) and out.min(axis=(0,1))[2]>=-1e-12
    distances=lambda x:np.linalg.norm(x[:,None]-x[None,:],axis=-1)
    for frame in out:assert np.allclose(distances(frame),distances(data['nodes']))
    assert meta['prescribed_release_frame']==5


def test_same_observation_budget_and_no_target_fields():
    data,plan=fixture();first,_=predict_initial_shape(data,plan)
    data['observed'][1]+=300;data['observed'][:,3]+=100
    second,_=predict_initial_shape(data,plan)
    assert np.array_equal(first,second)
    data['evaluation']={}
    with pytest.raises(ValueError,match='forbidden'):predict_initial_shape(data,plan)
