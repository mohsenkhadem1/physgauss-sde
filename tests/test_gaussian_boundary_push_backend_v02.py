import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.push_backend import estimate_plane


def fixture():
    controls=np.zeros((8,4,3));controls[:,:,2]=np.linspace(.2,.1,8)[:,None]
    return dict(train_end=5,observed=np.zeros((5,4,3)),observed_mask=np.ones((5,4),bool),
                nodes=np.array([[0.,0.,0.],[.01,0,0],[0,.01,0],[0,0,.01]]),controls=controls)


def test_prefix_only_axis_and_future_action_origin():
    data=fixture();q,n,info=estimate_plane(data,[0,2,4])
    assert np.allclose(n[0,0],[0,0,-1])
    data['controls'][5:]+=10;data['observed'][1]+=100
    q2,n2,info2=estimate_plane(data,[0,2,4])
    assert np.array_equal(n,n2) and info==info2
    assert np.array_equal(q[:5],q2[:5]) and not np.array_equal(q[5:],q2[5:])
    shifted,_,_=estimate_plane(data,[0,2,4],standoff_m=.01)
    assert np.allclose(shifted-q2,.01*n)


def test_degenerate_or_outward_fails():
    data=fixture();data['controls'][:5]=data['controls'][0]
    with pytest.raises(ValueError,match='insufficient'):estimate_plane(data,[0,2,4])
    data=fixture();data['controls'][:5]=data['controls'][:5][::-1]
    with pytest.raises(ValueError,match='unsupported'):estimate_plane(data,[0,2,4])


def test_no_suffix_objects_or_target_fields():
    data=fixture();data['observed']=np.zeros((8,4,3))
    with pytest.raises(ValueError,match='Only prefix'):estimate_plane(data,[0,2,4])
    data=fixture();data['evaluation']={}
    with pytest.raises(ValueError,match='forbidden'):estimate_plane(data,[0,2,4])
