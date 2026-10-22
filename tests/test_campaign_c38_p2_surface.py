import numpy as np
import torch
from experiments.campaign_c38_p2_surface import quadratic_boundary_faces,bind_surface,checked_p2_depth
from experiments.campaign_static_p2_block import enrich


def geometry():
    X,e=enrich(np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]),np.array([[0,1,2,3]]));f=quadratic_boundary_faces(e,np.array([[0,1,2],[0,3,1]]));return X,e,f


def test_true_midnodes_interpolation_shared_edges_and_autograd():
    X,e,f=geometry();b=bind_surface(f,4);q=torch.tensor(X,dtype=torch.float64,requires_grad=True);q2=q.clone();q2[f[0,3:],2]+=.002
    rendered=b.evaluate(q2);assert float(rendered[:,2].max().detach())>=.002
    np.testing.assert_allclose(b.weights.sum(1),1,atol=1e-16)
    # Two triangulated faces share all edge samples; no duplicate position nodes.
    assert len(b.node_ids)==2*15-5
    rendered.square().sum().backward();assert torch.isfinite(q.grad).all() and q.grad.abs().sum()>0
    np.testing.assert_array_equal(quadratic_boundary_faces(e,f),f)


def test_curved_depth_level_failure_and_controlled_refinement():
    X,e,f=geometry();f=f[:1];q=torch.tensor(X,dtype=torch.float64);q[f[0,3:],2]=.002
    origin=torch.tensor([.23,.31,1.],dtype=torch.float64);direction=torch.tensor([[0.,0,-1.]],dtype=torch.float64)
    _,coarse=checked_p2_depth(q,f,origin,direction,levels=(1,2),depth_tolerance_m=1e-6)
    assert not coarse['passed']
    # 8->16 actually differs by30um: retain that failure, then test finer levels.
    _,insufficient=checked_p2_depth(q,f,origin,direction,levels=(8,16),depth_tolerance_m=2e-5)
    assert not insufficient['passed']
    result,check=checked_p2_depth(q,f,origin,direction,levels=(16,32,64),depth_tolerance_m=2e-5)
    assert check['passed'] and bool(result.visible.all())
    b=np.array([1-.23-.31,.23,.31]);exact=1-4*.002*(b[0]*b[1]+b[0]*b[2]+b[1]*b[2])
    assert abs(float(result.depth_m[0])-exact)<4e-6
