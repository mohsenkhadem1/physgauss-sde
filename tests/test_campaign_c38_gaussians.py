import numpy as np
import pytest
import torch
from experiments.campaign_c38_gaussians import P2MaterialGaussians,p2_shapes,p2_gradients,EDGES
from experiments.campaign_static_p2_block import enrich,shapes,shape_gradients

def geometry():
    x=np.array([[0.,0,0],[.2,0,0],[0,.1,0],[0,0,.08]])
    X,E=enrich(x,np.array([[0,1,2,3]]))
    return torch.tensor(X,dtype=torch.float64),torch.tensor(E)

def test_reference_and_rigid_objectivity_mass_and_SPD():
    X,E=geometry();m=P2MaterialGaussians.from_elements(X,E,625.);s=m.transport(X)
    torch.testing.assert_close(s.centers,m.reference_centers);torch.testing.assert_close(s.deformation_gradients,torch.eye(3,dtype=X.dtype)[None])
    theta=.6;R=torch.tensor([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]],dtype=X.dtype)
    q=X@R.T+X.new_tensor([.3,-.2,.1]);moved=m.transport(q)
    torch.testing.assert_close(moved.deformation_gradients,R[None]);torch.testing.assert_close(moved.covariances,R@m.reference_covariances@R.T)
    torch.testing.assert_close(moved.total_mass,m.total_mass if hasattr(m,'total_mass') else m.masses.sum())
    assert torch.linalg.eigvalsh(moved.covariances).min()>0
    torch.testing.assert_close(moved.bulk_density*moved.volumes,moved.masses)

def test_exact_quadratic_center_and_gradient_not_P1_shortcut():
    X,E=geometry();m=P2MaterialGaussians.from_elements(X,E,625.)
    q=torch.stack([X[:,0]+.5*X[:,0]**2,X[:,1]+.8*X[:,0]*X[:,1],X[:,2]],-1)
    s=m.transport(q);c=m.reference_centers[0]
    expected=torch.stack([c[0]+.5*c[0]**2,c[1]+.8*c[0]*c[1],c[2]])
    torch.testing.assert_close(s.centers[0],expected)
    F=torch.tensor([[1+c[0],0,0],[.8*c[1],1+.8*c[0],0],[0,0,1]],dtype=X.dtype)
    torch.testing.assert_close(s.deformation_gradients[0],F)
    assert not torch.allclose(s.centers,q[E[:,:4]].mean(1),atol=1e-8,rtol=0)

def test_numpy_P2_ordering_batch_and_differentiability():
    X,E=geometry();b=torch.tensor([[.1,.2,.3,.4]],dtype=X.dtype);g=torch.randn(1,4,3,dtype=X.dtype)
    np.testing.assert_allclose(p2_shapes(b),shapes(b.numpy()))
    np.testing.assert_allclose(p2_gradients(b,g),shape_gradients(b.numpy(),g.numpy()))
    model=P2MaterialGaussians.from_elements(X,E,625.);q=torch.stack([X,X*1.02]).requires_grad_();state=model.transport(q)
    assert state.centers.shape==(2,1,3) and state.covariances.shape==(2,1,3,3)
    (state.centers.square().sum()+state.covariances.square().sum()).backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum()>0

def test_invalid_midpoints_and_inverted_local_field_fail():
    X,E=geometry();wrong=X.clone();wrong[E[0,4],0]+=.01
    with pytest.raises(ValueError,match='midpoint'):P2MaterialGaussians.from_elements(wrong,E,625.)
    model=P2MaterialGaussians.from_elements(X,E,625.);q=X.clone();q[:,0]*=-1
    with pytest.raises(ValueError,match='Jacobian'):model.transport(q)
