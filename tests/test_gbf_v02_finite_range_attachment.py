import torch
from scripts.gaussian_boundary_flow_v02.finite_range_attachment import gaussian_bond


def test_energy_force_tangent_and_finite_range():
    d=torch.tensor([[[.1,.05,-.02],[.4,0.,0.]]],dtype=torch.float64,requires_grad=True)
    w=torch.tensor([[2.,3.]],dtype=d.dtype);s=torch.tensor([[.2]],dtype=d.dtype)
    U,r,H=gaussian_bond(d,w,s);g=torch.autograd.grad(U,d,create_graph=True)[0]
    assert torch.allclose(g,r,atol=1e-12)
    for i in range(3):
        row=torch.autograd.grad(g[...,i].sum(),d,retain_graph=True)[0]
        assert torch.allclose(row,H[...,i,:],atol=1e-12)
    _,far,_=gaussian_bond(d.detach()*100,w,s)
    assert far.abs().max()<1e-100
    assert (r*d).sum()>0


def test_rotation_covariance_and_deactivation():
    d=torch.randn(2,5,3,dtype=torch.float64);w=torch.ones(2,5,dtype=d.dtype);s=torch.ones(2,1,dtype=d.dtype)
    R=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=d.dtype)
    U,r,H=gaussian_bond(d,w,s);Ur,rr,Hr=gaussian_bond(d@R.T,w,s)
    assert torch.allclose(U,Ur)
    assert torch.allclose(rr,r@R.T)
    assert torch.allclose(Hr,R@H@R.T)
    U0,r0,H0=gaussian_bond(d,w*0,s)
    assert U0==0 and r0.abs().max()==0 and H0.abs().max()==0
