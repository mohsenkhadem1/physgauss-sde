"""Independent numerical audit: conservation/work, units and documented limits."""
import pytest
import torch
from physgauss.gaussian_boundary import LinearBoundarySystem
from physgauss.boundary_program import BoundaryProgram
from scripts.gaussian_boundary_flow.model import Block
from physgauss.physics import material_response


def test_backward_euler_work_balance_with_moving_constraint():
    torch.manual_seed(46);dtype=torch.float64
    A=torch.randn(2,6,6,dtype=dtype);K=A.transpose(-1,-2)@A
    A=torch.randn(2,6,6,dtype=dtype);D=A.transpose(-1,-2)@A*.1
    mass=torch.rand(2,6,dtype=dtype)+.5;dt=.01
    fixed=torch.tensor([True,False,False,False,False,True])
    system=LinearBoundarySystem(mass,K,D,dt,fixed)
    u=torch.randn(2,6,dtype=dtype)*.1;v=torch.randn_like(u)*.1;force=torch.randn_like(u)
    target=u+torch.randn_like(u)*.01
    un,vn,reaction=system.step(u,v,force,target)
    def energy(q,w):return .5*(mass*w*w).sum(-1)+.5*torch.einsum('bi,bij,bj->b',q,K,q)
    balance=energy(un,vn)-energy(u,v)
    expected=dt*((force+reaction)*vn).sum(-1)-dt*torch.einsum('bi,bij,bj->b',vn,D,vn)-.5*(mass*(vn-v)**2).sum(-1)-.5*torch.einsum('bi,bij,bj->b',un-u,K,un-u)
    assert torch.allclose(balance,expected,atol=1e-12,rtol=1e-12)
    assert reaction[:,~fixed].abs().max()<1e-12


def test_integrated_load_resultant_and_gravity_units():
    b=Block(dtype=torch.float64);z=torch.tensor([[-5.,0.,5.,-3.,2.,1.],[0.,1.,-2.,3.,-4.,2.]],dtype=torch.float64)
    _,load,gravity=b.operators(z,.005);p=b.parameters(z)
    assert torch.allclose(load.reshape(2,8,3).sum(1).norm(dim=-1),p[:,2].exp(),rtol=1e-12,atol=1e-12)
    expected=gravity.new_tensor([0.,-.5*9.81,0.]).expand(2,-1)
    assert torch.allclose(gravity.reshape(2,8,3).sum(1),expected,atol=1e-12)


def test_linear_model_is_not_finite_rotation_objective_documented_limit():
    b=Block(dtype=torch.float64);theta=torch.tensor(.6,dtype=torch.float64)
    R=torch.stack([torch.stack([theta.cos(),-theta.sin(),theta*0]),torch.stack([theta.sin(),theta.cos(),theta*0]),theta.new_tensor([0.,0.,1.])])
    x=b.mesh.rest@R.T;u=(x-b.mesh.rest).flatten()
    linear=.5*u@b.K@u
    _,nonlinear,_=material_response(x,b.mesh,theta.new_tensor(1.))
    assert abs(float(nonlinear))<1e-14
    assert float(linear)>1e-7


def test_boundary_program_rejects_nan_query():
    p=BoundaryProgram(torch.tensor([0.,1.]),torch.zeros(1,2,3))
    with pytest.raises(ValueError):p.at(float('nan'))


def test_boundary_program_rejects_nan_activation():
    with pytest.raises(ValueError):BoundaryProgram(torch.tensor([0.,1.]),torch.zeros(1,2,3),active_start=float('nan'))


def test_common_mass_stiffness_force_scale_is_motion_unobservable():
    dtype=torch.float64;m=torch.tensor([[1.,2.,3.]],dtype=dtype);K=torch.diag_embed(m*8);D=torch.diag_embed(m*.3);f=m.new_tensor([[.2,-1.,.7]])
    a=LinearBoundarySystem(m,K,D,.01);b=LinearBoundarySystem(7*m,7*K,7*D,.01)
    u=torch.zeros_like(m);v=u.clone();us=u.clone();vs=v.clone()
    for _ in range(30):
        u,v,_=a.step(u,v,f);us,vs,_=b.step(us,vs,7*f)
    assert torch.allclose(u,us,atol=1e-14,rtol=1e-12)
    assert torch.allclose(v,vs,atol=1e-14,rtol=1e-12)


def test_boundary_program_rejects_infinite_query_but_allows_default_activation():
    p=BoundaryProgram(torch.tensor([0.,1.]),torch.zeros(1,2,3))
    assert p.at(.5)[2]==1
    for t in (float('inf'),float('-inf')):
        with pytest.raises(ValueError):p.at(t)
    with pytest.raises(ValueError):BoundaryProgram(torch.tensor([0.,1.]),torch.zeros(1,2,3),active_end=float('nan'))
