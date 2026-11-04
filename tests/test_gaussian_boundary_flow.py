import torch
from physgauss.physics import cube_mesh,material_response
from physgauss.gaussian_boundary import (surface_quadrature,gaussian_gate,
    robin_matrices,traction_force,LinearBoundarySystem,plane_impact)


def test_surface_force_and_moment_are_consistent():
    mesh=cube_mesh(side=.1,density=500)
    p,a,N=surface_quadrature(mesh)
    assert torch.allclose(a.sum(),a.new_tensor(.06))
    g=torch.ones(1,len(a),dtype=a.dtype)
    t=a.new_tensor([[2.,-3.,4.]])
    force=traction_force(N,a,g,t)[0]
    assert torch.allclose(force.sum(0),a.sum()*t[0])
    assert torch.allclose(torch.linalg.cross(mesh.rest,force).sum(0),
                          (torch.linalg.cross(p,t.expand_as(p))*a[:,None]).sum(0))


def test_robin_energy_gradient_and_dissipation():
    mesh=cube_mesh();p,a,N=surface_quadrature(mesh)
    g=gaussian_gate(p,p.mean(0)[None],torch.eye(3,dtype=p.dtype)[None])
    mat=torch.diag(p.new_tensor([1.,2.,3.]))[None]
    K,D=robin_matrices(N,a,g,mat,mat*.3)
    u=torch.randn(1,24,dtype=p.dtype,requires_grad=True)
    U=.5*torch.einsum('bi,bij,bj->',u,K,u)
    grad=torch.autograd.grad(U,u)[0]
    assert torch.allclose(grad,torch.einsum('bij,bj->bi',K,u))
    assert torch.linalg.eigvalsh(K).min()>-1e-12
    assert torch.einsum('bi,bij,bj->',u,D,u)>0


def test_task_frame_equivariance_and_locality():
    p=torch.randn(9,3,dtype=torch.float64)
    c=torch.zeros(1,3,dtype=p.dtype);cov=torch.diag(p.new_tensor([.1,.2,.3]))[None]
    R=p.new_tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    b=p.new_tensor([2.,3.,4.])
    assert torch.allclose(gaussian_gate(p,c,cov),gaussian_gate(p@R.T+b,c,cov,rotation=R,translation=b))
    assert gaussian_gate(p+100,c,cov).max()<1e-20


def test_exact_moving_constraint_and_reaction_balance():
    K=torch.eye(3,dtype=torch.float64)[None]*2
    m=torch.ones(1,3,dtype=K.dtype)
    system=LinearBoundarySystem(m,K,K*.1,.01,torch.tensor([True,False,False]))
    u=torch.zeros_like(m);v=u.clone();f=u.clone();f[:,1]=2
    target=u.clone();target[:,0]=.03
    un,vn,r=system.step(u,v,f,target)
    assert torch.allclose(un[:,0],target[:,0])
    assert r[:,1:].abs().max()<1e-12
    assert r[:,0].abs().min()>0


def test_passive_backward_euler_energy_decreases():
    K=torch.eye(3,dtype=torch.float64)[None]*10
    m=torch.ones(1,3,dtype=K.dtype);s=LinearBoundarySystem(m,K,K*.03,.01)
    u=torch.ones_like(m);v=torch.zeros_like(m)
    previous=float(.5*(u*(K@u[...,None]).squeeze(-1)).sum())
    for _ in range(50):
        u,v,_=s.step(u,v,torch.zeros_like(u))
        E=float(.5*(m*v*v).sum()+.5*(u*(K@u[...,None]).squeeze(-1)).sum())
        assert E<=previous+1e-12
        previous=E


def test_impact_no_attraction_and_friction_bound():
    x=torch.tensor([[0.,-.01,0.],[0.,.01,0.],[0.,0.,0.]])
    v=torch.tensor([[1.,-2.,0.],[0.,-2.,0.],[0.,1.,0.]])
    _,vn,j=plane_impact(x,v,torch.ones(3),torch.tensor([0.,1.,0.]),0.,.5,.2)
    assert torch.allclose(vn[0,1],torch.tensor(1.))
    assert j[1:].abs().max()==0
    assert abs(j[0,0])<=.2*j[0,1]+1e-7


def test_existing_elastic_tangent_has_rigid_translation_nullspace():
    from scripts.gaussian_boundary_flow.model import Block
    b=Block(dtype=torch.float64)
    for direction in torch.eye(3,dtype=torch.float64):
        assert (b.K@direction.repeat(8)).abs().max()<1e-12
    assert torch.linalg.eigvalsh(b.K).min()>-1e-12


def test_future_program_does_not_leak_into_prefix():
    from scripts.gaussian_boundary_flow.model import Block,observe
    b=Block();z=torch.zeros(2,6)
    base=b.rollout(z)
    for kind in ('release','triangle','reverse'):
        changed=b.rollout(z,kind=kind)
        assert torch.equal(observe(base,False),observe(changed,False))
        assert not torch.equal(base[:,17:],changed[:,17:])


def test_rollout_parameter_derivatives_match_finite_difference():
    from scripts.gaussian_boundary_flow.model import Block
    b=Block(dtype=torch.float64);z=torch.zeros(1,6,dtype=torch.float64,requires_grad=True)
    loss=b.rollout(z,duration=.1)[:,-1,7,1]
    grad=torch.autograd.grad(loss.sum(),z)[0]
    eps=1e-4
    for k in (0,1,2,3,4,5):
        shift=torch.zeros_like(z);shift[0,k]=eps
        fd=(b.rollout(z.detach()+shift,duration=.1)[:,-1,7,1]-b.rollout(z.detach()-shift,duration=.1)[:,-1,7,1])/(2*eps)
        assert torch.allclose(grad[:,k],fd,atol=1e-8,rtol=1e-3)


def test_flow_sampling_is_reproducible_and_has_parameter_axes():
    from scripts.gaussian_boundary_flow.flow import ConditionalFlow
    flow=ConditionalFlow(4,width=16);c=torch.zeros(2,4)
    a=flow.sample(c,3,4,torch.Generator().manual_seed(13))
    b=flow.sample(c,3,4,torch.Generator().manual_seed(13))
    assert a.shape==(2,3,6)
    assert torch.equal(a,b)


def test_temporal_program_no_silent_extrapolation_and_release():
    import pytest
    from physgauss.boundary_program import BoundaryProgram
    times=torch.tensor([0.,.5,1.]);values=torch.tensor([[[0.,0.,0.],[.1,0.,0.],[.1,0.,0.]]],requires_grad=True)
    p=BoundaryProgram(times,values,active_start=0.,active_end=.75)
    value,velocity,active=p.at(.25)
    assert torch.allclose(value,torch.tensor([[.05,0.,0.]]))
    assert torch.allclose(velocity,torch.tensor([[.2,0.,0.]]))
    assert active==1 and p.at(.8)[2]==0
    value.sum().backward();assert values.grad.abs().sum()>0
    with pytest.raises(ValueError):p.at(1.1)


def test_batched_task_frames():
    p=torch.randn(7,3,dtype=torch.float64);R=torch.eye(3,dtype=p.dtype).repeat(2,1,1)
    t=torch.tensor([[1.,2.,3.],[-3.,0.,1.]],dtype=p.dtype)
    c=torch.zeros(2,3,dtype=p.dtype);cov=R*.1
    expected=gaussian_gate(p,c,cov)
    actual=gaussian_gate(p[None]+t[:,None],c,cov,rotation=R,translation=t)
    assert torch.allclose(expected,actual)
