import numpy as np
import pytest
import torch
from scripts.external_benchmark.fem import ExternalFEM
from scripts.gaussian_boundary_flow_v02.object_prestrain import InitialPrestrainFEM
from scripts.gaussian_boundary_flow_v02.boundary_preload import (
    attachment_preload_force,PreloadedExternalFEM,InitialPrestrainPreloadFEM)


def payload():
    nodes=np.array(list(__import__('itertools').product((-.1,.1),repeat=3)),float)
    weights=np.exp(-((nodes[:,2]-.1)/.05)**2)[None]
    return {'nodes':nodes,'attachment_weights':weights,
            'controller_targets':np.tile(nodes,(2,1,1,1)),
            'attachment_active':np.array([[1.],[0.]]),'floor':-.1}


def test_kernel_resultant_and_torque_match_applied_boundary_vectors():
    dtype=torch.float64;torch.manual_seed(1)
    x=torch.randn(7,3,dtype=dtype);w=torch.rand(2,7,dtype=dtype)
    a=torch.tensor([1.,.4],dtype=dtype);b=torch.tensor([[1.,2.,3.],[-2.,.5,1.]],dtype=dtype)
    f=attachment_preload_force(w,a,b)
    assert torch.allclose(f.sum(0),(a[:,None]*b).sum(0),atol=1e-12)
    centroid=(w/w.sum(-1,keepdim=True))@x
    expected=(a[:,None]*torch.linalg.cross(centroid,b)).sum(0)
    assert torch.allclose(torch.linalg.cross(x,f).sum(0),expected,atol=1e-12)


def test_task_rotation_and_world_coordinate_covariance():
    dtype=torch.float64;w=torch.tensor([[1.,2.,3.]],dtype=dtype);a=torch.ones(1,dtype=dtype)
    b=torch.tensor([[2.,-1.,.5]],dtype=dtype)
    R=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=dtype)
    world=attachment_preload_force(w,a,b)
    task=attachment_preload_force(w,a,b,rotations=R[None])
    assert torch.allclose(task,world@R.T,atol=1e-12)
    x=torch.randn(3,3,dtype=dtype);rest=torch.randn_like(x)
    assert torch.allclose((world*(x-rest)).sum(),(task*((x-rest)@R.T)).sum(),atol=1e-12)


def test_zero_preload_recovers_original_and_combined_prestrain_models():
    data=payload();base=ExternalFEM(data,device='cpu',field=False,contact=False)
    added=PreloadedExternalFEM(data,device='cpu',field=False,contact=False)
    theta=torch.zeros(added.full_parameter_count,dtype=base.dtype);x=base.rest*1.02
    for a,b in zip(base.evaluate(x,theta[:added.base_parameter_count],0,True),added.evaluate(x,theta,0,True)):
        assert torch.equal(a,b)
    natural=InitialPrestrainFEM(data,device='cpu',field=False,contact=False)
    combined=InitialPrestrainPreloadFEM(data,device='cpu',field=False,contact=False)
    theta=torch.zeros(combined.full_parameter_count,dtype=base.dtype);theta[combined.i_prestrain]=.7
    for a,b in zip(natural.evaluate(x,theta[:combined.base_parameter_count],0,True),combined.evaluate(x,theta,0,True)):
        assert torch.equal(a,b)


def test_preload_energy_residual_and_tangent_are_consistent():
    m=PreloadedExternalFEM(payload(),device='cpu',field=False,contact=False)
    theta=torch.zeros(m.full_parameter_count,dtype=m.dtype);theta[-3:]=m.t([.2,-.1,.5]);theta.requires_grad_(True)
    x=(m.rest*1.01).requires_grad_(True)
    E,r,H=m.evaluate(x,theta,0,True)
    auto=torch.autograd.grad(E,x,create_graph=True)[0]
    autoH=torch.autograd.functional.jacobian(lambda q:m.evaluate(q,theta,0)[1],x).reshape(m.ndof,m.ndof)
    assert torch.allclose(r,auto,atol=1e-12,rtol=1e-10)
    assert torch.allclose(H,autoH,atol=1e-12,rtol=1e-10)
    g=torch.autograd.grad(E,theta)[0][-3:]
    expected=-((m.weights/m.weights.sum(-1,keepdim=True))@(x-m.rest))[0]
    assert torch.allclose(g,expected,atol=1e-12)


def test_inactive_attachment_removes_preload_and_preserves_gravity():
    m=InitialPrestrainPreloadFEM(payload(),device='cpu',field=False,contact=False)
    theta=torch.zeros(m.full_parameter_count,dtype=m.dtype);theta[m.i_prestrain]=.5;theta[-3:]=m.t([3.,-4.,1.]);theta[m.i_floor]=-torch.inf
    assert m.preload_force(theta,1).abs().max()==0
    _,r,_=m.evaluate(m.rest,theta,1)
    assert torch.allclose(r.sum(0),m.t([0.,0.,1.]),atol=1e-12)


def test_localized_preload_can_balance_net_gravity_without_erasing_local_residual():
    m=InitialPrestrainPreloadFEM(payload(),device='cpu',field=False,contact=False)
    theta=torch.zeros(m.full_parameter_count,dtype=m.dtype);theta[-1]=1.
    _,r,_=m.evaluate(m.rest,theta,0)
    assert torch.allclose(r.sum(0),torch.zeros(3,dtype=m.dtype),atol=1e-12)
    assert float(r.norm())>1e-3  # Net balance does not claim exact nodal equilibrium.
    _,released,_=m.evaluate(m.rest,theta,1)
    assert torch.allclose(released.sum(0),m.t([0.,0.,1.]),atol=1e-12)


def test_active_zero_support_is_rejected_and_inactive_zero_support_is_safe():
    w=torch.zeros(1,5,dtype=torch.float64);b=torch.ones(1,3,dtype=w.dtype)
    with pytest.raises(ValueError):attachment_preload_force(w,torch.ones(1,dtype=w.dtype),b)
    assert attachment_preload_force(w,torch.zeros(1,dtype=w.dtype),b).abs().max()==0


def test_declared_task_frame_validates_rotation_and_composes_with_prestrain():
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    m=InitialPrestrainPreloadFEM(payload(),device='cpu',field=False,contact=False,preload_frame='task',preload_rotations=R[None])
    theta=torch.zeros(m.full_parameter_count,dtype=m.dtype);theta[-3]=1
    assert torch.allclose(m.preload_force(theta,0).sum(0),m.t([0.,1.,0.]))
    with pytest.raises(ValueError):PreloadedExternalFEM(payload(),device='cpu',preload_frame='task',preload_rotations=np.zeros((1,3,3)))


def test_mixins_compose_with_dynamic_gauge_removal():
    from scripts.external_benchmark.dynamic_fem import DynamicFEM
    from scripts.gaussian_boundary_flow_v02.boundary_preload import AttachmentPreloadMixin
    class Combined(AttachmentPreloadMixin,InitialPrestrainFEM,DynamicFEM):
        pass
    data=payload();combined=Combined(data,device='cpu',field=False,contact=False)
    dynamic=DynamicFEM(data,device='cpu',field=False,contact=False)
    x=combined.rest*1.01
    for m in (combined,dynamic):m.inertial_target=m.rest.clone();m.inertial_dt=.01
    theta=torch.zeros(combined.full_parameter_count,dtype=combined.dtype)
    for a,b in zip(combined.evaluate(x,theta,0,True),dynamic.evaluate(x,theta[:combined.i_prestrain],0,True)):
        assert torch.allclose(a,b,atol=1e-12,rtol=1e-11)
