"""Energy/tangent/adjoint checks independent of benchmark target trajectories."""
import numpy as np
import torch
from scripts.external_benchmark.fem import ExternalFEM
from scripts.external_benchmark.data import rigid_map, quat_rotation

def fixture():
    x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]],dtype=float)*.1
    return dict(nodes=x,attachment_weights=np.ones((1,5)),controller_targets=x[None,None]+np.array([0,0,.005]),attachment_active=np.ones((1,1)),floor=-.1)

def test_energy_gradient_and_hessian():
    m=ExternalFEM(fixture(),device='cpu');th=m.t([-5,2,3,.1,.2]);x=(m.rest+m.t([.02,-.01,.03])).requires_grad_()
    E,r,H=m.evaluate(x,th,0,True)
    automatic,=torch.autograd.grad(E,x,create_graph=True)
    assert torch.allclose(r,automatic,atol=1e-10,rtol=1e-9)
    h=torch.autograd.functional.hessian(lambda z:m.evaluate(z,th,0)[0],x).reshape(m.ndof,m.ndof)
    assert torch.allclose(H,h,atol=1e-10,rtol=1e-9)

def test_equilibrium_adjoint():
    m=ExternalFEM(fixture(),device='cpu');th=m.t([-5,2,3,.1,.2]).requires_grad_()
    x=m.predict(th,0);loss=(x*x).sum();g,=torch.autograd.grad(loss,th)
    fd=[]
    for i in range(len(th)):
        h=torch.zeros_like(th);h[i]=1e-4
        a=m.solve(th.detach()+h,0)[0];b=m.solve(th.detach()-h,0)[0]
        fd.append(float(((a*a).sum()-(b*b).sum())/2e-4))
    assert np.allclose(g.detach().numpy(),fd,atol=1e-6,rtol=.005)

def test_rigid_transport_and_quaternion_order():
    R=quat_rotation(np.array([np.sqrt(.5),0,0,np.sqrt(.5)]))
    assert np.allclose(np.array([1,0,0])@R.T,[0,1,0])
    x=fixture()['nodes'];y=x@R.T+np.array([.2,-.1,.4])
    rotation,translation=rigid_map(x,y)
    assert np.allclose(x@rotation+translation,y)

def test_compliant_contact_tangent():
    data=fixture();data['floor']=0.
    m=ExternalFEM(data,device='cpu',contact=True)
    th=m.t([-5,2,3,.1,.2,1.]);x=(m.rest+m.t([.02,-.01,.03])).requires_grad_()
    E,r,H=m.evaluate(x,th,0,True)
    automatic,=torch.autograd.grad(E,x,create_graph=True)
    h=torch.autograd.functional.hessian(lambda z:m.evaluate(z,th,0)[0],x).reshape(m.ndof,m.ndof)
    assert torch.allclose(r,automatic,atol=1e-10,rtol=1e-9)
    assert torch.allclose(H,h,atol=1e-10,rtol=1e-9)


def test_inertia_and_material_partition_tangent():
    m=ExternalFEM(fixture(),device='cpu',materials=3)
    th=m.t([-5,2,3,.1,-.2,.3,.2])
    m.inertial_target=m.rest+.01;m.inertial_dt=.02
    x=(m.rest+m.t([.02,-.01,.03])).requires_grad_()
    E,r,H=m.evaluate(x,th,0,True)
    automatic,=torch.autograd.grad(E,x,create_graph=True)
    h=torch.autograd.functional.hessian(lambda z:m.evaluate(z,th,0)[0],x).reshape(m.ndof,m.ndof)
    assert torch.allclose(r,automatic,atol=1e-10,rtol=1e-9)
    assert torch.allclose(H,h,atol=1e-10,rtol=1e-9)


def test_release_has_physical_free_fall_acceleration():
    data=fixture();data['attachment_active'][:]=0
    m=ExternalFEM(data,device='cpu',field=False)
    theta=m.t([-2,2,3,0,0]);dt=.001
    m.inertial_target=m.rest.clone();m.inertial_dt=dt
    x,_,diagnostic=m.solve(theta,0)
    change=((x-m.rest)*m.mass[:,None]).sum(0).numpy()*m.scale
    assert np.allclose(change,[0,0,-9.81*dt**2],atol=2e-10,rtol=1e-5)
    assert diagnostic['residual']<1e-7


def test_observation_projection_preserves_affine_motion():
    from scripts.external_benchmark.data import interpolate_matrix
    nodes=fixture()['nodes'];points=np.array([nodes.mean(0),nodes[:4].mean(0)])
    A=interpolate_matrix(nodes,points)
    transform=np.array([[1.1,.2,0],[.1,.8,-.1],[0,.1,1.2]])
    displacement=nodes@transform+np.array([.04,-.02,.1])-nodes
    assert np.allclose(points+A@displacement,points@transform+[.04,-.02,.1])


def test_cpu_gpu_equilibrium_and_adjoint_agree():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    results=[]
    for device in ['cpu','cuda:0']:
        model=ExternalFEM(fixture(),device=device)
        theta=model.t([-5,2,3,.1,.2]).requires_grad_()
        prediction=model.predict(theta,0)
        gradient,=torch.autograd.grad((prediction**2).sum(),theta)
        results.append((prediction.detach().cpu().numpy(),gradient.detach().cpu().numpy()))
    assert np.allclose(results[0][0],results[1][0],rtol=1e-7,atol=1e-9)
    assert np.allclose(results[0][1],results[1][1],rtol=1e-6,atol=1e-9)


def test_positive_volume_extension_has_consistent_tangent():
    from scripts.external_benchmark.fem_positive_volume import PositiveVolumeFEM
    m=PositiveVolumeFEM(fixture(),device='cpu');theta=m.t([-5,2,3,.1,.2])
    x=(m.rest*.2).requires_grad_()
    assert 0<m.min_jacobian(x)<.03
    energy,residual,hessian=m.evaluate(x,theta,0,True)
    gradient,=torch.autograd.grad(energy,x,create_graph=True)
    automatic=torch.autograd.functional.hessian(lambda y:m.evaluate(y,theta,0)[0],x).reshape(m.ndof,m.ndof)
    assert torch.allclose(residual,gradient,atol=1e-9,rtol=1e-8)
    assert torch.allclose(hessian,automatic,atol=1e-8,rtol=1e-8)
