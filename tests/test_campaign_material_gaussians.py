"""Physical shape transport and normalized material mass; no renderer/solver."""
import math
import pytest
import torch
from experiments.campaign_material_gaussians import MaterialGaussians,SurfaceAppearanceAttachments


def mesh(dtype=torch.float64,device='cpu',two=True):
    vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]],dtype=dtype,device=device)
    tets=torch.tensor([[0,1,2,3],[4,1,2,3]] if two else [[0,1,2,3]],device=device)
    return vertices,tets


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_reference_covariance_matches_uniform_tetrahedron_and_normalized_masses(dtype):
    x,t=mesh(dtype);g=MaterialGaussians.from_tetrahedra(x,t,torch.tensor([600.,900.],dtype=dtype))
    torch.testing.assert_close(g.reference_volumes,torch.tensor([1/6,1/3],dtype=dtype))
    torch.testing.assert_close(g.masses,torch.tensor([100.,300.],dtype=dtype))
    torch.testing.assert_close(g.mass_weights.sum(),torch.tensor(1.,dtype=dtype))
    torch.testing.assert_close(g.volume_weights.sum(),torch.tensor(1.,dtype=dtype))
    expected=torch.full((3,3),-1/80,dtype=dtype);expected.diagonal().fill_(3/80)
    torch.testing.assert_close(g.reference_covariances[0],expected)
    assert bool((torch.linalg.eigvalsh(g.reference_covariances)>0).all())
    state=g.transport(x);torch.testing.assert_close(state.deformation_gradients,torch.eye(3,dtype=dtype).expand(2,3,3))
    torch.testing.assert_close(state.covariances,g.reference_covariances)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_finite_rigid_motion_objectivity_and_density_covariance(dtype):
    x,t=mesh(dtype);g=MaterialGaussians.from_tetrahedra(x,t,700.)
    a=.83;R=torch.tensor([[math.cos(a),-math.sin(a),0.],[math.sin(a),math.cos(a),0.],[0.,0.,1.]],dtype=dtype)
    shift=torch.tensor([.7,-.4,1.2],dtype=dtype);reference=g.transport(x);moved=g.transport(x@R.T+shift)
    torch.testing.assert_close(moved.centers,g.reference_centers@R.T+shift)
    torch.testing.assert_close(moved.deformation_gradients,R.expand(2,3,3),atol=2e-6 if dtype==torch.float32 else 1e-13,rtol=2e-6 if dtype==torch.float32 else 1e-13)
    torch.testing.assert_close(moved.covariances,R@g.reference_covariances@R.T)
    torch.testing.assert_close(moved.jacobians,torch.ones(2,dtype=dtype))
    queries=torch.tensor([[.2,.3,.1],[.6,.5,.4]],dtype=dtype)
    torch.testing.assert_close(moved.density(queries@R.T+shift),reference.density(queries),atol=.001 if dtype==torch.float32 else 1e-10,rtol=2e-6 if dtype==torch.float32 else 1e-12)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_exact_affine_transport_and_batch_axes(dtype):
    x,t=mesh(dtype);g=MaterialGaussians.from_tetrahedra(x,t,800.)
    F=torch.tensor([[1.2,.2,.1],[0.,.8,.1],[0.,0.,1.4]],dtype=dtype)
    shifts=torch.tensor([[0.,0.,0.],[.3,.1,-.2]],dtype=dtype)
    states=x@F.T+shifts[:,None,:];states=states[:,None].expand(2,3,len(x),3)
    s=g.transport(states);J=torch.linalg.det(F)
    assert s.centers.shape==(2,3,2,3)
    torch.testing.assert_close(s.deformation_gradients,F.expand(2,3,2,3,3))
    torch.testing.assert_close(s.covariances,(F@g.reference_covariances@F.T).expand(2,3,2,3,3))
    torch.testing.assert_close(s.volumes,(J*g.reference_volumes).expand(2,3,2))
    torch.testing.assert_close(s.bulk_density,torch.full((2,3,2),800.,dtype=dtype)/J)
    torch.testing.assert_close(s.bulk_density*s.volumes,g.masses.expand(2,3,2))
    assert bool((torch.linalg.eigvalsh(s.covariances)>0).all())
    q=torch.zeros(2,3,4,3,dtype=dtype);assert s.density(q).shape==(2,3,4)


def test_independent_gaussian_quadrature_conserves_mass_after_compression_and_shear():
    x,t=mesh(two=False);g=MaterialGaussians.from_tetrahedra(x,t,750.)
    F=torch.tensor([[.6,.3,.1],[0.,1.4,.2],[0.,0.,.8]],dtype=x.dtype)
    for s in [g.transport(x),g.transport(x@F.T+.4)]:
        z=torch.linspace(-6,6,25,dtype=x.dtype);grid=torch.stack(torch.meshgrid(z,z,z,indexing='ij'),-1)
        L=torch.linalg.cholesky(s.covariances[0]);queries=(grid@L.T+s.centers[0]).reshape(-1,3)
        rho=s.density(queries).reshape(25,25,25)*torch.linalg.det(L)
        integral=torch.trapezoid(torch.trapezoid(torch.trapezoid(rho,z,dim=-1),z,dim=-1),z,dim=-1)
        torch.testing.assert_close(integral,g.total_mass,atol=2e-5,rtol=1e-7)
        torch.testing.assert_close(s.normalized_occupancy(queries)*g.total_mass,s.density(queries))


def test_autograd_through_shared_nodal_centers_deformation_covariance_and_density():
    x,t=mesh(two=False);g=MaterialGaussians.from_tetrahedra(x,t,torch.tensor(750.,dtype=x.dtype,requires_grad=True))
    current=(x@torch.tensor([[1.1,.1,0.],[0.,.9,.1],[0.,0.,1.2]],dtype=x.dtype).T+.2).requires_grad_()
    queries=torch.tensor([[.4,.35,.3],[.6,.5,.4]],dtype=x.dtype)
    def output(vertices):
        state=g.transport(vertices)
        return torch.cat([state.centers.flatten(),state.covariances.flatten(),state.density(queries)])
    assert torch.autograd.gradcheck(output,(current,),eps=1e-6,atol=2e-5,rtol=2e-4)
    state=g.transport(current);loss=state.density(queries).sum();grad=torch.autograd.grad(loss,(current,g.reference_density),retain_graph=False)
    assert all(bool(torch.isfinite(v).all()) for v in grad)
    assert bool(grad[0].abs().sum()>0) and bool(grad[1].abs().sum()>0)


def test_surface_appearance_uses_same_nodal_field_and_keeps_features_attached():
    x,_=mesh();features=torch.tensor([[.2,.4,.6],[.7,.5,.3]],dtype=x.dtype,requires_grad=True)
    appearance=SurfaceAppearanceAttachments.from_surface_faces([[0,1,2],[1,2,4]],features)
    F=torch.tensor([[1.,.2,0.],[0.,1.,.1],[0.,0.,1.]],dtype=x.dtype);new=(x@F.T+.3).requires_grad_()
    positions,attributes=appearance.transport(new)
    torch.testing.assert_close(positions,x[appearance.faces].mean(1)@F.T+.3)
    assert attributes is features
    positions.sum().backward();assert new.grad is not None and bool((new.grad[1]>0).all())


def test_rejects_degenerate_reference_inversion_invalid_density_and_mismatched_dtype():
    x,t=mesh(two=False)
    with pytest.raises(ValueError):MaterialGaussians.from_tetrahedra(x,[[0,1,2,2]],750.)
    with pytest.raises(ValueError):MaterialGaussians.from_tetrahedra(x,t,-1.)
    g=MaterialGaussians.from_tetrahedra(x,t,750.)
    inverted=x.clone();inverted[:,0]*=-1
    with pytest.raises(ValueError,match='Jacobian'):g.transport(inverted)
    with pytest.raises(ValueError,match='dtype'):g.transport(x.float())
    with pytest.raises(TypeError):g.to(dtype=torch.float16)
    # Reversed mesh winding is harmless: transport refers to its own rest chart.
    reversed_g=MaterialGaussians.from_tetrahedra(x,t[:,[0,2,1,3]],750.)
    torch.testing.assert_close(reversed_g.transport(x).deformation_gradients,torch.eye(3,dtype=x.dtype)[None])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gpu_float32_transport_and_density_backward():
    device='cuda:'+str(min(2,torch.cuda.device_count()-1))
    x,t=mesh();g=MaterialGaussians.from_tetrahedra(x,t,750.).to(device=device,dtype=torch.float32)
    current=(g.rest_vertices*1.1).requires_grad_();s=g.transport(current)
    loss=s.density(s.centers.detach()).sum()+s.covariances.square().sum();loss.backward()
    assert current.grad.device.type=='cuda' and bool(torch.isfinite(current.grad).all())
