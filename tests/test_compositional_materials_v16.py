import pytest
import torch
from scripts.compositional_neural_mechanics.materials import SharedMaterialLaw,teacher_energy

def rotation():
 q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));q[:,0]*=torch.linalg.det(q);return q

def test_reference_normalization_all_latents_and_objectivity():
 torch.manual_seed(31);law=SharedMaterialLaw().double();z=torch.randn(8,2,dtype=torch.float64)*4
 I=torch.eye(3,dtype=torch.float64).expand(8,3,3).clone().requires_grad_(True)
 torch.testing.assert_close(law(I,z),torch.zeros(8,dtype=torch.float64),atol=1e-10,rtol=0)
 torch.testing.assert_close(law.stress(I,z),torch.zeros_like(I),atol=1e-9,rtol=0)
 F=I.detach()+torch.randn_like(I)*.12;R=rotation();Q=rotation()
 torch.testing.assert_close(law(R@F,z),law(F,z),atol=1e-9,rtol=1e-10)
 torch.testing.assert_close(law(F@Q,z),law(F,z),atol=1e-9,rtol=1e-10)
 torch.testing.assert_close(law.stress(R@F,z),R@law.stress(F,z),atol=1e-8,rtol=1e-9)

def test_stress_tangent_parameter_grad_and_point_independence():
 law=SharedMaterialLaw().double();z=torch.tensor([.2,-.3],dtype=torch.float64,requires_grad=True)
 F=torch.diag(torch.tensor([1.1,.95,1.03],dtype=torch.float64)).requires_grad_(True)
 assert torch.autograd.gradcheck(lambda f:law(f,z),(F,),eps=1e-6,atol=1e-4)
 assert torch.autograd.gradgradcheck(lambda f:law(f,z),(F,),eps=1e-6,atol=1e-3)
 stress=law.stress(F,z);loss=stress.square().sum();loss.backward()
 assert z.grad is not None and torch.isfinite(z.grad).all()
 assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in law.parameters())
 batch=F.detach().repeat(3,1,1);zs=z.detach().repeat(3,1)
 torch.testing.assert_close(law(batch,zs),law(F.detach(),z.detach()).expand(3))

def test_positive_energy_extreme_latents_and_invalid_orientation():
 torch.manual_seed(44);law=SharedMaterialLaw().double()
 F=torch.diag_embed(torch.exp(torch.randn(200,3,dtype=torch.float64)*.7));z=torch.randn(200,2,dtype=torch.float64)*20
 assert (law(F,z)>=-1e-8).all()
 with pytest.raises(ValueError):law(-torch.eye(3,dtype=torch.float64),z[0])

def test_two_families_same_linear_stiffness_different_large_strain():
 z=torch.tensor([0.,0.],dtype=torch.float64);I=torch.eye(3,dtype=torch.float64)
 H=torch.zeros_like(I);H[0,1]=1
 for family in ('mooney_rivlin','yeoh'):
  t=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
  W=teacher_energy(I+t*H,z,family);first=torch.autograd.grad(W,t,create_graph=True)[0]
  second=torch.autograd.grad(first,t)[0]
  torch.testing.assert_close(second,torch.tensor(4000.,dtype=torch.float64))
 F=I+.7*H
 assert teacher_energy(F,z,'yeoh')>teacher_energy(F,z,'mooney_rivlin')*1.1
