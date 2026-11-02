import torch
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw

def test_energy_stress_tangent_exact_scale_and_rigid_reference():
 torch.manual_seed(191);law=ScaleSeparatedMaterialLaw().double()
 rotation,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));rotation[:,0]*=torch.linalg.det(rotation)
 states=[torch.eye(3,dtype=torch.float64),rotation,torch.tensor([[1.15,.2,0],[0,.9,.1],[0,0,1.05]],dtype=torch.float64)]
 for F in states:
  z=torch.tensor([0.,.3,1.],dtype=torch.float64)
  W=law(F,z);P=law.stress(F,z)
  H=torch.autograd.functional.jacobian(lambda f:law.stress(f,z),F)
  for scale in (.2,.7,2.,5.):
   zs=z.clone();zs[0]=torch.log(torch.tensor(scale,dtype=z.dtype))
   torch.testing.assert_close(law(F,zs),W*scale,atol=1e-10,rtol=1e-12)
   torch.testing.assert_close(law.stress(F,zs),P*scale,atol=1e-9,rtol=1e-12)
   Hs=torch.autograd.functional.jacobian(lambda f:law.stress(f,zs),F)
   torch.testing.assert_close(Hs,H*scale,atol=1e-8,rtol=1e-11)
  if torch.max(torch.abs(F.T@F-torch.eye(3,dtype=F.dtype)))<1e-10:
   assert abs(W)<1e-10 and torch.max(torch.abs(P))<1e-9

def test_log_scale_derivative_equals_energy_and_stress():
 law=ScaleSeparatedMaterialLaw().double();F=torch.tensor([[1.1,.2,0],[0,.9,0],[0,0,1.02]],dtype=torch.float64)
 z=torch.tensor([.2,-.3,-1.],dtype=torch.float64,requires_grad=True)
 W=law(F,z);grad=torch.autograd.grad(W,z,retain_graph=True)[0]
 torch.testing.assert_close(grad[0],W)
 P=law.stress(F,z);gradP=torch.autograd.grad(P.sum(),z)[0]
 torch.testing.assert_close(gradP[0],P.sum())
