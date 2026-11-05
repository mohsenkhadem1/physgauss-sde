import unittest
import torch
from scripts.pidg_comparison.mechanics import ConstitutivePrior,deformation_gradient
class MechanicsTransferTests(unittest.TestCase):
    def setUp(self):self.prior=ConstitutivePrior('cpu');torch.manual_seed(42)
    def test_affine_gradient_and_rigid_objectivity(self):
        x=torch.randn(20,3,dtype=torch.float64);R=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=x.dtype)
        F=deformation_gradient(lambda q:q@R.T+q.new_tensor([3.,-2.,1.]),x)
        torch.testing.assert_close(F,R.expand_as(F),atol=1e-12,rtol=1e-12)
        self.assertLess(abs(float(self.prior.energy(F.float()))),1e-5)
    def test_compression_and_inversion_have_finite_gradients(self):
        for stretch in [.7,-.7]:
            F=torch.diag(torch.tensor([stretch,1.,1.])).repeat(4,1,1).requires_grad_(True)
            loss=self.prior.energy(F);loss.backward();self.assertTrue(torch.isfinite(F.grad).all());self.assertGreater(float(loss),0.)
    def test_rotated_strain_same_energy(self):
        F=torch.diag(torch.tensor([.8,1.1,1.]))[None];R=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        torch.testing.assert_close(self.prior.energy(F),self.prior.energy(R@F))
if __name__=='__main__':unittest.main()
