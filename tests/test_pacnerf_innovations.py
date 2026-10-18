"""The innovation model must be an exact reparameterization of the old density."""
import math
from pathlib import Path
import sys
import unittest
import torch
from test_pacnerf import fixture

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:from pacnerf_innovations import InnovationModel
finally:sys.path.pop(0)


class InnovationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def points(self,model):
        rng=torch.Generator().manual_seed(522)
        z=.1*torch.randn(2,model.dimension,dtype=torch.float64,generator=rng)
        z[:,:2]=torch.tensor([[.2,-.3],[-.2,.4]],dtype=z.dtype)
        return z

    def test_density_and_round_trip_all_four_arms(self):
        for noise in ['constant','neural']:
            for force in ['marginal','joint']:
                with self.subTest(noise=noise,force=force):
                    model=fixture(noise,force);innovation=InnovationModel(model)
                    z=self.points(model);physical,jac=innovation.transform(z)
                    recovered,inverse_jac=innovation.transform(physical,inverse=True)
                    torch.testing.assert_close(recovered,z,atol=2e-8,rtol=2e-8)
                    torch.testing.assert_close(inverse_jac,jac,atol=2e-8,rtol=2e-8)
                    expected=model.objective(physical)-jac
                    actual=innovation.objective(z)
                    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-9)

    def test_transformed_density_gradient_matches_original_jacobian(self):
        for force in ['marginal','joint']:
            with self.subTest(force=force):
                model=fixture('neural',force);innovation=InnovationModel(model)
                z=self.points(model)[:1].requires_grad_()
                physical,jac=innovation.transform(z)
                old=model.objective(physical)-jac
                old_grad,=torch.autograd.grad(old.sum(),z)
                new_grad,=torch.autograd.grad(innovation.objective(z).sum(),z)
                torch.testing.assert_close(new_grad,old_grad,atol=2e-5,rtol=2e-6)

    def test_last_innovation_cannot_change_previous_states(self):
        model=fixture();innovation=InnovationModel(model);z=self.points(model)
        before=innovation.to_physical(z);z[:,model.base_dimension-1]+=.1
        after=innovation.to_physical(z)
        torch.testing.assert_close(after[:,:model.base_dimension-24],before[:,:model.base_dimension-24],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
