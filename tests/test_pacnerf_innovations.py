"""The innovation model must be an exact reparameterization of the old density."""
import math
from pathlib import Path
import sys
import unittest
import torch
from test_pacnerf import fixture

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from pacnerf_innovations import InnovationModel
    from pacnerf_innovation_metric import gauss_newton,pose_jacobian
    from pacnerf_comparison import make_target
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

    def test_full_rgb_metric_matches_direct_image_jacobian(self):
        model=fixture('neural','marginal');innovation=InnovationModel(model)
        z=self.points(model)[0]
        precision,_=gauss_newton(innovation,z)
        # Independent finite differences include the full causal transformation
        # and renderer, rather than the metric's pose-chain-rule assembly.
        directions=torch.randn(3,len(z),dtype=z.dtype,generator=torch.Generator().manual_seed(92))
        directions/=directions.norm(dim=-1,keepdim=True)
        eps=2e-5
        with torch.no_grad():
            plus=model.image_residual(innovation.to_physical(z[None]+eps*directions))
            minus=model.image_residual(innovation.to_physical(z[None]-eps*directions))
        expected=((plus-minus)/(2*eps)).square().sum(-1)+1
        actual=torch.einsum('bi,ij,bj->b',directions,precision,directions)
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)

    def test_pose_jacobian_does_not_read_unobserved_images(self):
        model=fixture();innovation=InnovationModel(model);z=self.points(model)[0]
        pose,jac=pose_jacobian(innovation,z)
        for image in model.data['images'].values():image.fill_(999)
        after,after_jac=pose_jacobian(innovation,z)
        torch.testing.assert_close(after,pose,atol=0,rtol=0)
        torch.testing.assert_close(after_jac,jac,atol=0,rtol=0)

    def test_sampling_target_uses_innovation_density_and_decodes_physical_states(self):
        for force in ['marginal','joint']:
            model=fixture('neural',force);innovation=InnovationModel(model)
            d=model.dimension;mean=torch.zeros(d,dtype=torch.float64)
            precision=2*torch.eye(d,dtype=mean.dtype)
            target=make_target(model,{'mean':mean,'precision':precision,'coordinates':'innovation'})
            w=self.points(model).requires_grad_()
            native=target.coordinates(w)
            expected=-innovation.objective(native)+target.center+.5*w.square().sum(-1)
            expected_grad,=torch.autograd.grad(expected.sum(),w)
            value,gradient,bad=target(w.detach())
            self.assertFalse(bool(bad.any()))
            torch.testing.assert_close(value,expected.detach(),atol=1e-10,rtol=1e-10)
            torch.testing.assert_close(gradient,expected_grad,atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(target.decode(w.detach()),innovation.to_physical(native.detach()),atol=0,rtol=0)
            replay,_=target.log_ratio_no_grad(w.detach())
            torch.testing.assert_close(replay,value,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
