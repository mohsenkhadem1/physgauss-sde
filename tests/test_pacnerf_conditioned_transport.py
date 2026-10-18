"""Guide coordinates must leave the full nonlinear physical target unchanged."""
from pathlib import Path
import sys
import unittest
import torch
from test_pacnerf import fixture
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pacnerf_discrepancy import DiscrepancyModel
from pacnerf_conditioned_transport import CachedPhysicalModel,ConditionedTransport


class ConditionedTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self,force):return CachedPhysicalModel(DiscrepancyModel(fixture('neural',force)))

    def test_roundtrip_and_independent_affine_jacobian_both_force_arms(self):
        for force in ['marginal','joint']:
            model=self.model(force);anchor=torch.zeros(model.dimension,dtype=torch.float64)
            guide=ConditionedTransport(model,anchor)
            z=.1*torch.randn(2,model.dimension,dtype=torch.float64,generator=torch.Generator().manual_seed(45))
            physical,jac=guide.transform(z)
            recovered,inverse_jac=guide.transform(physical,inverse=True)
            torch.testing.assert_close(recovered,z,atol=2e-7,rtol=2e-7)
            torch.testing.assert_close(jac,inverse_jac,atol=1e-9,rtol=1e-9)
            mean,root=guide.gaussian(z)
            # Direct affine transform determinant, independent of the summed
            # precision Cholesky log diagonals used by the target.
            identity=torch.eye(root.shape[-1],dtype=z.dtype).expand(2,-1,-1)
            affine=torch.linalg.solve_triangular(root.transpose(-1,-2),identity,upper=True)
            expected=torch.stack([torch.linalg.slogdet(a.contiguous()).logabsdet for a in affine])
            torch.testing.assert_close(jac,expected,atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(guide.objective(z),model.objective(physical)-expected,atol=1e-8,rtol=1e-9)

    def test_joint_gaussian_guide_marginal_matches_force_eliminated_guide(self):
        marginal=self.model('marginal');joint=self.model('joint')
        gm=ConditionedTransport(marginal,torch.zeros(marginal.dimension,dtype=torch.float64))
        gj=ConditionedTransport(joint,torch.zeros(joint.dimension,dtype=torch.float64))
        zm=torch.zeros(1,marginal.dimension,dtype=torch.float64)
        zj=torch.zeros(1,joint.dimension,dtype=torch.float64)
        zm[:,:2]=zj[:,:2]=torch.tensor([.2,-.3])
        zm[:,-3:]=zj[:,-3:]=torch.tensor([.4,-.2,.3])
        mm,lm=gm.gaussian(zm);mj,lj=gj.gaussian(zj)
        s=marginal.core_dimension-2
        cm=torch.cholesky_inverse(lm);cj=torch.cholesky_inverse(lj)
        # The tiny numerical ridge is proposal-only and differs slightly for
        # the joint/marginal matrices, so use its declared numerical scale.
        torch.testing.assert_close(mm,mj[:,:s],atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(cm,cj[:,:s,:s],atol=2e-5,rtol=2e-4)

    def test_parameter_dependent_jacobian_gradient_matches_finite_difference(self):
        model=self.model('marginal');guide=ConditionedTransport(model,torch.zeros(model.dimension,dtype=torch.float64))
        z=torch.zeros(1,model.dimension,dtype=torch.float64,requires_grad=True)
        value=guide.objective(z);gradient,=torch.autograd.grad(value.sum(),z)
        for index in [0,1,10,model.dimension-3,model.dimension-2,model.dimension-1]:
            direction=torch.zeros_like(z);direction[:,index]=1;eps=2e-5
            finite=(guide.objective(z+eps*direction)-guide.objective(z-eps*direction))/(2*eps)
            torch.testing.assert_close(finite[0],gradient[0,index],atol=2e-3,rtol=2e-4)

    def test_no_grad_cache_is_exact_and_cannot_freeze_parameter_gradients(self):
        model=self.model('marginal');plain=DiscrepancyModel(fixture())
        z=torch.zeros(2,model.dimension,dtype=torch.float64)
        with torch.no_grad():
            for delta in [0.,.2,-.3]:
                z[:,0]=delta
                actual=model.objective(z);expected=plain.objective(z)
                torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-12)
        z=z.requires_grad_()
        actual_grad,=torch.autograd.grad(model.objective(z).sum(),z)
        expected_grad,=torch.autograd.grad(plain.objective(z).sum(),z)
        torch.testing.assert_close(actual_grad,expected_grad,atol=0,rtol=0)

    def test_separable_renderer_and_image_cache_preserve_dense_values(self):
        base=DiscrepancyModel(fixture());cached=CachedPhysicalModel(base)
        z=.1*torch.randn(3,base.dimension,dtype=torch.float64,generator=torch.Generator().manual_seed(37))
        with torch.no_grad():
            for index in [None,-1,0,base.base_dimension-24]:
                if index is not None:z[:,index]+=.15
                q,_=base.decode(z)
                dense=base.renderer(q[:,base.frames],base.specs)
                fast=cached.renderer(q[:,cached.frames],cached.specs)
                torch.testing.assert_close(fast,dense,atol=5e-14,rtol=0)
                original=base.image_residual(z);actual=cached.image_residual(z)
                torch.testing.assert_close(actual,original,atol=2e-12,rtol=0)
                self.assertLess(float((actual.square().sum(-1)-original.square().sum(-1)).abs().max()),1e-8)
