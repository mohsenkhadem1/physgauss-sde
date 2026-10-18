"""Density, gradient, no-leakage and repeated-data uncertainty checks."""
import math
from pathlib import Path
import sys
import unittest
import torch
from test_pacnerf import fixture
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pacnerf_discrepancy import DiscrepancyModel,DiscrepancyInnovation,ConditionalInnovation,gauss_newton,spatial_whiten


class DiscrepancyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_correlated_whitening_matches_dense_inverse_covariance(self):
        x=torch.randn(2,3,4,5,3,dtype=torch.float64,generator=torch.Generator().manual_seed(93))
        rx,ry=.7,.4
        ar=lambda n,r:r**(torch.arange(n)[:,None]-torch.arange(n)[None]).abs().double()
        covariance=torch.kron(ar(4,ry),ar(5,rx))
        chol=torch.linalg.cholesky(covariance)
        expected=torch.linalg.solve_triangular(chol,x.flatten(2,3),upper=False)
        actual=spatial_whiten(x,rx,ry).flatten(2,3)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)

    def test_density_and_roundtrip_all_arms_with_learned_scales(self):
        for noise in ['constant','neural']:
            for force in ['marginal','joint']:
                model=DiscrepancyModel(fixture(noise,force));innovation=DiscrepancyInnovation(model)
                z=.1*torch.randn(2,model.dimension,dtype=torch.float64,generator=torch.Generator().manual_seed(61))
                z[:,-3:]=torch.tensor([[.3,-.4,.2],[-.2,.5,-.3]])
                physical,jac=innovation.transform(z)
                recovered,inverse_jac=innovation.transform(physical,inverse=True)
                torch.testing.assert_close(recovered,z,atol=2e-8,rtol=2e-8)
                torch.testing.assert_close(inverse_jac,jac,atol=2e-8,rtol=2e-8)
                torch.testing.assert_close(model.objective(physical)-jac,innovation.objective(z),atol=2e-6,rtol=1e-9)

    def test_hyperparameter_gradients_include_covariance_and_likelihood_normalizers(self):
        model=DiscrepancyModel(fixture());innovation=DiscrepancyInnovation(model)
        z=torch.zeros(1,model.dimension,dtype=torch.float64,requires_grad=True)
        physical,jac=innovation.transform(z)
        old_gradient,=torch.autograd.grad((model.objective(physical)-jac).sum(),z)
        gradient,=torch.autograd.grad(innovation.objective(z).sum(),z)
        torch.testing.assert_close(gradient,old_gradient,atol=2e-5,rtol=2e-6)
        for i in [-3,-2,-1]:
            direction=torch.zeros_like(z);direction[:,i]=1;eps=1e-5
            finite=(innovation.objective(z+eps*direction)-innovation.objective(z-eps*direction))/(2*eps)
            torch.testing.assert_close(finite[0],gradient[0,i],atol=2e-5,rtol=2e-5)
        # Increasing scale changes the Gaussian determinant even at zero error.
        a=z.detach().clone();b=a.clone();b[:,-1]=1
        self.assertGreater(float(model.image_normalizer(b)-model.image_normalizer(a)),0)

    def test_joint_and_marginal_targets_integrate_the_same_expanded_load_prior(self):
        marginal=DiscrepancyModel(fixture(force='marginal'))
        joint=DiscrepancyModel(fixture(force='joint'))
        z=torch.zeros(1,marginal.dimension,dtype=torch.float64);z[:,-3:]=torch.tensor([.3,-.2,.5])
        mean,cov,_=marginal.force_conditionals(z)
        zj=torch.cat((z[:,:marginal.base_dimension],mean,z[:,-3:]),-1)
        # p(z,a)=p(z)*N(a|z); evaluate the exact normalizer at conditional mean.
        expected=-.5*(torch.linalg.slogdet(cov).logabsdet+marginal.force_dimension*math.log(2*math.pi))
        torch.testing.assert_close(joint.objective(zj)-marginal.objective(z),-expected,atol=2e-6,rtol=2e-8)

    def test_gauss_newton_metric_matches_full_residual_directional_derivatives(self):
        model=DiscrepancyModel(fixture());innovation=DiscrepancyInnovation(model)
        z=torch.zeros(model.dimension,dtype=torch.float64)
        precision,_=gauss_newton(innovation,z)
        direction=torch.randn(2,len(z),dtype=z.dtype,generator=torch.Generator().manual_seed(76));direction/=direction.norm(dim=-1,keepdim=True)
        # Explicitly include an RGB-scale direction, which the old metric lacks.
        direction[0].zero_();direction[0,-1]=1
        eps=1e-5
        with torch.no_grad():
            plus=model.image_residual(innovation.to_physical(z[None]+eps*direction))
            minus=model.image_residual(innovation.to_physical(z[None]-eps*direction))
        expected=1+((plus-minus)/(2*eps)).square().sum(-1)
        actual=torch.einsum('bi,ij,bj->b',direction,precision,direction)
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=3e-5)

    def test_later_unobserved_images_cannot_set_the_prior_or_initial_correlation(self):
        base=fixture();first=DiscrepancyModel(base)
        for (camera,frame),image in list(base.data['images'].items()):
            if frame:base.data['images'][camera,frame]=torch.full_like(image,999.)
        second=DiscrepancyModel(base)
        self.assertEqual(first.specification(),second.specification())
        torch.testing.assert_close(first.acceleration_mapping,second.acceleration_mapping,atol=0,rtol=0)

    def test_added_noise_and_smooth_load_integrals_match_independent_full_matrix_reference(self):
        from verify_pacnerf_discrepancy import independent_integrals
        for force in ['marginal','joint']:
            model=DiscrepancyModel(fixture('neural',force))
            z=torch.zeros(2,model.dimension,dtype=torch.float64)
            z[:,0]=torch.tensor([0.,9.]);z[:,-3:]=torch.tensor([[.3,-.2,.1],[-.4,.2,.3]])
            physical=DiscrepancyInnovation(model).to_physical(z)
            self.assertTrue(independent_integrals(model,physical)['all_pass'])

    def test_covariance_filter_matches_independent_precision_filter(self):
        from pacnerf_innovations import InnovationModel
        model=DiscrepancyModel(fixture())
        z=.2*torch.randn(2,model.dimension,dtype=torch.float64,generator=torch.Generator().manual_seed(53))
        core=ConditionalInnovation(model,z[:,-3:])
        fast,jac=core.transform(z[:,:-3])
        reference,reference_jac=InnovationModel.transform(core,z[:,:-3])
        torch.testing.assert_close(fast,reference,atol=1e-8,rtol=1e-8)
        torch.testing.assert_close(jac,reference_jac,atol=1e-8,rtol=1e-8)

    def test_repeated_linear_gaussian_missing_frame_intervals_have_nominal_coverage(self):
        # Independent direct conditioning of an integrated acceleration process
        # plus smooth global Gaussian force. 5000 independent datasets, not 5000
        # correlated coordinates from one deterministic PAC trajectory.
        times=torch.arange(1,9,dtype=torch.float64)/8
        minimum=torch.minimum(times[:,None],times[None])
        maximum=torch.maximum(times[:,None],times[None])
        process=.4**2*minimum.square()*(3*maximum-minimum)/6
        force=(times.square()/2)[:,None]*.8
        prior=process+force@force.T
        observed=torch.tensor([0,2,7]);missing=torch.tensor([3,4,5,6]);sensor=.03
        cross=prior[missing][:,observed]
        yy=prior[observed][:,observed]+sensor**2*torch.eye(3,dtype=torch.float64)
        gain=torch.linalg.solve(yy,cross.T).T
        conditional=prior[missing][:,missing]-gain@cross.T
        rng=torch.Generator().manual_seed(225)
        truth=torch.randn(5000,8,dtype=torch.float64,generator=rng)@torch.linalg.cholesky(prior).T
        images=truth[:,observed]+sensor*torch.randn(5000,3,dtype=torch.float64,generator=rng)
        standardized=(truth[:,missing]-images@gain.T)/conditional.diag().sqrt()
        coverage=(standardized.abs()<=1.6448536269514722).double().mean(0)
        self.assertTrue(bool(((coverage>.88)&(coverage<.92)).all()),coverage.tolist())
        # Joint Gaussian conditioning and Woodbury marginalization agree.
        low_rank=force[observed];q=process[observed][:,observed]+sensor**2*torch.eye(3,dtype=torch.float64)
        inverse=torch.linalg.inv(q)
        woodbury=inverse-inverse@low_rank@torch.linalg.solve(torch.eye(1,dtype=q.dtype)+low_rank.T@inverse@low_rank,low_rank.T@inverse)
        torch.testing.assert_close(woodbury,torch.linalg.inv(yy),atol=1e-10,rtol=1e-10)


if __name__=='__main__':unittest.main()
