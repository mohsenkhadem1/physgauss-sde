import unittest

import numpy as np
import torch

from experiments.pidg_style_baseline import (
    FieldConfig, PIDGStyleField, batch_jacobian, cofactor_matrix,
    convective_acceleration, neo_hookean_cauchy, physics_residuals,
    sample_reference_volume, observation_loss,
)


class PIDGStyleBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.rest = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        self.config = FieldConfig(width=12, depth=2, frequencies=1)

    def test_exact_initial_conditions_survive_trained_weights(self):
        model = PIDGStyleField(self.rest, self.config)
        with torch.no_grad():
            model.output.weight.normal_();model.output.bias.normal_()
        t = torch.zeros(4,1,requires_grad=True)
        fields = model(self.rest, t)
        torch.testing.assert_close(fields['positions_m'],self.rest,rtol=0,atol=0)
        torch.testing.assert_close(fields['velocity_m_s'],torch.zeros_like(self.rest),rtol=0,atol=0)
        initial_v = batch_jacobian(fields['positions_m'],t).squeeze(-1)
        torch.testing.assert_close(initial_v,torch.zeros_like(initial_v),rtol=0,atol=0)

    def test_convective_term_is_not_kinetic_energy_gradient(self):
        points = torch.tensor([[.1,.2,.3],[.2,.3,.4]],requires_grad=True)
        velocity = torch.stack((points[:,1].square(),points[:,0]*0,points[:,2]*0),-1)
        derivative = batch_jacobian(velocity,points)
        acceleration = convective_acceleration(velocity,derivative)
        torch.testing.assert_close(acceleration,torch.zeros_like(acceleration))
        wrong = torch.einsum('bi,bij->bj',velocity,derivative)
        self.assertGreater(float(wrong.detach().abs().max()),.05)

    def test_neo_hookean_stress_rotation_and_dilation(self):
        a = .7
        rotation = torch.tensor([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]],dtype=torch.float64)
        stress = neo_hookean_cauchy(rotation[None],5000.,.35)
        torch.testing.assert_close(stress,torch.zeros_like(stress),rtol=0,atol=1e-10)
        s = 1.2; f = torch.eye(3,dtype=torch.float64)[None]*s
        stress = neo_hookean_cauchy(f,5000.,.35)
        mu=5000/(2*1.35);lam=5000*.35/(1.35*.3);j=s**3
        expected=(mu*(s*s-1)+lam*np.log(j))/j
        torch.testing.assert_close(stress,torch.eye(3,dtype=torch.float64)[None]*expected)

    def test_cofactor_matches_inverse_and_is_finite_at_singular(self):
        f = torch.randn(5,3,3,dtype=torch.float64)+torch.eye(3,dtype=torch.float64)*3
        expected = torch.linalg.det(f)[:,None,None]*torch.linalg.inv(f).transpose(-1,-2)
        torch.testing.assert_close(cofactor_matrix(f),expected)
        self.assertTrue(torch.isfinite(cofactor_matrix(torch.zeros(2,3,3))).all())

    def test_rest_residual_zero_and_perturbed_training_gradient_finite(self):
        model = PIDGStyleField(self.rest,self.config)
        points = torch.rand(7,3)*.2
        times = torch.rand(7)*3
        residual = physics_residuals(model,points,times)
        for key in ['kinematic','momentum','constitutive','inversion']:
            torch.testing.assert_close(residual[key],torch.zeros_like(residual[key]),rtol=0,atol=1e-6)
        with torch.no_grad():model.output.weight.normal_(std=.001)
        residual = physics_residuals(model,points,times)
        loss = sum(residual[k].square().mean() for k in ['kinematic','momentum','constitutive','inversion'])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.output.weight.grad)
        self.assertGreater(float(model.output.weight.grad.detach().abs().sum()),0)
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())

    def test_lagrangian_momentum_matches_analytic_translation(self):
        config=self.config
        class Translation:
            def __init__(self):
                self.config=config;self.length=torch.tensor(1.)
                self.youngs_modulus_pa=torch.tensor(5000.)
                self.log_youngs_modulus=torch.tensor(5000.).log()
                self.reference_log_youngs=self.log_youngs_modulus
            def __call__(self,x,t):
                acceleration=x.new_tensor([.1,-.2,.3])
                displacement=.5*t.square()*acceleration
                return {'positions_m':x+displacement,'velocity_m_s':t*acceleration,
                        'cauchy_stress_pa':(x[:,:1].square()*0)[:,:,None].expand(-1,3,3),
                        'displacement_m':displacement}
        points=torch.rand(5,3);times=torch.linspace(.1,2.9,5)
        residual=physics_residuals(Translation(),points,times)
        scale=5000+600/9
        expected=torch.tensor([.1,-.2,.3])[None].expand(5,-1)*600/scale
        torch.testing.assert_close(residual['momentum'],expected)
        torch.testing.assert_close(residual['kinematic'],torch.zeros(5,3))
        torch.testing.assert_close(residual['constitutive'],torch.zeros(5,3,3))

    def test_volume_samples_stay_inside_tetrahedron_and_mean_correct(self):
        generator = torch.Generator().manual_seed(8)
        points = sample_reference_volume(self.rest,torch.tensor([[0,1,2,3]]),10000,generator)
        self.assertTrue((points>=0).all()); self.assertTrue((points.sum(-1)<=1+1e-6).all())
        torch.testing.assert_close(points.mean(0),torch.full((3,),.25),rtol=0,atol=.008)

    def test_observation_loss_has_gradients_and_no_truth_argument(self):
        rgb=torch.ones(1,3,4,3,requires_grad=True)*.3
        alpha=torch.ones(1,3,4,requires_grad=True)*.5
        loss,parts=observation_loss({'rgb':rgb,'alpha':alpha},torch.zeros_like(rgb),torch.ones_like(alpha))
        self.assertEqual(set(parts),{'rgb_mse_weighted','silhouette_mse'})
        rgb_grad,alpha_grad=torch.autograd.grad(loss,(rgb,alpha))
        self.assertTrue((rgb_grad>0).all());self.assertTrue((alpha_grad<0).all())


if __name__ == '__main__':
    unittest.main()
