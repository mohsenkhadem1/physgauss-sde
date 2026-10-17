import unittest
import torch
from physgauss.split_hmc import SplitHMCConfig, sample_split_hmc, split_trajectory
from physgauss.chain_diagnostics import scalar_diagnostics


class SplitHamiltonianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_gaussian_rotation_in_high_dimension(self):
        def target(q):
            return q.new_zeros(len(q)), torch.zeros_like(q), torch.zeros(len(q), dtype=torch.bool)
        result = sample_split_hmc(target, 1448, SplitHMCConfig(warmup=0, draws=1000, step_size=.4))
        self.assertEqual(result['counts']['retained']['accepted'], [1000]*4)
        self.assertLess(float(result['energy_errors'].abs().max()), 2e-11)
        self.assertAlmostEqual(float(result['white'].square().mean()), 1., delta=.01)
        self.assertTrue(scalar_diagnostics(result['white'][..., 0])['diagnostic_pass'])

    def test_reversibility_for_nonlinear_correction(self):
        def target(q):
            return -.1*q.pow(4).sum(1), -.4*q.pow(3), torch.zeros(len(q), dtype=torch.bool)
        rng = torch.Generator().manual_seed(57)
        q = torch.randn(4, 3, dtype=torch.float64, generator=rng)
        p = torch.randn(q.shape, dtype=q.dtype, generator=rng)
        value, g, _ = target(q)
        a, b, v, grad, bad = split_trajectory(q, p, value, g, target, .17, 7)
        x, y, *_ = split_trajectory(a, b, v, grad, target, .17, 7)
        torch.testing.assert_close(x, q, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(y, p, rtol=1e-12, atol=1e-12)
        self.assertFalse(bool(bad.any()))

    def test_nonlinear_banana_has_correct_mean_and_covariance(self):
        # q0~N(0,1), q1|q0~N(.6*(q0^2-1),1). Known non-Gaussian target.
        def target(q):
            offset = .6*(q[:, 0].square()-1)
            residual = q[:, 1]-offset
            ratio = -.5*residual.square()+.5*q[:, 1].square()
            g = torch.stack((1.2*q[:, 0]*residual, offset), 1)
            return ratio, g, ~torch.isfinite(ratio)
        r = sample_split_hmc(target, 2, SplitHMCConfig(warmup=500, draws=12000, seed=73))
        z = r['white'].flatten(0, 1)
        torch.testing.assert_close(z.mean(0), z.new_zeros(2), atol=.04, rtol=0)
        torch.testing.assert_close(z.var(0), z.new_tensor([1., 1.72]), atol=.09, rtol=0)
        for i in range(2):
            self.assertTrue(scalar_diagnostics(r['white'][..., i])['diagnostic_pass'])

    def test_invalid_trajectory_is_rejected(self):
        calls = [0]
        def target(q):
            calls[0] += 1
            return q.new_zeros(len(q)), torch.zeros_like(q), torch.full((len(q),), calls[0] > 1)
        r = sample_split_hmc(target, 2, SplitHMCConfig(warmup=0, draws=20))
        self.assertEqual(r['counts']['retained']['accepted'], [0]*4)
        self.assertEqual(r['counts']['retained']['invalid'], [20]*4)
        torch.testing.assert_close(r['white'], r['white'][:, :1].expand_as(r['white']))

    def test_surrogate_forces_still_target_full_density(self):
        class WrongForces:
            def __call__(self, q):
                # Deliberately omit the quartic term from proposal forces.
                return q.new_zeros(len(q)), torch.zeros_like(q), torch.zeros(len(q), dtype=torch.bool)
            def full_log_ratio(self, q):
                return -.4*q[:, 0].pow(4), torch.zeros(len(q), dtype=torch.bool)
        r = sample_split_hmc(WrongForces(), 1, SplitHMCConfig(warmup=0, draws=5000, seed=187))
        grid = torch.linspace(-5, 5, 50001, dtype=torch.float64)
        p = torch.exp(-.5*grid.square()-.4*grid.pow(4)); p /= p.sum()
        self.assertAlmostEqual(float(r['white'].square().mean()), float(p@grid.square()), delta=.02)
        self.assertLess(sum(r['counts']['retained']['accepted']), 20000)

    def test_continuation_matches_uninterrupted_frozen_chain(self):
        def target(q):
            return -.02*q.pow(4).sum(1), -.08*q.pow(3), torch.zeros(len(q), dtype=torch.bool)
        a = sample_split_hmc(target, 3, SplitHMCConfig(warmup=25, draws=40, seed=283))
        b = sample_split_hmc(target, 3, SplitHMCConfig(warmup=0, draws=40, seed=283, step_size=a['step_size']), resume=a)
        full = sample_split_hmc(target, 3, SplitHMCConfig(warmup=25, draws=80, seed=283))
        torch.testing.assert_close(b['white'], full['white'], rtol=0, atol=0)
        self.assertEqual(b['counts'], full['counts'])

    def test_physical_taylor_matches_value_gradient_and_curvature_at_reference(self):
        from physgauss.neural_marginal_study import default_plan, construct
        from physgauss.stochastic import NeuralDiffusion
        from physgauss.benchmark import Observations
        from physgauss.white_posterior import WhitePosterior
        plan = default_plan(True)
        model = construct(plan, NeuralDiffusion(), {'constant_amplitude': .002}, 'neural')
        mean = model.rollout(model.times.new_zeros(1, 8), model.times.new_zeros(1, 15))[0].detach()
        rgb = model.predict(mean[None], model.view_frames)[0].detach()+.001
        obs = Observations(model.view_frames, rgb, .015)
        # An arbitrary nonsingular reference suffices for this derivative check.
        chol = torch.eye(len(mean), dtype=mean.dtype)*10
        target = WhitePosterior(model, obs, mean, chol)
        z = mean.detach().requires_grad_()
        full = model.objective(z[None], obs).sum()
        approximate = target.proposal_objective(z[None]).sum()
        torch.testing.assert_close(approximate, full, rtol=1e-12, atol=1e-9)
        a, = torch.autograd.grad(full, z, create_graph=True)
        b, = torch.autograd.grad(approximate, z, create_graph=True)
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-8)
        direction = torch.randn(len(z), dtype=z.dtype, generator=torch.Generator().manual_seed(271))
        ha, = torch.autograd.grad(a@direction, z)
        hb, = torch.autograd.grad(b@direction, z)
        torch.testing.assert_close(ha, hb, rtol=1e-9, atol=1e-7)
