import math
import unittest
import torch
from physgauss.benchmark import Config, Observations
from physgauss.forces import ForceSceneModel
from physgauss.marginal_forces import GaussianLoadElimination, MarginalConfig, TrajectoryForceModel
from physgauss.marginal_inference import recover_force_moments


class MarginalForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def example(self):
        gen = torch.Generator().manual_seed(190)
        g = .2*torch.randn(9, 3, dtype=torch.float64, generator=gen)
        sd = torch.linspace(.1, .3, 9, dtype=torch.float64)
        r = torch.randn(4, 9, dtype=torch.float64, generator=gen)
        return GaussianLoadElimination(g, sd), r

    def test_woodbury_density_whitening_and_force_posterior_match_dense_gaussian(self):
        e, r = self.example()
        covariance = torch.diag(e.sd.square())+e.mapping@e.mapping.T
        dense = torch.distributions.MultivariateNormal(torch.zeros(9, dtype=r.dtype), covariance)
        torch.testing.assert_close(e.marginal_nll(r), -dense.log_prob(r), rtol=1e-12, atol=1e-12)
        expected_mean = (e.mapping.T@torch.linalg.solve(covariance, r.T)).T
        expected_cov = torch.eye(3, dtype=r.dtype)-e.mapping.T@torch.linalg.solve(covariance, e.mapping)
        torch.testing.assert_close(e.conditional_mean(r), expected_mean, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(e.conditional_covariance, expected_cov, rtol=1e-12, atol=1e-12)
        value = r.clone().requires_grad_()
        grad, = torch.autograd.grad(e.marginal_nll(value).sum(), value)
        torch.testing.assert_close(grad, torch.linalg.solve(covariance, r.T).T, rtol=1e-12, atol=1e-12)

    def test_normalized_joint_equals_marginal_times_conditional(self):
        e, r = self.example()
        a = r.new_tensor([[.2, -.1, .8]]).expand(4, -1)
        conditional = torch.distributions.MultivariateNormal(e.conditional_mean(r), e.conditional_covariance)
        torch.testing.assert_close(e.joint_nll(r, a), e.marginal_nll(r)-conditional.log_prob(a), atol=1e-12, rtol=1e-12)
        offset = .5*3*math.log(2*math.pi)-e.cholesky.diagonal().log().sum()
        torch.testing.assert_close(e.joint_nll(r, e.conditional_mean(r))-e.marginal_nll(r), offset.expand(4))

    def test_log_determinant_prevents_unbounded_force_variance(self):
        e, r = self.example()
        # A perfect zero residual cannot prefer arbitrarily broad forces.
        wide = GaussianLoadElimination(100*e.mapping, e.sd)
        self.assertGreater(float(wide.marginal_nll(torch.zeros_like(r))[0]),
                           float(e.marginal_nll(torch.zeros_like(r))[0]))
        for bad in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                MarginalConfig(velocity_diffusion=bad).validate()

    def test_trajectory_residual_is_exact_new_simulator_innovation_with_si_units(self):
        m = TrajectoryForceModel(Config.quick())
        gen = torch.Generator().manual_seed(191)
        theta = .1*torch.randn(2, 8, dtype=torch.float64, generator=gen)
        a = .2*torch.randn(2, 15, dtype=torch.float64, generator=gen)
        white = torch.randn(2, m.nsteps, m.free_dof, dtype=torch.float64, generator=gen)
        state = m.rollout(theta, a, white)
        expected = a@m.elimination.mapping.T+white.flatten(1)*m.elimination.sd
        torch.testing.assert_close(m.residual(state), expected, atol=2e-15, rtol=1e-11)
        steps = m.step_trajectory(state)
        torch.testing.assert_close(steps.x[:, 1:]-steps.x[:, :-1],
                                   steps.v[:, 1:]*m.step_sizes[None, :, None, None], atol=1e-15, rtol=1e-12)
        torch.testing.assert_close(steps.x[:, :, m.mesh.pinned], m.mesh.rest[m.mesh.pinned][None, None].expand(2, m.nsteps+1, -1, -1))

    def test_zero_added_innovation_rollout_matches_existing_force_simulator(self):
        cfg = Config.quick()
        old = ForceSceneModel(cfg); new = TrajectoryForceModel(cfg)
        gen = torch.Generator().manual_seed(192)
        z = .15*torch.randn(2, old.dimension, dtype=torch.float64, generator=gen)
        state = new.rollout(z[:, :8], z[:, 8:])
        torch.testing.assert_close(new.trajectory(state).x, old.trajectory(z).x, atol=1e-14, rtol=1e-12)
        torch.testing.assert_close(new.trajectory(state).v, old.trajectory(z).v, atol=1e-14, rtol=1e-12)

    def test_nonlinear_mechanics_and_image_jacobians_match_branch_autograd(self):
        cfg = Config.quick(); cfg.frames = 5; cfg.substeps = 1; cfg.observed_frames = (0, 4)
        m = TrajectoryForceModel(cfg)
        gen = torch.Generator().manual_seed(193)
        theta = .12*torch.randn(1, 8, dtype=torch.float64, generator=gen)
        state = m.rollout(theta, torch.zeros(1, 15, dtype=torch.float64))[0]
        obs = Observations(torch.tensor([0, 4]), m.predict(state[None], torch.tensor([0, 4]))[0].detach(), .015)
        images, dynamics = m.jacobians(state, obs)
        direction = torch.randn(state.shape, dtype=state.dtype, generator=gen)
        value = state.detach().requires_grad_()
        residual = m.image_residual(value[None], obs).flatten()
        weights = torch.randn(residual.shape, dtype=state.dtype, generator=gen)
        grad, = torch.autograd.grad(weights@residual, value)
        torch.testing.assert_close(weights@(images@direction), grad@direction, atol=1e-9, rtol=1e-10)
        _, jvp = torch.autograd.functional.jvp(lambda q: m.residual(q[None])[0], state, direction)
        torch.testing.assert_close(dynamics@direction, jvp, atol=1e-11, rtol=1e-11)

    def test_schur_elimination_preserves_joint_local_gaussian_mean_and_covariance(self):
        e, r = self.example()
        gen = torch.Generator().manual_seed(194)
        j = torch.randn(9, 5, dtype=r.dtype, generator=gen)
        wj = j/e.sd[:, None]
        hxx = torch.eye(5, dtype=r.dtype)+wj.T@wj
        hxa = -wj.T@e.white_mapping
        full = torch.cat((torch.cat((hxx, hxa), 1), torch.cat((hxa.T, e.precision), 1)), 0)
        whitened_j = e.whiten(j.T).T
        reduced = torch.eye(5, dtype=r.dtype)+whitened_j.T@whitened_j
        torch.testing.assert_close(torch.linalg.inv(full)[:5, :5], torch.linalg.inv(reduced), atol=1e-12, rtol=1e-12)
        rx = j.T@torch.linalg.solve(torch.diag(e.sd.square()), r[0])
        ra = -e.white_mapping.T@(r[0]/e.sd)
        full_step = torch.linalg.solve(full, torch.cat((rx, ra)))[:5]
        marginal_rhs = rx-hxa@torch.linalg.solve(e.precision, ra)
        torch.testing.assert_close(full_step, torch.linalg.solve(reduced, marginal_rhs), atol=1e-12, rtol=1e-12)

    def test_force_recovery_includes_trajectory_uncertainty(self):
        e, r = self.example()
        moments = recover_force_moments(e, r)
        torch.testing.assert_close(moments["covariance"], e.conditional_covariance+torch.cov(e.conditional_mean(r).T))
        self.assertGreater(float(moments["between_covariance"].trace()), 0.)
        gen = torch.Generator().manual_seed(195)
        samples = e.sample_conditional(r[:1].expand(15000, -1), gen)
        torch.testing.assert_close(samples.mean(0), e.conditional_mean(r[:1])[0], atol=.012, rtol=.02)
        torch.testing.assert_close(torch.cov(samples.T), e.conditional_covariance, atol=.008, rtol=.06)

    def test_causal_trajectory_density_has_declared_constant_jacobian_and_no_truth_leak(self):
        cfg = Config.quick(); cfg.frames = 5; cfg.substeps = 1; cfg.observed_frames = (0, 4)
        m = TrajectoryForceModel(cfg)
        theta = torch.zeros(1, 8, dtype=torch.float64)
        state = m.rollout(theta, torch.zeros(1, 15, dtype=torch.float64))[0]
        jac = torch.autograd.functional.jacobian(lambda q: m.residual(q[None])[0], state)[:, 8:]
        expected = jac.shape[0]*math.log(m.marginal_config.velocity_coordinate_scale)
        self.assertAlmostEqual(float(torch.linalg.slogdet(jac)[1]), expected, places=10)
        cfg.truth_external_pulse = True; cfg.truth_stochastic = False
        other = TrajectoryForceModel(cfg)
        torch.testing.assert_close(m.residual(state[None]), other.residual(state[None]), atol=0, rtol=0)
        a = m.elimination.conditional_mean(m.residual(state[None]))
        torch.testing.assert_close(m.external_forces(state[None]), m.force_prior.nodal_force(a, m.step_times))
        with self.assertRaises(NotImplementedError):
            m.force_record(state[None])


if __name__ == "__main__":
    unittest.main()
