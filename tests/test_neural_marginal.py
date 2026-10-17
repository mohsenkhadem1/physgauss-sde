import math
import json
from pathlib import Path
import unittest
import torch
from physgauss.benchmark import Config, Observations
from physgauss.accurate_pipeline import AccurateViewModel
from physgauss.continuous_render import ContinuousGaussianRenderer, VolumeRenderConfig
from physgauss.neural_marginal import (NeuralMarginalModel, InternalNoiseConfig,
    BlockLoadElimination, covariance_information)
from physgauss.stochastic import NeuralDiffusion


class NeuralMarginalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def small(self, mode="neural"):
        torch.manual_seed(7101)
        c = Config.quick(); c.frames = 5; c.duration = .12; c.substeps = 1; c.observed_frames = (0, 4)
        model = NeuralMarginalModel(c, NeuralDiffusion(), InternalNoiseConfig(mode=mode))
        model.set_views([(0, 0), (4, 0)])
        model.renderer = ContinuousGaussianRenderer(model.mesh, 12, VolumeRenderConfig(bins=24))
        return model

    def state(self, model):
        generator = torch.Generator().manual_seed(7102)
        theta = .1*torch.randn(1, 8, dtype=torch.float64, generator=generator)
        coefficients = .1*torch.randn(1, 15, dtype=torch.float64, generator=generator)
        white = .3*torch.randn(1, model.nsteps, model.free_dof, dtype=torch.float64, generator=generator)
        return model.covariance_rollout(theta, coefficients, white), coefficients, white

    def test_state_dependent_elimination_matches_dense_normal_density_and_gradient(self):
        m = self.small()
        state, a, _ = self.state(m)
        state = state.detach().requires_grad_()
        elim = m.elimination_at(state)
        residual = m.residual(state)
        q = torch.block_diag(*elim.covariance[0].unbind())
        g = m.load_mapping.flatten(0, 1)
        c = q+g@g.T
        expected = -torch.distributions.MultivariateNormal(torch.zeros_like(residual[0]), covariance_matrix=c).log_prob(residual[0])
        actual = elim.marginal_nll(residual)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-11)
        ga, = torch.autograd.grad(actual, state, retain_graph=True)
        ge, = torch.autograd.grad(expected, state)
        torch.testing.assert_close(ga, ge, rtol=1e-9, atol=1e-9)
        q_inv_g = torch.linalg.solve(q, g)
        cov = torch.linalg.inv(torch.eye(15, dtype=state.dtype)+g.T@q_inv_g)
        mean = cov@g.T@torch.linalg.solve(q, residual[0])
        torch.testing.assert_close(elim.conditional_mean(residual)[0], mean, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(elim.conditional_covariance[0], cov, rtol=1e-11, atol=1e-11)

    def test_causal_covariance_rollout_has_normalized_innovations_and_jacobian(self):
        m = self.small(); state, a, white = self.state(m)
        elim = m.elimination_at(state)
        centered = m.residual(state)-torch.einsum("sdk,bk->bsd", m.load_mapping, a).flatten(1)
        torch.testing.assert_close(elim.whiten_q(centered), white, rtol=1e-9, atol=1e-12)
        # Covariance depends only on past variables, so the raw residual
        # Jacobian remains triangular with the centered-drag diagonal.
        jac = torch.autograd.functional.jacobian(lambda z: m.residual(z[None])[0], state[0])[:, 8:]
        expected = m.log_dynamic_jacobian(state)[0]+jac.shape[0]*math.log(m.marginal_config.velocity_coordinate_scale)
        torch.testing.assert_close(torch.linalg.slogdet(jac)[1], expected, rtol=1e-12, atol=1e-12)
        obs = Observations(m.view_frames, m.predict(state, m.view_frames)[0].detach(), .015)
        point = state.detach().requires_grad_()
        objective = m.objective(point, obs).sum()
        gradient, = torch.autograd.grad(objective, point)
        direction = torch.randn(point.shape, dtype=point.dtype, generator=torch.Generator().manual_seed(7103))
        direction /= direction.norm()
        h = 1e-6
        finite = (m.objective(point.detach()+h*direction, obs)-m.objective(point.detach()-h*direction, obs))/(2*h)
        torch.testing.assert_close((gradient*direction).sum(), finite[0], rtol=1e-6, atol=1e-6)

    def test_structured_covariance_fisher_matches_dense_full_state_derivatives(self):
        for mode in ("constant", "neural"):
            m = self.small(mode); state = self.state(m)[0][0].detach()
            elim = m.elimination_at(state[None])
            local = m.local_covariance_derivatives(state)
            mapping = m.local_state_map(state)
            actual = covariance_information(elim, local, mapping)
            def full_covariance(z):
                blocks = m.noise_covariance(z[None])[0]
                eye = torch.eye(m.nsteps, dtype=z.dtype)
                return torch.einsum("st,sab->satb", eye, blocks).reshape(m.nsteps*m.free_dof, -1)
            dc = torch.autograd.functional.jacobian(full_covariance, state, vectorize=True, strategy="forward-mode")
            g = m.load_mapping.flatten(0, 1)
            chol = torch.linalg.cholesky(full_covariance(state)+g@g.T)
            d = dc.permute(2, 0, 1)
            white = torch.linalg.solve_triangular(chol, d, upper=False)
            white = torch.linalg.solve_triangular(chol, white.transpose(-1, -2), upper=False).transpose(-1, -2)
            expected = .5*white.flatten(1)@white.flatten(1).T
            torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-8)
            self.assertGreater(float(torch.linalg.eigvalsh(actual).min()), -1e-8)

    def test_internal_noise_preserves_net_impulse_and_torque_before_supports(self):
        m = self.small(); state = self.state(m)[0]
        full = m.step_trajectory(state)
        factor = m.internal_factor(full.x[:, :-1], full.v[:, :-1])
        impulse = factor*m.mesh.mass[None, None, :, None, None]
        torch.testing.assert_close(impulse.sum(2), torch.zeros_like(impulse.sum(2)), atol=1e-16, rtol=0)
        torque = torch.linalg.cross(full.x[:, :-1, :, :, None].expand_as(impulse), impulse, dim=3).sum(2)
        torch.testing.assert_close(torque, torch.zeros_like(torque), atol=1e-16, rtol=0)

    def test_floor_model_reproduces_current_objective_and_precision(self):
        m = self.small("floor"); state = self.state(m)[0]
        old = AccurateViewModel(m.config).set_views(m.view_specs); old.renderer = m.renderer
        obs = Observations(m.view_frames, m.predict(state, m.view_frames)[0].detach(), .015)
        torch.testing.assert_close(m.objective(state, obs), old.objective(state, obs), rtol=1e-12, atol=1e-10)
        torch.testing.assert_close(m.fisher_precision(state[0], obs)[0], old.information(state[0], obs)[0], rtol=1e-12, atol=1e-10)

    def test_profile_offset_is_state_dependent_and_covariance_is_not_frozen(self):
        m = self.small(); state = self.state(m)[0].detach().requires_grad_()
        elim = m.elimination_at(state); residual = m.residual(state)
        conditional = elim.conditional_mean(residual)
        offset = .5*15*math.log(2*math.pi)-elim.cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        torch.testing.assert_close(elim.joint_nll(residual, conditional)-elim.marginal_nll(residual), offset, rtol=1e-11, atol=1e-11)
        grad, = torch.autograd.grad(offset.sum(), state)
        self.assertGreater(float(grad.norm()), 1e-5)
        self.assertFalse(any(p.requires_grad for p in m.network.parameters()))

    def test_force_mixture_averages_state_dependent_conditional_covariance(self):
        from physgauss.neural_marginal_inference import force_mixture
        from physgauss.marginal_inference import trajectory_gaussian_draws
        m = self.small(); center = self.state(m)[0][0].detach()
        fit = {"mean": center, "precision_cholesky": 50*torch.eye(len(center), dtype=center.dtype)}
        result = force_mixture(m, fit, 12, 7120)
        samples = trajectory_gaussian_draws(fit, 12, 7120)
        elimination = m.elimination_at(samples)
        means = elimination.conditional_mean(m.residual(samples))
        torch.testing.assert_close(result["mean"], means.mean(0), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(result["conditional_covariance"], elimination.conditional_covariance.mean(0), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(result["covariance"], elimination.conditional_covariance.mean(0)+torch.cov(means.T), rtol=1e-12, atol=1e-12)

    def test_independent_stochastic_reference_refines_and_has_correct_zero_noise_limit(self):
        from physgauss.stochastic_reference import coupled_reference, reference_path
        times = torch.linspace(0., .12, 5, dtype=torch.float64)
        result = coupled_reference(times, 4101, fine_substeps=32, tolerance_m=.001)
        for row in result["audit"]["grids"]:
            self.assertLess(row["maximum_internal_net_impulse_Ns"], 1e-14)
            self.assertLess(row["maximum_internal_impulse_torque_Nms"], 1e-14)
            self.assertEqual(row["support_position_error_m"], 0.)
        # Independent zero-noise drift accuracy uses the published converged
        # RK4 positions, rather than the inference integrator as ground truth.
        root = Path(__file__).resolve().parents[1]/"examples/simulation-view-audit/high_accuracy_reference/no_load.json"
        target = torch.tensor(json.loads(root.read_text())["positions_m"], dtype=torch.float64)
        t = torch.linspace(0., 1., 21, dtype=torch.float64)
        brownian = torch.zeros(20*96, 3, dtype=torch.float64)
        x, _, _ = reference_path(t, brownian, 96, stochastic=False)
        free = self.small().free
        self.assertLess(float((x[:, free]-target[:, free]).square().mean().sqrt()), .0001)

    def test_staggered_brownian_kicks_are_nested_and_stop_at_last_midpoint(self):
        from physgauss.stochastic_reference import staggered_increments
        raw = torch.arange(32, dtype=torch.float64).reshape(-1, 1).expand(-1, 3)
        actual = staggered_increments(raw, 4)
        expected = torch.stack([raw[a:b].sum(0) for a, b in ((0, 4), (4, 12), (12, 20), (20, 28))])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
