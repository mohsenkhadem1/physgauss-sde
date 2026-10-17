import json
import math
from pathlib import Path
import unittest
import torch
from physgauss.benchmark import Config, Observations
from physgauss.accurate_dynamics import StaggeredForceModel
from physgauss.continuous_render import ContinuousGaussianRenderer, VolumeRenderConfig
from physgauss.marginal_forces import TrajectoryForceModel


class AccuratePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def small(self):
        c = Config.quick(); c.frames = 5; c.substeps = 1; c.observed_frames = (0, 4)
        return StaggeredForceModel(c)

    def test_staggered_causal_density_jacobian_and_force_innovations(self):
        m = self.small(); gen = torch.Generator().manual_seed(6601)
        theta = .12*torch.randn(2, 8, dtype=torch.float64, generator=gen)
        a = .2*torch.randn(2, 15, dtype=torch.float64, generator=gen)
        white = torch.randn(2, m.nsteps, m.free_dof, dtype=torch.float64, generator=gen)
        state = m.rollout(theta, a, white)
        torch.testing.assert_close(m.residual(state), a@m.elimination.mapping.T+white.flatten(1)*m.elimination.sd, atol=2e-15, rtol=1e-10)
        jac = torch.autograd.functional.jacobian(lambda q: m.residual(q[None])[0], state[0])[:, 8:]
        expected = m.log_dynamic_jacobian(state[:1])[0]+jac.shape[0]*math.log(m.marginal_config.velocity_coordinate_scale)
        torch.testing.assert_close(torch.linalg.slogdet(jac)[1], expected, atol=1e-12, rtol=1e-12)
        value = state[:1].detach().requires_grad_()
        derivative, = torch.autograd.grad(m.log_dynamic_jacobian(value).sum(), value)
        h = 1e-5; shift = torch.zeros_like(value); shift[0, 1] = h
        fd = (m.log_dynamic_jacobian(value.detach()+shift)-m.log_dynamic_jacobian(value.detach()-shift))/(2*h)
        torch.testing.assert_close(derivative[:, 1], fd, rtol=1e-8, atol=1e-9)

    def test_staggered_accuracy_and_second_order_refinement_against_RK4(self):
        root = Path(__file__).resolve().parents[1]/"examples/simulation-view-audit/high_accuracy_reference"
        for condition in ("no_load", "external_pulse"):
            ref = json.loads((root/f"{condition}.json").read_text())
            errors = []
            for substeps in (3, 6):
                m = StaggeredForceModel(Config(substeps=substeps))
                raw = m.times.new_tensor([[math.log(19.5), math.log(.55), .11, -.035, .06, -.12, .04, .10]])
                theta = (raw-m.prior.mean)/m.prior.std
                # Analytic pulse loading is evaluated directly, avoiding force
                # basis approximation in this integrator-accuracy check.
                def external(t, x, previous):
                    strength = math.exp(-.5*((t-.5)/.08)**2) if condition == "external_pulse" else 0.
                    return strength*m.free[None, :, None]*m.times.new_tensor([.45, 0., -.30])
                with torch.no_grad():
                    state = m.rollout(theta, torch.zeros(1, 15, dtype=torch.float64), external_acceleration=external)
                    target = m.times.new_tensor(ref["positions_m"])
                    error = m.trajectory(state).x[0, :, m.free]-target[:, m.free]
                    errors.append(float(error.square().mean().sqrt()))
            self.assertLess(errors[1], .00015)
            self.assertGreater(errors[0]/errors[1], 3.5)

    def test_continuous_volume_is_permutation_invariant_and_energy_bounded(self):
        m = self.small(); renderer = ContinuousGaussianRenderer(m.mesh, 12, VolumeRenderConfig(bins=32))
        theta = torch.zeros(1, 8, dtype=torch.float64)
        state = m.rollout(theta, torch.zeros(1, 15, dtype=torch.float64))
        x = m.trajectory(state).x[:, [2]]
        optical, fractions = renderer.components(x, m.eye[[2]], m.rotation[[2]])
        permutation = torch.randperm(30, generator=torch.Generator().manual_seed(6602))
        expected = renderer.integrate(optical, fractions, renderer.colors)
        actual = renderer.integrate(optical[:, :, permutation], fractions[:, :, :, permutation], renderer.colors[permutation])
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-13)
        self.assertGreaterEqual(float(actual.min()), .035-1e-12)
        self.assertLessEqual(float(actual.max()), float(renderer.colors.max())+1e-12)
        colors = torch.full_like(renderer.colors, .4)
        transmission = torch.exp(-optical.sum(2))[..., None]
        analytic = .4*(1-transmission)+.035*transmission
        torch.testing.assert_close(renderer.integrate(optical, fractions, colors), analytic.expand_as(expected), atol=1e-13, rtol=1e-13)

    def test_tied_depth_rendering_has_continuous_finite_difference_gradients(self):
        m = self.small(); r = ContinuousGaussianRenderer(m.mesh, 12, VolumeRenderConfig(bins=48))
        z = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        direction = (~m.mesh.pinned).to(z.dtype)[None, None, :, None]*z.new_tensor([.6, -.1, .3])
        x = m.mesh.rest[None, None]+z*direction
        value = r(x, m.eye[[0]], m.rotation[[0]]).square().sum()
        gradient, = torch.autograd.grad(value, z)
        for h in (1e-5, 1e-7):
            with torch.no_grad():
                plus = r(m.mesh.rest[None, None]+h*direction, m.eye[[0]], m.rotation[[0]])
                minus = r(m.mesh.rest[None, None]-h*direction, m.eye[[0]], m.rotation[[0]])
                fd = (plus.square().sum()-minus.square().sum())/(2*h)
            torch.testing.assert_close(fd, gradient, atol=2e-6, rtol=1e-6)

    def test_staggered_profile_marginal_identity_includes_same_jacobian(self):
        m = self.small(); m.renderer = ContinuousGaussianRenderer(m.mesh, 12, VolumeRenderConfig(bins=24))
        state = m.rollout(torch.zeros(1, 8, dtype=torch.float64), torch.zeros(1, 15, dtype=torch.float64))
        obs = Observations(torch.tensor([0, 4]), m.predict(state, torch.tensor([0, 4]))[0].detach(), .015)
        a = m.elimination.conditional_mean(m.residual(state))
        offset = .5*15*math.log(2*math.pi)-m.elimination.cholesky.diagonal().log().sum()
        torch.testing.assert_close(m.objective(state, obs, a)-m.objective(state, obs), offset[None], atol=1e-11, rtol=1e-12)

    def test_assembled_dynamics_jacobians_match_full_path_AD(self):
        from physgauss.accurate_pipeline import dynamics_jacobian
        gen = torch.Generator().manual_seed(6603)
        for model in (self.small(), TrajectoryForceModel(self.small().config)):
            theta = .15*torch.randn(1, 8, dtype=torch.float64, generator=gen)
            force = .12*torch.randn(1, 15, dtype=torch.float64, generator=gen)
            state = model.rollout(theta, force)[0]
            assembled = dynamics_jacobian(model, state)
            exact = torch.autograd.functional.jacobian(lambda q: model.residual(q[None])[0], state,
                                                       vectorize=True, strategy="forward-mode")
            torch.testing.assert_close(assembled, exact, rtol=1e-10, atol=1e-11)

    def test_multicamera_information_matches_independent_full_path_AD(self):
        from physgauss.accurate_pipeline import AccurateViewModel
        c = self.small().config
        m = AccurateViewModel(c).set_views([(0, 0), (4, 0), (4, 100)])
        m.renderer = ContinuousGaussianRenderer(m.mesh, 12, VolumeRenderConfig(bins=24))
        state = m.rollout(torch.zeros(1, 8, dtype=torch.float64), torch.zeros(1, 15, dtype=torch.float64))[0]
        obs = Observations(m.view_frames, m.predict(state[None], m.view_frames)[0].detach(), .015)
        actual, _, _ = m.information(state, obs)
        ji = torch.autograd.functional.jacobian(lambda q: m.image_residual(q[None], obs)[0], state,
                                               vectorize=True, strategy="forward-mode")
        jd = torch.autograd.functional.jacobian(lambda q: m.residual(q[None])[0], state,
                                               vectorize=True, strategy="forward-mode")
        white = m.elimination.whiten(jd.T).T
        prior = torch.zeros_like(actual); prior[:8, :8] = torch.eye(8, dtype=state.dtype)
        expected = prior+ji.T@ji+white.T@white
        expected[1, 1] += m.normalizer_curvature(state)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-9)

    def test_accurate_model_does_not_read_truth_flags(self):
        c = self.small().config
        m = StaggeredForceModel(c)
        state = m.rollout(torch.zeros(1, 8, dtype=torch.float64), torch.zeros(1, 15, dtype=torch.float64))
        expected = m.residual(state)
        c.truth_external_pulse = True; c.truth_stochastic = not c.truth_stochastic
        torch.testing.assert_close(m.residual(state), expected, atol=0., rtol=0.)


if __name__ == "__main__":
    unittest.main()
