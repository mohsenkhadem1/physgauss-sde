import unittest
import torch
from physgauss.benchmark import Config, Observations, SceneModel, make_benchmark
from physgauss.forces import ForcePrior, ForceConfig, ForceSceneModel
from physgauss.force_inference import GaussianTargetCorrection, fitted_reference
from physgauss.mcmc import MixtureReference, SamplerConfig, sample_nonlinear
from physgauss.scale_inference import LinearComponent, ScalePosterior
from physgauss.stochastic import NeuralDiffusion


class ForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_temporal_projection_has_declared_inducing_covariance(self):
        model = ForceSceneModel(Config.quick())
        prior = model.force_prior
        phi = prior.features(prior.knots)
        torch.testing.assert_close(phi@phi.T, prior.kernel(prior.knots, prior.knots), atol=2e-8, rtol=2e-8)
        self.assertEqual(prior.dimension, 15)
        with self.assertRaises(ValueError):
            prior.features(torch.tensor([-.1], dtype=torch.float64))

    def test_nodal_forces_have_newton_units_and_correct_total(self):
        model = ForceSceneModel(Config.quick())
        gen = torch.Generator().manual_seed(23)
        z = torch.randn(3, 15, dtype=torch.float64, generator=gen)
        prior = model.force_prior
        total = prior.total_force(z, model.step_times)
        nodal = prior.nodal_force(z, model.step_times)
        torch.testing.assert_close(nodal.sum(2), total, atol=1e-15, rtol=1e-12)
        self.assertEqual(float(nodal[:, :, model.mesh.pinned].abs().sum()), 0.)

    def test_disabled_force_matches_original_solver_and_logging_is_passive(self):
        config = Config.quick()
        diffusion = NeuralDiffusion()
        model = ForceSceneModel(config, diffusion, force=False)
        original = SceneModel(config, diffusion)
        gen = torch.Generator().manual_seed(8)
        z = .1*torch.randn(2, model.dimension, dtype=torch.float64, generator=gen)
        with torch.no_grad():
            baseline = original.trajectory(z)
            recorded, forces, impulses = model.force_record(z)
            regular = model.trajectory(z)
        torch.testing.assert_close(recorded.x, baseline.x, atol=1e-14, rtol=1e-12)
        torch.testing.assert_close(recorded.x, regular.x, atol=0, rtol=0)
        self.assertEqual(float(forces.abs().sum()), 0.)
        torch.testing.assert_close(impulses.sum(2), torch.zeros_like(impulses.sum(2)), atol=1e-18, rtol=0)

    def test_force_coefficients_have_correct_image_gradients_and_no_truth_flag_leak(self):
        config = Config.quick()
        model = ForceSceneModel(config)
        z = torch.zeros(1, model.dimension, dtype=torch.float64); z[0, 14] = .2
        variable = z.clone().requires_grad_()
        loss = model.predict(variable).square().sum()
        grad, = torch.autograd.grad(loss, variable)
        shift = torch.zeros_like(z); shift[0, 14] = 1e-5
        with torch.no_grad():
            fd = (model.predict(z+shift).square().sum()-model.predict(z-shift).square().sum())/(2e-5)
            other = Config.quick(); other.truth_external_pulse = True; other.truth_stochastic = False
            torch.testing.assert_close(model.predict(z), ForceSceneModel(other).predict(z), rtol=0, atol=0)
        self.assertAlmostEqual(float(grad[0, 14]), float(fd), delta=1e-7+abs(float(fd))*1e-5)
        self.assertGreater(abs(float(fd)), 1e-8)

    def test_arbitrary_gaussian_correction_matches_normalized_density_ratio(self):
        jac = torch.tensor([[1., .4], [-.3, 1.]], dtype=torch.float64)
        class LinearModel:
            def predict(self, z, indices):
                return z@jac.T
        obs = Observations(torch.tensor([0]), jac.new_tensor([.3, -.2]), .7)
        mean = jac.new_tensor([.1, -.4])
        chol = torch.linalg.cholesky(jac.new_tensor([[2., .3], [.3, 1.5]]))
        component = LinearComponent(1., mean, chol, 0.)
        posterior = ScalePosterior([component], jac.new_ones(1), jac.new_ones(1), mean, jac.new_zeros(1), jac)
        ref = MixtureReference(posterior)
        correction = GaussianTargetCorrection(LinearModel(), obs, ref)
        z = jac.new_tensor([[.5, -.2], [1., .8]])
        values, grad = correction.value_and_grad(z, jac.new_ones(2))
        prior = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=jac.dtype), torch.eye(2, dtype=jac.dtype))
        q = torch.distributions.MultivariateNormal(mean, precision_matrix=chol@chol.T)
        likelihood = -.5*((z@jac.T-obs.rgb)/obs.sigma).square().sum(1)
        torch.testing.assert_close(values[0], prior.log_prob(z)+likelihood-q.log_prob(z))
        expected_grad = -(z@jac.T-obs.rgb)@jac/obs.sigma**2-z+(z-mean)@(chol@chol.T)
        torch.testing.assert_close(grad, expected_grad)
        chain = sample_nonlinear(ref, correction, SamplerConfig(warmup=200, draws=3000,
            independent_probability=.25, transport_probability=0., langevin=True, seed=171))
        precision = torch.eye(2, dtype=jac.dtype)+jac.T@jac/obs.sigma**2
        expected_covariance = torch.linalg.inv(precision)
        expected_mean = torch.linalg.solve(precision, jac.T@obs.rgb/obs.sigma**2)
        samples = chain["coordinates"].flatten(0, 1)
        torch.testing.assert_close(samples.mean(0), expected_mean, atol=.035, rtol=.04)
        torch.testing.assert_close(torch.cov(samples.T), expected_covariance, atol=.03, rtol=.08)

    def test_nonlinear_curvature_matches_exact_linear_gaussian_precision(self):
        from types import SimpleNamespace
        jac = torch.tensor([[1., .4], [-.3, 1.], [.2, -.1]], dtype=torch.float64)
        class LinearModel:
            dimension = 2
            config = SimpleNamespace(adam_steps=0, lbfgs_steps=12, finite_difference_chunk=2)
            def predict(self, z, indices):
                return z@jac.T
        obs = Observations(torch.tensor([0]), jac.new_tensor([.3, -.2, .4]), .7)
        reference, record = fitted_reference(LinearModel(), obs, progress=None, exact_hessian=True)
        component = reference.components[0]
        expected = torch.eye(2, dtype=jac.dtype)+jac.T@jac/obs.sigma**2
        torch.testing.assert_close(component.precision_cholesky@component.precision_cholesky.T,
                                   expected, atol=1e-9, rtol=1e-9)
        self.assertEqual(record["proposal_precision"]["floored_eigenvalues"], 0)

    def test_truth_force_record_reproduces_pulse_and_stochastic_trajectory(self):
        from physgauss.force_study import truth_force_record
        config = Config.quick(); config.truth_external_pulse = True
        benchmark = make_benchmark(config)
        force, impulse = truth_force_record(config, benchmark)
        mesh = ForceSceneModel(config).mesh
        self.assertGreater(float(force.abs().sum()), 0.)
        self.assertGreater(float(impulse.abs().sum()), 0.)
        self.assertEqual(float(force[:, :, mesh.pinned].abs().sum()), 0.)
        torch.testing.assert_close(impulse.sum(2), torch.zeros_like(impulse.sum(2)), atol=1e-18, rtol=0)

    def test_indefinite_curvature_falls_back_to_positive_information_metric(self):
        from types import SimpleNamespace
        class QuadraticObservation:
            dimension = 1
            config = SimpleNamespace(adam_steps=0, lbfgs_steps=0, finite_difference_chunk=1)
            def predict(self, z, indices):
                return z.square()
        observations = Observations(torch.tensor([0]), torch.ones(1, dtype=torch.float64), .5)
        reference, record = fitted_reference(QuadraticObservation(), observations, progress=None, exact_hessian=True)
        self.assertLess(record["proposal_precision"]["minimum_raw_eigenvalue"], 0)
        self.assertIn("fallback", record["proposal_precision"]["method"])
        torch.testing.assert_close(reference.components[0].precision_cholesky, torch.ones(1, 1, dtype=torch.float64))

    def test_window_average_preserves_impulse_and_declared_units(self):
        from physgauss.force_study import window_average_force
        dt = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
        forces = torch.full((2, 4, 3, 3), 2., dtype=dt.dtype)
        impulses = torch.zeros_like(forces); impulses[:, 1, 0, 0] = .6
        averaged = window_average_force(forces, impulses, dt, substeps=2)
        self.assertAlmostEqual(float(averaged[0, 0, 0, 0]), 4.)
        self.assertAlmostEqual(float(averaged[0, 1, 0, 0]), 2.)
        duration = dt.reshape(-1, 2).sum(1)
        integrated = (averaged*duration[None, :, None, None]).sum(1)
        expected = (forces*dt[None, :, None, None]+impulses).sum(1)
        torch.testing.assert_close(integrated, expected)


if __name__ == "__main__":
    unittest.main()
