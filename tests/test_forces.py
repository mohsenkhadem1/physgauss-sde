import unittest
import torch
from physgauss.benchmark import Config, Observations, SceneModel, make_benchmark
from physgauss.forces import ForcePrior, ForceConfig, ForceSceneModel, evaluate_force_posterior
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
        # The renderer sorts by depth. At symmetric scenes a finite difference
        # can jump compositing branches; a derivative does not exist there.
        # Use generic states and explicitly verify the same ordering at +/- h.
        from physgauss.camera import project
        generator = torch.Generator().manual_seed(319)
        frame = torch.tensor([config.frames-1])
        shift = torch.zeros(1, model.dimension, dtype=torch.float64); shift[0, 14] = 1e-5
        def order_at(point):
            trajectory = model.trajectory(point)
            nodes = trajectory.x[:, frame][:, :, model.mesh.cells]
            centers = torch.einsum("pk,bftkd->bftpd", model.renderer.bary, nodes).flatten(2, 3)
            _, depth, _ = project(centers, model.eye[frame], model.rotation[frame], config.image_size)
            return depth.argsort(-1)
        with torch.no_grad():
            for _ in range(20):
                z = .15*torch.randn(1, model.dimension, dtype=torch.float64, generator=generator)
                if torch.equal(order_at(z+shift), order_at(z-shift)):
                    break
            else:
                self.fail("No smooth depth-order branch found for the force gradient check.")
        variable = z.clone().requires_grad_()
        loss = model.predict(variable, frame).square().sum()
        grad, = torch.autograd.grad(loss, variable)
        with torch.no_grad():
            fd = (model.predict(z+shift, frame).square().sum()-model.predict(z-shift, frame).square().sum())/(2e-5)
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

    def test_sorted_force_crps_matches_independent_pairwise_definition(self):
        from physgauss.force_study import sorted_crps
        from physgauss.metrics import crps
        generator = torch.Generator().manual_seed(17)
        samples = torch.randn(17, 5, 3, dtype=torch.float64, generator=generator)
        target = torch.randn(5, 3, dtype=torch.float64, generator=generator)
        torch.testing.assert_close(sorted_crps(samples, target), crps(samples, target), atol=1e-14, rtol=1e-14)

    def test_portable_coefficient_posterior_reconstructs_force_covariance_and_draws(self):
        from dataclasses import asdict
        model = ForceSceneModel(Config.quick())
        prior = model.force_prior
        generator = torch.Generator().manual_seed(991)
        coefficients = torch.randn(32, prior.dimension, dtype=torch.float64, generator=generator)
        fields = {"duration_s": model.config.duration, "inducing_times_s": prior.knots.tolist(),
            "spatial_nodal_weights": prior.spatial_weights.tolist(), "force_prior": asdict(prior.config),
            "whitened_force_coefficient_mean": coefficients.mean(0).tolist(),
            "whitened_force_coefficient_covariance": torch.cov(coefficients.T).tolist(),
            "whitened_force_coefficient_draws": coefficients.tolist()}
        times = model.times.new_tensor([.03, .16, .37])
        posterior = evaluate_force_posterior(fields, times)
        target = prior.total_force(coefficients, times)
        torch.testing.assert_close(posterior["total_draws_N"], target, atol=1e-14, rtol=1e-12)
        torch.testing.assert_close(posterior["mean_total_N"], target.mean(0), atol=1e-14, rtol=1e-12)
        torch.testing.assert_close(posterior["joint_total_covariance_N2"], torch.cov(target.flatten(1).T), atol=1e-14, rtol=1e-12)
        torch.testing.assert_close(posterior["nodal_draws_N"], prior.nodal_force(coefficients, times), atol=1e-14, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
