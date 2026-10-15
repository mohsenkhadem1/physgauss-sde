import unittest
import torch
from physgauss.benchmark import Config, SceneModel, make_benchmark
from physgauss.inference import Fit, objective, sample_coordinates
from physgauss.metrics import crps, energy_score, interval_metrics
from physgauss.stochastic import NeuralDiffusion


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_known_empirical_probabilistic_scores(self):
        samples = torch.tensor([[[-1.]], [[1.]]], dtype=torch.float64)
        truth = torch.zeros(1, 1, dtype=torch.float64)
        self.assertAlmostEqual(float(crps(samples, truth)), .5)
        self.assertAlmostEqual(float(energy_score(samples, truth)), .5)
        intervals = interval_metrics(samples, truth)
        self.assertEqual(intervals["coverage"], 1)
        self.assertAlmostEqual(intervals["mean_width_m"], 1.8)

    def test_precision_sampling_orientation(self):
        h = torch.tensor([[4., 1.], [1., 2.]], dtype=torch.float64)
        fit = Fit(torch.zeros(2, dtype=torch.float64), torch.linalg.cholesky(h), [], 0, 0, 0)
        draws = sample_coordinates(fit, 40000, seed=92)
        torch.testing.assert_close(torch.cov(draws.T), torch.linalg.inv(h), rtol=.04, atol=.01)

    def test_latent_noise_has_observation_gradients_and_batches_match(self):
        c = Config.quick()
        torch.manual_seed(7)
        net = NeuralDiffusion()
        model = SceneModel(c, net)
        z = torch.zeros(1, model.dimension, dtype=torch.float64, requires_grad=True)
        benchmark = make_benchmark(c)
        loss = objective(model, benchmark.observations, z)
        gradient, = torch.autograd.grad(loss, z)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient[:, 8:].abs().max()), 1e-7)
        with torch.no_grad():
            batched = model.predict(z.detach().expand(2, -1), benchmark.observations.indices)
            single = model.predict(z.detach(), benchmark.observations.indices)
        torch.testing.assert_close(batched[0], single[0])
        torch.testing.assert_close(batched[1], single[0])

    def test_withheld_truth_is_not_an_observation_alias(self):
        benchmark = make_benchmark(Config.quick())
        before = benchmark.observations.rgb.clone()
        benchmark.truth_rgb.fill_(99)
        benchmark.truth.x.fill_(99)
        torch.testing.assert_close(benchmark.observations.rgb, before)

    def test_bad_observation_schedules_are_rejected(self):
        for obs in [(0, 0, 20), (0, 7, 4, 20), (0, 3), (0, 99)]:
            with self.assertRaises(ValueError):
                Config(observed_frames=obs).validate()


if __name__ == "__main__":
    unittest.main()
