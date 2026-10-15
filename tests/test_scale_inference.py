import math
import unittest
import torch
from physgauss.scale_inference import (ScalePosterior, draw_coordinates,
    linear_component, scale_posterior_from_linearization, scale_prior)


class ScaleInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_prior_is_normalized_and_keeps_zero_atom(self):
        p = scale_prior()
        self.assertAlmostEqual(float(p.sum()), 1.)
        self.assertAlmostEqual(float(p[0]), .5)
        self.assertTrue(bool((p > 0).all()))
        with self.assertRaises(ValueError):
            scale_prior((0., 1., .5))

    def test_low_rank_evidence_matches_dense_observation_covariance(self):
        gen = torch.Generator().manual_seed(12)
        jac = torch.randn(7, 4, generator=gen, dtype=torch.float64)
        data = torch.randn(7, generator=gen, dtype=torch.float64)
        for scale in [0., .25, 1., 3.]:
            component = linear_component(jac, data, 2, scale)
            a = jac * jac.new_tensor([1., 1., scale, scale])
            covariance = torch.eye(7, dtype=a.dtype) + a @ a.T
            dense_nle = .5 * (data @ torch.linalg.solve(covariance, data)
                              + torch.linalg.slogdet(covariance).logabsdet)
            self.assertAlmostEqual(component.negative_log_evidence, float(dense_nle), places=10)
            dense_mean = a.T @ torch.linalg.solve(covariance, data)
            dense_cov = torch.eye(4, dtype=a.dtype) - a.T @ torch.linalg.solve(covariance, a)
            torch.testing.assert_close(component.mean, dense_mean, atol=1e-12, rtol=1e-10)
            torch.testing.assert_close(torch.cholesky_inverse(component.precision_cholesky),
                                       dense_cov, atol=1e-12, rtol=1e-10)

    def test_evidence_responds_to_noise_in_observed_subspace(self):
        jac = torch.eye(5, dtype=torch.float64)
        _, prior, quiet = scale_posterior_from_linearization(jac, torch.zeros(5, dtype=jac.dtype), 1)
        data = torch.tensor([0., 8., -7., 6., -8.], dtype=jac.dtype)
        _, _, active = scale_posterior_from_linearization(jac, data, 1)
        self.assertGreater(float(quiet[0]), float(prior[0]))
        self.assertLess(float(active[0]), 1e-8)

    def test_unobserved_noise_preserves_scale_prior(self):
        jac = torch.zeros(4, 6, dtype=torch.float64)
        jac[:, :2] = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., -1.]])
        _, prior, posterior = scale_posterior_from_linearization(jac, torch.ones(4, dtype=jac.dtype), 2)
        torch.testing.assert_close(prior, posterior, atol=1e-12, rtol=1e-10)

    def test_zero_scale_matches_marginal_physical_model(self):
        gen = torch.Generator().manual_seed(66)
        jac = torch.randn(9, 5, generator=gen, dtype=torch.float64)
        data = torch.randn(9, generator=gen, dtype=torch.float64)
        augmented = linear_component(jac, data, 2, 0.)
        physical = linear_component(jac[:, :2], data, 2, 0.)
        self.assertAlmostEqual(augmented.negative_log_evidence, physical.negative_log_evidence, places=11)
        torch.testing.assert_close(augmented.mean[:2], physical.mean)
        torch.testing.assert_close(augmented.mean[2:], torch.zeros(3, dtype=jac.dtype))

    def test_mixture_sampling_preserves_scale_and_latent_correlations(self):
        jac = torch.tensor([[1., .5], [0., 1.], [1., -.3]], dtype=torch.float64)
        data = torch.tensor([.2, -.3, .1], dtype=jac.dtype)
        components, prior, weights = scale_posterior_from_linearization(jac, data, 1, scales=(0., 1.))
        posterior = ScalePosterior(components, prior, weights, torch.zeros(2, dtype=jac.dtype),
                                   torch.zeros(3, dtype=jac.dtype), jac)
        x, scales, ids = draw_coordinates(posterior, 30000, seed=19)
        for i, component in enumerate(components):
            selected = x[ids == i]
            self.assertGreater(len(selected), 200)
            self.assertAlmostEqual(float((ids == i).double().mean()), float(weights[i]), delta=.015)
            torch.testing.assert_close(selected.mean(0), component.mean, atol=.025, rtol=.08)
            torch.testing.assert_close(torch.cov(selected.T), torch.cholesky_inverse(component.precision_cholesky),
                                       atol=.025, rtol=.08)
            self.assertTrue(bool((scales[ids == i] == component.scale).all()))


if __name__ == "__main__":
    unittest.main()
