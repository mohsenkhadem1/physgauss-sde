import math
import unittest
import torch
from physgauss.scale_inference import ScalePosterior, LinearComponent, scale_posterior_from_linearization
from physgauss.mcmc import (MixtureReference, SamplerConfig, sample_nonlinear,
    metropolis_log_acceptance, langevin_log_correction)
from physgauss.chain_diagnostics import rank_normalize, scalar_diagnostics


def reference_for(jac, data, scene_dimensions, scales):
    components, prior, weights = scale_posterior_from_linearization(jac, data, scene_dimensions, scales)
    return ScalePosterior(components, prior, weights, jac.new_zeros(jac.shape[1]),
                          jac.new_zeros(jac.shape[0]), jac)


class NonlinearSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_linear_target_recovers_gaussian_mixture_and_frozen_adaptation(self):
        jac = torch.tensor([[1., 1.], [.3, -.6]], dtype=torch.float64)
        posterior = reference_for(jac, jac.new_tensor([.5, -.2]), 1, (0., .5, 1.))
        ref = MixtureReference(posterior)
        def correction(z, scales):
            return z.new_zeros(len(z)), z.new_zeros(len(z)), torch.zeros(len(z), dtype=torch.bool)
        config = SamplerConfig(warmup=100, draws=4000, seed=123)
        result = sample_nonlinear(ref, correction, config)
        z = result["coordinates"].flatten(0, 1)
        expected = sum(w*c.mean for w, c in zip(posterior.posterior_weights, posterior.components))
        torch.testing.assert_close(z.mean(0), expected, atol=.04, rtol=.03)
        for i, w in enumerate(posterior.posterior_weights):
            self.assertAlmostEqual(float((result["component_indices"] == i).double().mean()), float(w), delta=.035)
        self.assertEqual(result["transition_counts"]["retained"]["pcn_attempts"],
                         result["transition_counts"]["retained"]["pcn_accepts"])
        # Repeated short runs include identical warmup and retained random draws.
        short = SamplerConfig(warmup=25, draws=20)
        a, b = sample_nonlinear(ref, correction, short), sample_nonlinear(ref, correction, short)
        torch.testing.assert_close(a["coordinates"], b["coordinates"], rtol=0, atol=0)
        self.assertEqual(a["final_beta_per_chain"], b["final_beta_per_chain"])

    def test_nonlinear_target_matches_numerical_quadrature(self):
        # Nonlinear y = z + .3 z^2, N(0,1) prior; local reference y=z.
        jac = torch.ones(1, 1, dtype=torch.float64)
        posterior = reference_for(jac, jac.new_tensor([1.]), 1, (0., 1.))
        ref = MixtureReference(posterior, fixed_scale=0.)
        def correction(z, scales):
            linear = -.5*(z[:, 0]-1).square()
            actual = -.5*(z[:, 0]+.3*z[:, 0].square()-1).square()
            return actual-linear, actual, torch.zeros(len(z), dtype=torch.bool)
        result = sample_nonlinear(ref, correction, SamplerConfig(warmup=500, draws=6000, seed=219))
        grid = torch.linspace(-8, 6, 20001, dtype=torch.float64)
        density = torch.exp(-.5*(grid+.3*grid.square()-1).square()-.5*grid.square())
        weights = density/density.sum()
        samples = result["coordinates"][..., 0].flatten()
        self.assertAlmostEqual(float(samples.mean()), float(weights@grid), delta=.04)
        self.assertAlmostEqual(float(samples.square().mean()), float(weights@grid.square()), delta=.06)

    def test_transport_acceptance_includes_component_weights_and_jacobian(self):
        # Direct transformed densities agree with the implemented acceptance rule.
        old = torch.tensor([0]); new = torch.tensor([1]); dtype = torch.float64
        logw = torch.tensor([.8, .2], dtype=dtype).log()
        logg = torch.tensor([.6, .4], dtype=dtype).log()
        ratio = torch.tensor([-.3], dtype=dtype); proposed = torch.tensor([.7], dtype=dtype)
        answer = metropolis_log_acceptance(ratio, proposed, old, new, logw, logg, torch.tensor([True]))
        expected = .7-(-.3) + math.log(.2/.8) + math.log(.6/.4)
        self.assertAlmostEqual(float(answer[0]), expected, places=12)

    def test_underflowed_reference_component_retains_log_support(self):
        posterior = reference_for(torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64), 1, (0., 1.))
        posterior.components[1].negative_log_evidence = 10000.
        ref = MixtureReference(posterior)
        self.assertTrue(bool(torch.isfinite(ref.log_weights).all()))
        self.assertEqual(float(ref.log_weights[1].exp()), 0.)
        proposal = .95*ref.log_weights.exp()+.05*ref.prior_weights
        self.assertGreater(float(proposal[1]), 0.)

    def test_zero_target_states_rejected_and_counted(self):
        posterior = reference_for(torch.ones(1, 1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64), 1, (0., 1.))
        ref = MixtureReference(posterior, fixed_scale=0.)
        calls = [0]
        def correction(z, scales):
            calls[0] += 1
            invalid = torch.full((len(z),), calls[0] > 1, dtype=torch.bool)
            return torch.where(invalid, -torch.inf, z.new_zeros(len(z))), z.new_zeros(len(z)), invalid
        result = sample_nonlinear(ref, correction, SamplerConfig(warmup=0, draws=20))
        self.assertEqual(result["transition_counts"]["retained"]["invalid_proposals"], [20]*4)
        self.assertTrue(bool((result["coordinates"] == result["coordinates"][:, :1]).all()))

    def test_rank_ties_are_averaged(self):
        values = torch.tensor([[1., 1., 2., 3.]], dtype=torch.float64)
        ranks = values.new_tensor([[1.5, 1.5, 3., 4.]])
        torch.testing.assert_close(rank_normalize(values), torch.special.ndtri((ranks-.375)/4.25))

    def test_langevin_metropolis_ratio_matches_direct_normal_densities(self):
        old = torch.tensor([[.1, -.2], [1., .3]], dtype=torch.float64)
        new = torch.tensor([[.6, .8], [-.5, 1.]], dtype=torch.float64)
        g0, g1 = old*.3, new*.7
        beta = old.new_tensor([.4, .8])
        a = (1-beta.square()).sqrt()[:, None]
        forward = torch.distributions.Normal(a*old+(1-a)*g0, beta[:, None]).log_prob(new).sum(1)
        reverse = torch.distributions.Normal(a*new+(1-a)*g1, beta[:, None]).log_prob(old).sum(1)
        expected = reverse-forward-.5*(new.square().sum(1)-old.square().sum(1))
        torch.testing.assert_close(langevin_log_correction(old, new, g0, g1, beta), expected)

    def test_langevin_recovers_nonlinear_scale_mixture_quadrature(self):
        jac = torch.tensor([[0., 1.]], dtype=torch.float64)
        posterior = reference_for(jac, jac.new_tensor([1.]), 1, (0., .5, 1., 2.))
        ref = MixtureReference(posterior)
        class Correction:
            def value_and_grad(self, z, scales):
                raw = z[:, 1]*scales
                actual = -.5*(raw+.3*raw.square()-1).square()
                linear = -.5*(raw-1).square()
                grad = torch.zeros_like(z)
                grad[:, 1] = (-(raw+.3*raw.square()-1)*(1+.6*raw)+(raw-1))*scales
                return (actual-linear, actual, torch.zeros(len(z), dtype=torch.bool)), grad
        result = sample_nonlinear(ref, Correction(), SamplerConfig(warmup=300, draws=4000, seed=612, langevin=True))
        grid = torch.linspace(-8, 8, 20001, dtype=torch.float64)
        evidence = []
        for component in posterior.components:
            raw = grid*component.scale
            evidence.append(torch.trapezoid(torch.exp(-.5*(raw+.3*raw.square()-1).square()-.5*grid.square()), grid))
        expected = torch.stack(evidence)*posterior.prior_weights
        expected /= expected.sum()
        for i, p in enumerate(expected):
            self.assertAlmostEqual(float((result["component_indices"] == i).double().mean()), float(p), delta=.035)

    def test_physical_nonlinear_correction_gradient_matches_finite_difference(self):
        from physgauss.benchmark import Config, SceneModel, make_benchmark
        from physgauss.nonlinear_study import NonlinearCorrection
        from physgauss.stochastic import NeuralDiffusion
        config = Config.quick()
        model = SceneModel(config, NeuralDiffusion())
        obs = make_benchmark(config).observations
        jac = torch.zeros(obs.rgb.numel(), model.dimension, dtype=torch.float64)
        posterior = reference_for(jac, jac.new_zeros(len(jac)), 8, (0., 1.))
        posterior.reference_rgb = model.predict(posterior.reference[None], obs.indices)[0].detach()
        z = posterior.reference[None].clone(); z[0, 0] = .15; z[0, 8] = .4
        scales = z.new_tensor([1.])
        correction = NonlinearCorrection(model, obs, posterior)
        values, grad = correction.value_and_grad(z, scales)
        for column in [0, 3, 8]:
            shift = torch.zeros_like(z); shift[0, column] = 1e-5
            fd = (correction(z+shift, scales)[0]-correction(z-shift, scales)[0])/(2e-5)
            self.assertAlmostEqual(float(grad[0, column]), float(fd[0]), delta=1e-3+abs(float(fd[0]))*1e-5)

    def test_diagnostics_detect_shift_scale_and_frozen_chains(self):
        gen = torch.Generator().manual_seed(19)
        normal = torch.randn(4, 4000, dtype=torch.float64, generator=gen)
        self.assertTrue(scalar_diagnostics(normal)["diagnostic_pass"])
        shifted = normal.clone(); shifted[0] += 2
        self.assertFalse(scalar_diagnostics(shifted)["diagnostic_pass"])
        scaled = normal.clone(); scaled[0] *= .1
        self.assertGreater(scalar_diagnostics(scaled)["rank_folded_split_rhat"], 1.01)
        constant = scalar_diagnostics(torch.zeros_like(normal))
        self.assertIsNone(constant["rank_folded_split_rhat"])
        self.assertFalse(constant["diagnostic_pass"])

    def test_ess_tracks_known_ar1_autocorrelation(self):
        gen = torch.Generator().manual_seed(782)
        innovations = torch.randn(4, 12000, dtype=torch.float64, generator=gen)
        correlated = innovations.clone()
        for i in range(1, correlated.shape[1]):
            correlated[:, i] = .8*correlated[:, i-1]+.6*innovations[:, i]
        d = scalar_diagnostics(correlated[:, 2000:])
        expected = 4*10000*(1-.8)/(1+.8)
        self.assertAlmostEqual(d["bulk_ess"], expected, delta=.20*expected)


if __name__ == "__main__":
    unittest.main()
