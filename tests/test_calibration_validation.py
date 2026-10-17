import math
import unittest
import torch
from physgauss.calibrated_diffusion import CalibratedDiffusion
from physgauss.calibrated_study import default_plan, check_splits, case_condition
from physgauss.uncertainty_validation import (conformal_quantile, trajectory_score,
    apply_conformal, finite_difference_precision, wilson_interval)


class CalibrationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_physical_feature_sign_symmetry_and_differentiability(self):
        net = CalibratedDiffusion()
        features = torch.tensor([[.02, -.3], [0., 0.], [-.1, .4]], dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(net(features), net(-features), rtol=0, atol=0)
        gradient, = torch.autograd.grad(net(features).sum(), features)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertEqual(sum(p.numel() for p in net.parameters()), 33)
        self.assertTrue(bool((net(features) > .0001).all()))

    def test_conformal_order_statistic_and_insufficient_calibration(self):
        self.assertEqual(conformal_quantile(list(range(1, 20)), .1), 18.)
        self.assertTrue(math.isinf(conformal_quantile([1., 2.], .1)))
        with self.assertRaises(ValueError): conformal_quantile([math.nan])
        # 84 correlated coordinates contribute ONE score per trajectory.
        self.assertEqual(trajectory_score(torch.ones(7, 4, 3), torch.full((7, 4, 3), .5)), 2.)

    def test_independent_split_and_preset_condition_assignment(self):
        plan = default_plan(); check_splits(plan)
        plan["interval_test_seeds"] = plan["interval_calibration_seeds"][:]
        with self.assertRaises(ValueError): check_splits(plan)
        self.assertEqual(case_condition(82001), case_condition(82001))

    def test_frozen_quantile_does_not_depend_on_test_truth(self):
        records = [{"mode": "neural", "seed": i, "gap_max_standardized_error": float(i),
                    "gap_mean_sd_m": .001, "raw_component_90_coverage": .5, "gap_rmse_m": .002}
                   for i in range(1, 22)]
        first = apply_conformal(records, list(range(1, 20)), [20, 21])[0]
        records[-1]["gap_max_standardized_error"] = 1e9
        second = apply_conformal(records, list(range(1, 20)), [20, 21])[0]
        self.assertEqual(first["sd_multiplier"], second["sd_multiplier"])
        self.assertEqual(first["sd_multiplier"], 18.)
        with self.assertRaises(ValueError): apply_conformal(records, [1, 2], [2, 3])

    def test_conformal_repeated_case_coverage_on_known_exchangeable_scores(self):
        generator = torch.Generator().manual_seed(810)
        scores = torch.rand(3000, 20, generator=generator, dtype=torch.float64)
        # n=19, rank18: expected repeated-split coverage 18/20=.9.
        covered = scores[:, -1] <= scores[:, :19].sort(1).values[:, 17]
        self.assertLess(abs(float(covered.double().mean())-.9), .025)
        lo, hi = wilson_interval(18, 20)
        self.assertLess(lo, .9); self.assertGreater(hi, .9)

    def test_observed_hessian_matches_known_quadratic_and_rejects_indefinite(self):
        precision = torch.tensor([[3., .4], [.4, 2.]], dtype=torch.float64)
        class Quadratic:
            def objective(self, state, observations): return .5*torch.einsum("bi,ij,bj->b", state, precision, state)
        fit = {"mean": torch.tensor([.1, -.3], dtype=torch.float64),
               "precision_cholesky": torch.tensor([[2., 0.], [.3, 1.]], dtype=torch.float64)}
        result, audit, white = finite_difference_precision(Quadratic(), None, fit, epsilon=.001)
        chol = result["precision_cholesky"]
        torch.testing.assert_close(chol@chol.T, precision, rtol=1e-11, atol=1e-11)
        self.assertLess(audit["relative_antisymmetry"], 1e-10)
        precision[0, 0] = -3.
        with self.assertRaises(RuntimeError): finite_difference_precision(Quadratic(), None, fit)

    def test_exact_target_correction_and_whitened_gradient(self):
        import importlib.util
        from pathlib import Path
        import sys
        path = Path(__file__).resolve().parents[1]/"scripts/audit_calibrated_posterior.py"
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("posterior_audit", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        precision = torch.tensor([[2., .2], [.2, 3.]], dtype=torch.float64)
        class Quadratic:
            def objective(self, state, observations): return .5*torch.einsum("bi,ij,bj->b", state, precision, state)
        fit = {"mean": torch.zeros(2, dtype=torch.float64), "precision_cholesky": torch.linalg.cholesky(precision)}
        reference = module.SingleReference(fit)
        white = torch.tensor([[.2, -.5], [1., .3]], dtype=torch.float64)
        points = reference.decode(white, None)
        values, gradient = module.ExactCorrection(Quadratic(), None, reference).value_and_grad(points, None)
        torch.testing.assert_close(values[0], torch.zeros(2, dtype=torch.float64), rtol=0, atol=1e-14)
        torch.testing.assert_close(reference.whiten_gradient(gradient, None), torch.zeros_like(white), rtol=0, atol=1e-14)
        changed = dict(fit); changed["precision_cholesky"] = torch.eye(2, dtype=torch.float64)
        ref2 = module.SingleReference(changed)
        values, grad = module.ExactCorrection(Quadratic(), None, ref2).value_and_grad(white, None)
        expected = -.5*torch.einsum("bi,ij,bj->b", white, precision-torch.eye(2), white)
        torch.testing.assert_close(values[0], expected)
        torch.testing.assert_close(grad, -white@(precision-torch.eye(2)))
        direct = module.ExactCorrection(Quadratic(), None, ref2)(white, None)
        torch.testing.assert_close(direct[0], values[0])
        torch.testing.assert_close(direct[1], values[1])
        self.assertTrue(torch.equal(direct[2], values[2]))
        sys.path.pop(0)

    def test_local_schur_volume_observed_hessian_matches_independent_dense_AD(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
        from sparse_observed_curvature import observed_precision, directional_refinement
        from test_neural_marginal import NeuralMarginalTests
        from physgauss.benchmark import Observations
        m = NeuralMarginalTests().small(); m.network = CalibratedDiffusion()
        for parameter in m.network.parameters(): parameter.requires_grad_(False)
        point = NeuralMarginalTests().state(m)[0][0].detach()
        rgb = m.predict(point[None], m.view_frames)[0].detach()
        noise = .00001*torch.randn(rgb.shape, dtype=rgb.dtype, generator=torch.Generator().manual_seed(990))
        obs = Observations(m.view_frames, rgb+noise, .015)
        fitted = {"mean": point, "precision_cholesky": torch.eye(len(point), dtype=point.dtype)}
        actual, details, white = observed_precision(m, obs, fitted)
        dense = torch.autograd.functional.hessian(lambda z: m.objective(z[None], obs).sum(), point, vectorize=False)
        chol = actual["precision_cholesky"]
        torch.testing.assert_close(chol@chol.T, dense, rtol=1e-9, atol=1e-8)
        self.assertLess(details["relative_antisymmetry"], 1e-10)
        checks = directional_refinement(m, obs, fitted, white)
        self.assertLess(max(checks[-1]["relative_Hv_errors"]), 1e-4)
        sys.path.pop(0)

    def test_fixed_cube_projection_does_not_depend_on_tied_eigenspace_solver(self):
        import json
        from pathlib import Path
        from unittest.mock import patch
        from physgauss.physics import cube_mesh
        from physgauss.stochastic import spatial_basis
        mesh = cube_mesh()
        expected = json.loads((Path(__file__).resolve().parents[1]/"examples/neural-marginal/model_constants.json").read_text())["internal_spatial_basis"]
        with patch("torch.linalg.eigh", side_effect=RuntimeError("Backend-dependent eigenspace")):
            actual = spatial_basis(mesh, 3, .45)
            torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype), rtol=0, atol=0)
            with self.assertRaises(RuntimeError): spatial_basis(mesh, 2, .45)
        torch.testing.assert_close(actual.square().sum(-1), torch.ones(19, dtype=actual.dtype), rtol=1e-14, atol=1e-14)

    def test_volume_emission_has_correct_empty_slab_first_and_second_derivatives(self):
        from physgauss.continuous_render import ContinuousGaussianRenderer
        colors = torch.tensor([[.2, .4, .7]], dtype=torch.float64)
        fractions = torch.ones(1, 1, 1, 1, dtype=torch.float64)
        for value in (0., 1e-35, 1e-12, 5e-4, .1):
            optical = torch.full((1, 1, 1, 1, 1), value, dtype=torch.float64, requires_grad=True)
            image = ContinuousGaussianRenderer.integrate(optical, fractions, colors)
            first, = torch.autograd.grad(image.sum(), optical, create_graph=True)
            second, = torch.autograd.grad(first.sum(), optical)
            depth = optical.detach().sum(2)
            expected_value = (-torch.expm1(-depth))[..., None]*colors+.035*torch.exp(-depth)[..., None]
            expected_first = torch.exp(-optical.detach())*(colors.sum()-.105)
            torch.testing.assert_close(image, expected_value, rtol=1e-12, atol=1e-15)
            torch.testing.assert_close(first, expected_first, rtol=1e-12, atol=1e-15)
            torch.testing.assert_close(second, -expected_first, rtol=1e-10, atol=1e-14)


if __name__ == "__main__": unittest.main()
