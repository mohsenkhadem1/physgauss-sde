import unittest
import torch
from physgauss.likelihood_audit import importance_diagnostics, likelihood_log_ratio


class LikelihoodAuditTests(unittest.TestCase):
    def test_identical_likelihood_has_unit_relative_ess(self):
        predicted = torch.tensor([[.1, .2], [.2, .3], [.3, .5]], dtype=torch.float64)
        ratios = likelihood_log_ratio(predicted, predicted, torch.zeros(2, dtype=predicted.dtype), .1)
        torch.testing.assert_close(ratios, torch.zeros_like(ratios))
        diagnostics = importance_diagnostics(ratios)
        self.assertAlmostEqual(diagnostics["relative_ess"], 1.)
        self.assertFalse(diagnostics["low_relative_ess_warning"])

    def test_known_gaussian_likelihood_ratio(self):
        actual = torch.tensor([[1.5]], dtype=torch.float64)
        linear = torch.tensor([[0.]], dtype=torch.float64)
        observed = torch.tensor([1.], dtype=torch.float64)
        self.assertAlmostEqual(float(likelihood_log_ratio(actual, linear, observed, .5)[0]), 1.5)

    def test_collapsed_weights_are_flagged_and_log_shift_is_invariant(self):
        ratios = torch.full((32,), -1000., dtype=torch.float64)
        ratios[0] = 0
        a, b = importance_diagnostics(ratios), importance_diagnostics(ratios + 10000.)
        self.assertAlmostEqual(a["importance_ess"], 1.)
        self.assertAlmostEqual(a["maximum_normalized_weight"], 1.)
        self.assertTrue(a["low_relative_ess_warning"])
        self.assertEqual(a["relative_ess"], b["relative_ess"])

    def test_zero_target_density_is_counted_and_all_zero_fails(self):
        values = torch.tensor([0., -torch.inf], dtype=torch.float64)
        diagnostic = importance_diagnostics(values)
        self.assertEqual(diagnostic["zero_target_weight_count"], 1)
        with self.assertRaises(RuntimeError):
            importance_diagnostics(torch.full((3,), -torch.inf))


if __name__ == "__main__":
    unittest.main()
