from dataclasses import replace
import unittest
import torch
from physgauss.benchmark import Config, SceneModel, make_benchmark
from physgauss.metrics import uncertainty_ranking
from physgauss.study import aggregate_cases, quick_plan, validate_plan


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_oracle_uncertainty_has_zero_sparsification_error(self):
        mean = torch.tensor([.1, .2, .3], dtype=torch.float64)
        sd = mean / 10
        samples = torch.stack((mean - sd, mean + sd))
        target = torch.zeros_like(mean)
        correct = uncertainty_ranking(samples, target)
        incorrect = uncertainty_ranking(torch.stack((mean - sd.flip(0), mean + sd.flip(0))), target)
        self.assertAlmostEqual(correct["normalized_ause"], 0.)
        self.assertAlmostEqual(correct["spearman_error_uncertainty"], 1.)
        self.assertGreater(incorrect["normalized_ause"], 0.)
        self.assertAlmostEqual(incorrect["spearman_error_uncertainty"], -1.)

    def test_study_rejects_unequal_observation_budgets(self):
        plan = quick_plan()
        validate_plan(plan)
        plan["schedules"]["extra_images"] = [0, 1, 2, 6, 8]
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_hidden_external_force_changes_truth_but_not_inference_predictions(self):
        config = Config.quick()
        pulse = replace(config, truth_external_pulse=True, truth_stochastic=False)
        quiet = replace(config, truth_external_pulse=False, truth_stochastic=False)
        a, b = SceneModel(pulse), SceneModel(quiet)
        z = torch.zeros(1, a.dimension, dtype=torch.float64)
        with torch.no_grad():
            torch.testing.assert_close(a.predict(z), b.predict(z), rtol=0, atol=0)
        truth_a, truth_b = make_benchmark(pulse), make_benchmark(quiet)
        self.assertGreater(float((truth_a.truth.x - truth_b.truth.x).abs().max()), 1e-5)

    def test_aggregation_counts_independent_seeds_and_rejects_duplicates(self):
        def case(seed, value):
            scores = {key: value for key in ["position_rmse_m", "position_crps_m",
                "trajectory_energy_score_m", "component_90_coverage", "component_90_mean_width_m",
                "rgb_psnr_db", "novel_view_psnr_db"]}
            return {"seed": seed, "condition": "stochastic", "schedule": "long_gap",
                "scale_posterior": {"probability_zero_process_noise": .3, "posterior_mean_scale": .8},
                "methods": {"adaptive_sde": {"scores": {"withheld": scores},
                    "ranking_withheld": {"normalized_ause": .1},
                    "observation_linearization_rmse_in_noise_sigma": .2}}}
        groups = aggregate_cases([case(10, 1.), case(20, 3.)])
        self.assertEqual(groups[0]["independent_realizations"], 2)
        self.assertEqual(groups[0]["metrics"]["position_rmse_m"]["mean"], 2.)
        self.assertAlmostEqual(groups[0]["metrics"]["position_rmse_m"]["seed_std"], 2 ** .5)
        with self.assertRaises(ValueError):
            aggregate_cases([case(10, 1.), case(10, 3.)])


if __name__ == "__main__":
    unittest.main()
