"""Numerical checks for the privileged diagnosis, independent of model outcomes."""
import copy
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from diagnose_gaussian_forward import control_physical, motion_metrics
from diagnose_gaussian_reference import covariance, credible_set, matched_hypotheses, normalize_log_mass
from censored_range_observation import CensoredRanges, CensoredRangeModel


class DiagnosisTests(unittest.TestCase):
    def test_control_replaces_all_forces_and_keeps_supports(self):
        original = np.arange(25, dtype=float)
        action = {'peak_force_N': [.01, .01, .005]}
        handles = np.arange(6, dtype=float).reshape(2, 3)/10
        physical = control_physical(original, action, handles)
        np.testing.assert_array_equal(physical[:10], original[:10])
        np.testing.assert_array_equal(physical[22:], original[22:])
        np.testing.assert_array_equal(physical[10:16], handles.ravel())
        np.testing.assert_array_equal(physical[16:19], action['peak_force_N'])
        np.testing.assert_array_equal(physical[19:22], 0)
        np.testing.assert_array_equal(original, np.arange(25))

    def test_correlated_covariance_does_not_count_common_error_as_independent(self):
        sigma = .0005
        matrix = covariance(2, 3, sigma)
        self.assertTrue(np.linalg.eigvalsh(matrix).min() > 0)
        self.assertAlmostEqual(matrix[0, 0]/sigma**2, 3)
        self.assertAlmostEqual(matrix[0, 1]/sigma**2, 2)
        self.assertAlmostEqual(matrix[0, 3]/sigma**2, 1)

    def test_log_mass_stability_and_credible_set(self):
        np.testing.assert_allclose(normalize_log_mass([-10000, -10001]), [1/(1+np.exp(-1)), 1/(1+np.exp(1))])
        np.testing.assert_array_equal(credible_set([.8, .15, .05]), [0, 1])
        np.testing.assert_array_equal(credible_set([.99, .005, .005]), [0])
        with self.assertRaises(ValueError):
            normalize_log_mass([-np.inf, -np.inf])

    def test_enumerated_posterior_matches_gaussian_bayes_rule(self):
        model = CensoredRangeModel([[.01]], .02, outlier_probability=0)
        means = np.array([[0.], [.1], [.2]])
        template = CensoredRanges(np.ones((3, 1), dtype=bool), means)
        observed = CensoredRanges(np.array([True]), np.array([.13]))
        weights, _ = model.condition(template, observed)
        expected = normalize_log_mass(-.5*(means[:, 0]-.13)**2/.01)
        np.testing.assert_allclose(weights, expected, atol=1e-13)

    def test_bank_rejects_changed_forces_or_cameras(self):
        case = {'group_id': 'g03', 'boundary_type': 'pin', 'force_type': 'push', 'seed': 1,
                'cells': [1], 'material': {'youngs_modulus_Pa': 1500, 'density': 700}}
        other = copy.deepcopy(case)
        other['material']['youngs_modulus_Pa'] = 5000
        ref = {key: np.zeros((1, 3)) for key in ['rest_points_m', 'tetrahedra', 'surface_faces']}
        camera = {key: np.zeros(1) for key in ['K', 'camera_to_world_usd_row', 'times_s', 'frame_ids']}
        matched_hypotheses([case, other], [{'force': .1}]*2, [ref]*2, [camera]*2)
        with self.assertRaises(ValueError):
            matched_hypotheses([case, other], [{'force': .1}, {'force': .2}], [ref]*2, [camera]*2)
        changed = copy.deepcopy(camera)
        changed['K'][0] = 1
        with self.assertRaises(AssertionError):
            matched_hypotheses([case, other], [{'force': .1}]*2, [ref]*2, [camera, changed])

    def test_oracle_is_not_assumed_better_than_static(self):
        rest = np.zeros((1, 3))
        truth = np.ones((3, 1, 3))*.001
        pred = -truth
        result = motion_metrics(pred, truth, rest, np.array([0., 1.5, 3.]))
        self.assertAlmostEqual(result['all_coordinate_rmse_m'], .002)
        self.assertAlmostEqual(result['all_static_rmse_m'], .001)
        self.assertAlmostEqual(result['all_displacement_cosine'], -1.)


if __name__ == '__main__':
    unittest.main()
