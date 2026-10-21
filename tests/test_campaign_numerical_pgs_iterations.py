import unittest
import numpy as np
from experiments.campaign_numerical_pgs_iterations import pair_metrics

class IterationPairMetricsTests(unittest.TestCase):
    def test_translation_contraction_does_not_hide_iteration_change(self):
        rest={'original':np.zeros((2,3)),'recentered':np.zeros((2,3))-np.array([0,0,.25])}
        low={'original':rest['original']+1e-3,'recentered':rest['recentered']+1.1e-3}
        high={'original':rest['original']+2e-3,'recentered':rest['recentered']+2e-3}
        r=pair_metrics(high,low,rest)
        self.assertTrue(r['translation_error_decreased'])
        self.assertTrue(r['translation_64_gate_passed'])
        self.assertFalse(r['both_iteration_differences_below_tolerance'])
        self.assertEqual(len(r['iteration_pairs']),2)

    def test_worsened_translation_is_reported_even_when_displacement_signal_large(self):
        rest={'original':np.zeros((2,3)),'recentered':np.zeros((2,3))-np.array([0,0,.25])}
        low={'original':rest['original']+1e-3,'recentered':rest['recentered']+1.01e-3}
        high={'original':rest['original']+1e-3,'recentered':rest['recentered']+1.5e-3}
        r=pair_metrics(high,low,rest)
        self.assertFalse(r['translation_error_decreased']);self.assertFalse(r['translation_64_gate_passed'])
