import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
from evaluate_gaussian_parameters import scalar_summary,categorical_summary


class ParameterMetricTests(unittest.TestCase):
    def test_scalar_marginal_coverage_and_units(self):
        result=scalar_summary(np.arange(101)*10.,500.)
        self.assertTrue(result['covered90']);self.assertEqual(result['median'],500.)
        self.assertEqual(result['width90'],900.);self.assertEqual(result['absolute_error'],0.)
        self.assertFalse(scalar_summary([1.,2.,3.],4.)['covered90'])

    def test_categorical_probability_scores(self):
        result=categorical_summary([.1,.7,.1,.1],1)
        self.assertTrue(result['correct']);self.assertAlmostEqual(result['brier'],.12)
        self.assertAlmostEqual(result['negative_log_probability'],-np.log(.7))


if __name__=='__main__':unittest.main()
