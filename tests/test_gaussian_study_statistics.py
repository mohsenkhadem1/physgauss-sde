import unittest
import numpy as np
from experiments.gaussian_study_statistics import paired_layout_bootstrap


class StatisticsTests(unittest.TestCase):
    def test_matched_replicates_are_one_independent_layout(self):
        first=paired_layout_bootstrap([1.,2.],[2.,4.],draws=1000)
        repeated=paired_layout_bootstrap([1.,1.,2.,2.],[2.,2.,4.,4.],groups=['a','a','b','b'],draws=1000)
        self.assertEqual(repeated['independent_layouts'],2)
        self.assertEqual(first['paired_difference95_interval'],repeated['paired_difference95_interval'])
        np.testing.assert_allclose(repeated['relative_improvement95_interval'],[.5,.5])

    def test_paired_equal_results_have_zero_difference(self):
        result=paired_layout_bootstrap([1.,7.,3.],[1.,7.,3.],draws=100)
        self.assertEqual(result['paired_difference95_interval'],[0.,0.])


if __name__=='__main__':unittest.main()
