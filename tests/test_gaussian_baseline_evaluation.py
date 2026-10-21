import tempfile
from pathlib import Path
import unittest

import numpy as np

from experiments.evaluate_gaussian_baseline_predictions import read_prediction


class BaselineEvaluationTests(unittest.TestCase):
    def test_dense_prediction_schemas_and_clock_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'prediction.npz';q=np.zeros((3,4,3));times=np.array([0.,1.5,3.])
            for key in ['positions_m','mean_positions_m']:
                np.savez(path,**{key:q,'times_s':times,'frame_ids':np.arange(3)})
                np.testing.assert_array_equal(read_prediction(path,4,times),q)
            with self.assertRaises(AssertionError):read_prediction(path,4,times+.1)
            with self.assertRaises(ValueError):read_prediction(path,5,times)

    def test_nonfinite_prediction_is_rejected_before_target_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'prediction.npz';q=np.zeros((3,4,3));q[1,2,0]=np.nan
            np.savez(path,positions_m=q)
            with self.assertRaises(ValueError):read_prediction(path,4,np.array([0.,1.5,3.]))


if __name__=='__main__':unittest.main()
