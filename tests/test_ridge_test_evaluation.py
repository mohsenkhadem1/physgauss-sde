import unittest
import numpy as np
from experiments.evaluate_ridge_gaussian_inverse import align_prefix_features

class PrefixEvaluationTests(unittest.TestCase):
    def test_only_past_pixels_needed_to_reconstruct_masked_training_grid(self):
        times=np.array([0.,.5,1.,2.,3.]);features=np.arange(6).reshape(3,2)
        padded=align_prefix_features((5,2),times,1.,features,times[:3])
        np.testing.assert_array_equal(padded[:3],features);np.testing.assert_array_equal(padded[3:],0)

    def test_future_or_reordered_observation_is_rejected(self):
        times=np.array([0.,.5,1.,2.,3.]);features=np.ones((3,2))
        with self.assertRaises(ValueError):align_prefix_features((5,2),times,1.,features,np.array([0.,.5,2.]))
        with self.assertRaises(AssertionError):align_prefix_features((5,2),times,1.,features,np.array([0.,1.,.5]))

if __name__=='__main__':unittest.main()
