import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from isaac_batch_dataset_worker import grid_offsets, match_rows, translated_camera


class IsaacBatchTests(unittest.TestCase):
    def test_isolated_grid_and_camera_translation(self):
        offsets = grid_offsets(8)
        distances = np.linalg.norm(offsets[:, None]-offsets[None], axis=-1)
        self.assertGreaterEqual(distances[distances > 0].min(), 4.)
        pose = np.eye(4); pose[3, :3] = [.2, -.3, .8]
        translated = translated_camera(pose, offsets[-1])
        np.testing.assert_array_equal(translated[:3], pose[:3])
        np.testing.assert_allclose(translated[3, :3]-offsets[-1], pose[3, :3])
        np.testing.assert_array_equal(pose[3, :3], [.2, -.3, .8])
        with self.assertRaises(ValueError): grid_offsets(9)
        with self.assertRaises(ValueError): grid_offsets(2, 3.)

    def test_tensor_order_is_recovered_and_mismatches_rejected(self):
        points = np.random.default_rng(2).normal(0, .05, (455, 3))
        expected = points[None]+grid_offsets(8)[:, None]
        order = np.array([5, 2, 1, 0, 7, 3, 6, 4])
        actual_to_case, case_to_actual = match_rows(expected[order].astype(np.float32), expected)
        np.testing.assert_array_equal(actual_to_case, order)
        np.testing.assert_array_equal(actual_to_case[case_to_actual], np.arange(8))
        broken = expected[order].copy(); broken[2, 9, 1] += .01
        with self.assertRaises(ValueError): match_rows(broken, expected)
        duplicates = expected[order].copy(); duplicates[1] = duplicates[0]
        with self.assertRaises(ValueError): match_rows(duplicates, expected)
        rigid = expected[:, 0]
        a, b = match_rows(rigid[order], rigid)
        np.testing.assert_array_equal(a[b], np.arange(8))


if __name__ == '__main__': unittest.main()
