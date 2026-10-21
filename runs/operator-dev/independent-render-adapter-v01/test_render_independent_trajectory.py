"""CPU calibration/guard controls; no Isaac or response data imported."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[3] / 'experiments/render_independent_trajectory.py'
SPEC = importlib.util.spec_from_file_location('independent_renderer', SOURCE)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def cube_capture():
    points = np.array([[x, y, z] for z in [0., .5] for y in [-.5, .5] for x in [-.5, .5]])
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7],
                      [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    pose = np.eye(4); pose[3, :3] = [0., 0., 2.]
    k = np.array([[40., 0., 32.], [0., 40., 32.], [0., 0., 1.]])
    y, x = np.mgrid[:64, :64]
    # Analytic front z=.5 plane with its finite square silhouette, not ray helper output.
    mask = (abs((x+.5-32)/40*1.5)<.5)&(abs((y+.5-32)/40*1.5)<.5)
    depth = np.where(mask, 1.5, np.inf).astype(np.float32)
    rgb = np.broadcast_to(mask[..., None]*np.array([70, 110, 170], dtype=np.uint8), (64, 64, 3)).copy()
    return points, faces, pose, k, mask, depth, rgb


class RenderAdapterTests(unittest.TestCase):
    def test_gpu_policy_rejects_wrong_renderer_and_large_auxiliary_allocations(self):
        valid = [{'index': 2, 'type': 'C+G', 'used_memory_mib': 1500.},
                 {'index': 0, 'type': 'C', 'used_memory_mib': 8.}]
        renderer.validate_gpu_contexts(valid)
        for devices in [[], [{'index': 0, 'type': 'C+G', 'used_memory_mib': 8.}],
                        valid+[{'index': 3, 'type': 'G', 'used_memory_mib': 8.}],
                        valid+[{'index': 3, 'type': 'C', 'used_memory_mib': 161.}]]:
            with self.subTest(), self.assertRaises(RuntimeError):
                renderer.validate_gpu_contexts(devices)

    def test_completed_input_and_tamper_guards(self):
        for fault in [None, 'unfinished', 'source', 'trajectory', 'nontrain']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); frozen = root/'provenance'; frozen.mkdir()
                points, faces, *_ = cube_capture()
                arrays = dict(rest_points_m=points, surface_faces=faces.astype(np.int32),
                              tetrahedra=np.array([[0, 1, 2, 4]], dtype=np.int32))
                np.savez_compressed(frozen/'static-reference.npz', **arrays)
                fixture = {'cases': [{'episode_id': 't0000', 'recipe': {'split': 'train'},
                    'reference_static_arrays': [{'key': key, 'dtype': str(value.dtype), 'shape': list(value.shape),
                         'sha256': renderer.array_sha256(value)} for key, value in arrays.items()]}]}
                fixture_path = frozen/'gaussian_independent_diagnostic_inputs_v01.json'
                renderer.write_json(fixture_path, fixture)
                (frozen/'frozen.py').write_text('# synthetic own provenance fixture\n')
                times = np.arange(31)/10
                np.savez_compressed(root/'trajectory.npz', times_s=times, actual_state_times_s=times,
                                    positions_m=np.broadcast_to(points, (31, len(points), 3)))
                trajectory_sha = renderer.sha256(root/'trajectory.npz')
                renderer.write_json(root/'protocol.json', {'episode_id': 't0000', 'response': 'diagonal_pulse_handle0',
                    'native_equivalence': False, 'predictive_reserved_responses_opened': False, 'output_hz': 10,
                    'saved_regenerated_static_reference_sha256': renderer.sha256(frozen/'static-reference.npz'),
                    'source_input_dependency_and_gate_sha256': {str(f): renderer.sha256(f) for f in [frozen/'frozen.py', fixture_path]}})
                renderer.write_json(root/'summary.json', {'case': 't0000', 'response': 'diagonal_pulse_handle0',
                    'completed': True, 'trajectory_sha256': trajectory_sha})
                renderer.write_json(root/'execution.json', {'terminal': True, 'completed': True})
                if fault == 'unfinished':
                    renderer.write_json(root/'execution.json', {'terminal': False, 'completed': False})
                elif fault == 'source':
                    (frozen/'frozen.py').write_text('# altered provenance\n')
                elif fault == 'trajectory':
                    (root/'trajectory.npz').write_bytes(b'corrupted')
                elif fault == 'nontrain':
                    protocol = json.loads((root/'protocol.json').read_text()); protocol['episode_id'] = 'reserved'; renderer.write_json(root/'protocol.json', protocol)
                if fault:
                    with self.assertRaises(ValueError):
                        renderer.load_completed_input(root, trajectory_sha)
                else:
                    result = renderer.load_completed_input(root, trajectory_sha)
                    self.assertEqual(result['episode'], 't0000')
                    np.testing.assert_array_equal(result['times_s'], times)

    def test_saved_frame_selection_has_no_resampling(self):
        np.testing.assert_array_equal(renderer.frame_indices([0, 15, 30], 31), [0, 15, 30])
        for bad in [[0, 0], [15, 0], [-1, 0], [0, 31], []]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                renderer.frame_indices(bad, 31)

    def test_camera_center_projection_and_rest_translation(self):
        points = cube_capture()[0]
        plan = renderer.rest_camera_plan(points)
        k, aperture = renderer.camera_intrinsics(320, 240)
        self.assertEqual(aperture, 27.)
        for t in [0., 1.5, 3.]:
            for offset in [(0., 0.), (60., 10.)]:
                pose = renderer.camera_pose(plan, t, offset)
                np.testing.assert_allclose(pose[:3, :3]@pose[:3, :3].T, np.eye(3), atol=1e-14)
                self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.)
                pixels, depth = renderer.project_points(np.array([plan['center_m']]), pose, k)
                np.testing.assert_allclose(pixels, [[160., 120.]], atol=1e-12)
                self.assertAlmostEqual(depth[0], plan['radius_m'])
        delta = np.array([3., -4., 2.])
        shifted = renderer.rest_camera_plan(points+delta)
        np.testing.assert_allclose(renderer.camera_pose(shifted, 1.5)[:3, :3], renderer.camera_pose(plan, 1.5)[:3, :3])
        np.testing.assert_allclose(renderer.camera_pose(shifted, 1.5)[3, :3], renderer.camera_pose(plan, 1.5)[3, :3]+delta)

    def test_ray_depth_against_analytic_plane(self):
        points, faces, pose, k, *_ = cube_capture()
        depth, cosine, _ = renderer.ray_surface(points, faces, pose, k, np.array([[32., 32.], [39., 35.], [3., 3.]]))
        np.testing.assert_allclose(depth[:2], 1.5)
        self.assertTrue(np.isinf(depth[2]))
        self.assertGreater(cosine[:2].min(), .9)

    def test_sensor_guard_accepts_analytic_capture(self):
        points, faces, pose, k, mask, depth, rgb = cube_capture()
        result = renderer.validate_capture(rgb, depth, mask, points, faces, pose, k)
        self.assertEqual(result['maximum_ray_depth_error_m'], 0.)
        self.assertEqual(result['independent_coarse_ray_mask_iou'], 1.)
        self.assertLess(result['maximum_unproject_reproject_error_px'], 1e-10)

    def test_sensor_guard_rejects_stale_depth_pose_and_intrinsics(self):
        points, faces, pose, k, mask, depth, rgb = cube_capture()
        wrong_depth = depth.copy(); wrong_depth[mask] += .001
        wrong_pose = pose.copy(); wrong_pose[3, 2] += .02
        wrong_k = k.copy(); wrong_k[0, 0] *= 2
        for actual_depth, actual_pose, actual_k in [(wrong_depth, pose, k), (depth, wrong_pose, k), (depth, pose, wrong_k)]:
            with self.subTest(), self.assertRaises(ValueError):
                renderer.validate_capture(rgb, actual_depth, mask, points, faces, actual_pose, actual_k)

    def test_semantic_mask_uses_object_label(self):
        ids = np.array([[0, 2], [3, 2]], dtype=np.uint32)
        mask, _ = renderer.semantic_object_mask({'data': ids, 'info': {'idToLabels': {'0': {'class': 'BACKGROUND'},
             '2': {'class': 'independent_object'}, '3': {'class': 'different_object'}}}}, ids.shape)
        np.testing.assert_array_equal(mask, [[False, True], [False, True]])
        with self.assertRaises(ValueError):
            renderer.semantic_object_mask({'data': ids, 'info': {'idToLabels': {}}}, ids.shape)


if __name__ == '__main__':
    unittest.main()
