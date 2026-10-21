import sys
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from isaac_calibrated_probes_v02 import calibrated_menu, command_at, depth_state_agreement, capture_current_state


class FreshProbeTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[-2., -2, -2], [2., -2, -2], [2., 2, -2], [-2., 2, -2]])
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])
        self.k = np.array([[622., 0, 320], [0, 622., 240], [0, 0, 1.]])
        self.pose = np.eye(4)

    def test_menu_covers_three_axes_and_distinct_temporal_families(self):
        menu = calibrated_menu(np.zeros((2, 3)), self.k, self.pose)
        self.assertEqual(len(menu['sensor']['times_s']), 31)
        vectors = np.array([a['peak_force_N'] for a in menu['candidates']])
        np.testing.assert_allclose(vectors[:3], .02*np.eye(3))
        self.assertEqual(menu['candidates'][3]['force_type_index'], 2)
        self.assertEqual(menu['candidates'][3]['force_type'], 'shear_oscillation')
        commands = np.array([command_at(menu['candidates'][3], t) for t in np.linspace(0, 3, 121)])
        self.assertGreater(commands[:, 0, 2].max(), .01)
        self.assertLess(commands[:, 0, 2].min(), -.01)
        np.testing.assert_array_equal(commands[:, 1], 0.)

    def test_ray_guard_rejects_stale_depth_with_correct_shapes(self):
        depth = np.full((480, 640), 2., np.float32); mask = np.ones_like(depth, bool)
        audit = depth_state_agreement(self.points, self.faces, self.pose, self.k, depth, mask)
        self.assertTrue(audit['passed'])
        moved = self.points.copy(); moved[:, 2] -= .001
        stale = depth_state_agreement(moved, self.faces, self.pose, self.k, depth, mask)
        self.assertFalse(stale['passed']); self.assertGreater(stale['maximum_error_m'], .0009)

    def test_synchronous_capture_retries_until_current_state_without_physics_time(self):
        moved = self.points.copy(); moved[:, 2] -= .001
        settings = types.SimpleNamespace(value=True)
        settings.get_as_bool = lambda key: settings.value
        settings.set_bool = lambda key, value: setattr(settings, 'value', value)
        carb = types.SimpleNamespace(settings=types.SimpleNamespace(get_settings=lambda: settings))
        calls = []
        def step(**kwargs):
            self.assertFalse(settings.value)
            self.assertEqual(kwargs['delta_time'], 0.)
            self.assertTrue(kwargs['wait_for_render'])
            calls.append(kwargs)
        rep = types.SimpleNamespace(orchestrator=types.SimpleNamespace(step=step))
        world = types.SimpleNamespace(current_time=1.5)
        rgb = types.SimpleNamespace(get_data=lambda **kwargs: np.zeros((480, 640, 4), np.uint8))
        depth = types.SimpleNamespace(get_data=lambda **kwargs: np.full((480, 640), 2.001 if len(calls) >= 2 else 2., np.float32))
        with patch.dict(sys.modules, {'carb': carb}):
            sensor, audit = capture_current_state(world, rep, rgb, depth, moved, self.faces, self.pose, self.k, attempts=3)
        self.assertEqual(audit['attempts'], 2); self.assertEqual(len(calls), 2)
        self.assertTrue(settings.value); self.assertEqual(world.current_time, 1.5)
        self.assertEqual(audit['physics_time_before_s'], audit['physics_time_after_s'])
        self.assertEqual(sensor[0].shape, (480, 640, 3))

    def test_real_rejected_v01_frame_is_identified_as_previous_geometry(self):
        root = Path(__file__).resolve().parents[1]/'runs/isaac-calibrated-probes-validation-v01'
        if not (root/'private/e0008/evaluation/trajectory.npz').exists():
            self.skipTest('Optional actual stale-render regression requires preserved v01 e0008')
        from PIL import Image
        q = np.load(root/'private/e0008/evaluation/trajectory.npz')['positions_m']
        faces = np.load(root/'private/e0008/reference.npz')['surface_faces']
        directory = root/'outcomes/e0008/evaluation/observations'
        cameras = np.load(directory/'cameras.npz')
        depth = np.load(directory/'depth/00003.npz')['depth_m']
        mask = np.asarray(Image.open(directory/'masks/00003.png')) > 0
        current = depth_state_agreement(q[3], faces, cameras['camera_to_world_usd_row'][3], cameras['K'], depth, mask)
        previous = depth_state_agreement(q[2], faces, cameras['camera_to_world_usd_row'][3], cameras['K'], depth, mask)
        self.assertFalse(current['passed']); self.assertTrue(previous['passed'])


if __name__ == '__main__': unittest.main()
