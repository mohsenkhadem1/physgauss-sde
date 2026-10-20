import sys
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from isaac_calibrated_probes import calibrated_menu, command_at, record_selection, load_selected_measurement


class CalibratedProbeTests(unittest.TestCase):
    def test_public_controls_preserve_fixed_hardware_and_distinct_eval(self):
        centers = np.array([[.1, 0, .25], [-.1, .02, .25]])
        menu = calibrated_menu(centers, np.eye(3), np.eye(4))
        self.assertEqual(len(menu['candidates']), 4)
        self.assertFalse(any('material' in key or 'support' in key for key in menu))
        np.testing.assert_array_equal(menu['known_handle_centers_m'], centers)
        vectors = np.array([a['peak_force_N'] for a in menu['candidates']])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), .02)
        np.testing.assert_array_equal(vectors[0], -vectors[1])
        np.testing.assert_array_equal(vectors[2], -vectors[3])
        self.assertGreater(menu['evaluation_query']['peak_force_N'][2], 0.)
        for action in menu['candidates']+[menu['evaluation_query']]:
            np.testing.assert_array_equal(command_at(action, 0.), np.zeros((2, 3)))
            np.testing.assert_allclose(command_at(action, .975)[0], action['peak_force_N'], atol=1e-9)
            np.testing.assert_array_equal(command_at(action, 3.), np.zeros((2, 3)))
            np.testing.assert_array_equal(command_at(action, 1.)[1], np.zeros(3))

    def test_selection_is_immutable_and_only_selected_sensors_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); public = root/'public/e0002'; public.mkdir(parents=True)
            menu = calibrated_menu(np.zeros((2, 3)), np.eye(3), np.eye(4))
            (public/'action-menu.json').write_text(json.dumps(menu))
            selected = record_selection(root, 'test-policy', 'e0002', 'poke_2', 'posterior-hash')
            with self.assertRaises(FileExistsError):
                record_selection(root, 'test-policy', 'e0002', 'poke_0')
            with self.assertRaises(ValueError):
                record_selection(root, 'other-policy', 'e0002', 'evaluation')
            sensor = root/'outcomes/e0002/poke_2/observations'
            for folder in ['rgb', 'depth', 'masks']: (sensor/folder).mkdir(parents=True)
            np.savez(sensor/'cameras.npz', K=np.eye(3), times_s=[0., 3.])
            (sensor/'rgb/00000.png').touch()
            # No private tree, nodal labels or any unselected response exists.
            loaded = load_selected_measurement(root, selected)
            self.assertEqual(loaded['action']['action_id'], 'poke_2')
            self.assertEqual(set(loaded), {'action', 'known_handle_centers_m', 'cameras', 'rgb_paths', 'depth_paths', 'mask_paths'})
            self.assertEqual(len(loaded['rgb_paths']), 1)
            with self.assertRaises(ValueError): load_selected_measurement(root, public/'action-menu.json')
            menu['candidates'][0]['cost'] = 2.
            (public/'action-menu.json').write_text(json.dumps(menu))
            with self.assertRaises(ValueError): load_selected_measurement(root, selected)


if __name__ == '__main__': unittest.main()
