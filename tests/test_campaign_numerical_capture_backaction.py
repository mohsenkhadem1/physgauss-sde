import ast
from pathlib import Path
import unittest
from experiments.campaign_numerical_capture_backaction import derive_worker


class BackactionTests(unittest.TestCase):
    def test_explicit_derivation_keeps_physics_and_never_fakes_sensor_evidence(self):
        p=Path(__file__).parents[1]/'experiments/isaac_calibrated_probes_v03.py';source=p.read_text()
        function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run_worker')
        original=ast.get_source_segment(source,function)+'\n';derived=derive_worker(original);compile(derived,'derived','exec')
        for text in ['world.step(render=False, update_fabric=True)','handles.apply_forces(force[handle_order])','body.set_nodal_positions','capture_current_state(']:
            self.assertEqual(original.count(text),derived.count(text))
        self.assertNotIn('Image.fromarray',derived)
        self.assertNotIn("'capture-audit.json'",derived)
        self.assertNotIn('maximum_sensor_ray_error_m',derived)
        self.assertIn("'sensor_qc_performed': False",derived)
        self.assertIn("'physics-observation-audit.json'",derived)
        with self.assertRaises(ValueError):derive_worker(original.replace("'capture-audit.json'","'unexpected.json'"))


if __name__=='__main__':unittest.main()
