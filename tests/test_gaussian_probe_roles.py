from pathlib import Path
import sys
import json
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from gaussian_probe_roles import load_role_manifest,control_role,require_teacher_control,known_control_label,teacher_control_label
from isaac_dataset import geometry
from gaussian_operator import physical_vector

class ProbeRoleTests(unittest.TestCase):
    def setUp(self):
        self.manifest=load_role_manifest(Path(__file__).resolve().parents[1]/'configs/gaussian_probe_roles_v01.json','6d2781a71a24ebfa9cb0691ce0dd7ae889c15eae1b38ca3164618523d5552c84')

    def test_only_declared_teacher_candidates_are_eligible(self):
        require_teacher_control(self.manifest,'t0000','poke_3','train')
        for case,action,split in [('t0012','poke_0','train'),('t0000','evaluation','train'),('t0000','reset_replay','train'),('t0000','poke_0','validation'),('t0019','poke_0','train')]:
            with self.assertRaises(ValueError):require_teacher_control(self.manifest,case,action,split)
        self.assertEqual(control_role(self.manifest,'t0012','poke_0','train'),'calibration_only')
        self.assertEqual(control_role(self.manifest,'t0000','evaluation','train'),'evaluation_only')

    def test_actual_t0000_known_force_label_replaces_private_original_force(self):
        root=Path(__file__).resolve().parents[1]/'runs/gaussian-boundary-neural-v01'
        if not (root/'plan.json').exists():self.skipTest('Optional original recipe artifact unavailable')
        case=next(c for c in json.loads((root/'plan.json').read_text())['episodes'] if c['episode_id']=='t0000')
        points,_,_,spec=geometry(case);original=physical_vector(case,spec,points)
        action={'action_id':'poke_3','peak_force_N':[0,0,.02],'force_type_index':2}
        label,bc,family=teacher_control_label(self.manifest,case,spec,points,action)
        with self.assertRaises(ValueError):teacher_control_label(self.manifest,{**case,'episode_id':'t0012'},spec,points,action)
        np.testing.assert_array_equal(label[:16],original[:16]);np.testing.assert_array_equal(label[22:],original[22:])
        np.testing.assert_allclose(label[16:22],[0,0,.02,0,0,0],atol=1e-9)
        self.assertEqual(family,2);self.assertEqual(bc,0)
        with self.assertRaises(ValueError):known_control_label({**case,'split':'test'},spec,points,action)
        with self.assertRaises(ValueError):known_control_label(case,spec,points,{**action,'action_id':'evaluation'})

if __name__=='__main__':unittest.main()
