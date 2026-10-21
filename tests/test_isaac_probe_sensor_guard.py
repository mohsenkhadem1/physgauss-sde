from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from isaac_probe_sensor_guard import ray_depth_conditioning,well_conditioned_depth_state_agreement
from isaac_calibrated_probes_v02 import depth_state_agreement

class ConditionedRayTests(unittest.TestCase):
    def test_plane_agrees_and_millimetre_stale_depth_rejected(self):
        points=np.array([[-2,-2,-2],[2,-2,-2],[2,2,-2],[-2,2,-2.]])
        faces=np.array([[0,1,2],[0,2,3]]);pose=np.eye(4);k=np.array([[622.,0,320],[0,622,240],[0,0,1]])
        depth=np.full((480,640),2.,np.float32);mask=np.ones_like(depth,bool)
        current=well_conditioned_depth_state_agreement(points,faces,pose,k,depth,mask)
        self.assertTrue(current['passed']);self.assertEqual(current['tolerance_m'],2e-6)
        points[:,2]-=.001
        self.assertFalse(well_conditioned_depth_state_agreement(points,faces,pose,k,depth,mask)['passed'])

    def test_exact_t0013_static_failure_passes_conditioned_same_tolerance(self):
        p=Path(__file__).resolve().parents[1]/'runs/operator-dev/t0013-depth-numerics-static-v01'
        if not (p/'truth/depth/00000.npz').exists():self.skipTest('Optional preserved static RTX regression')
        from PIL import Image
        q=np.load(p/'truth/trajectory.npz')['positions_m'][0];c=np.load(p/'truth/cameras.npz');f=np.load(p/'reference.npz')['surface_faces'];d=np.load(p/'truth/depth/00000.npz')['depth_m'];m=np.asarray(Image.open(p/'truth/masks/00000.png'))>0
        old=depth_state_agreement(q,f,c['camera_to_world_usd_row'][0],c['K'],d,m)
        new=well_conditioned_depth_state_agreement(q,f,c['camera_to_world_usd_row'][0],c['K'],d,m)
        self.assertFalse(old['passed']);self.assertTrue(new['passed']);self.assertEqual(new['tolerance_m'],old['tolerance_m'])
        self.assertGreater(new['excluded_grazing_rays'],0);self.assertLess(new['maximum_error_m'],5e-7)

    def test_actual_old_stale_e0008_remains_rejected(self):
        p=Path(__file__).resolve().parents[1]/'runs/isaac-calibrated-probes-validation-v01'
        if not (p/'private/e0008/evaluation/trajectory.npz').exists():self.skipTest('Optional actual stale regression')
        from PIL import Image
        q=np.load(p/'private/e0008/evaluation/trajectory.npz')['positions_m'];f=np.load(p/'private/e0008/reference.npz')['surface_faces'];d=np.load(p/'outcomes/e0008/evaluation/observations/depth/00003.npz')['depth_m'];m=np.asarray(Image.open(p/'outcomes/e0008/evaluation/observations/masks/00003.png'))>0;c=np.load(p/'outcomes/e0008/evaluation/observations/cameras.npz')
        self.assertFalse(well_conditioned_depth_state_agreement(q[3],f,c['camera_to_world_usd_row'][3],c['K'],d,m)['passed'])
        self.assertTrue(well_conditioned_depth_state_agreement(q[2],f,c['camera_to_world_usd_row'][3],c['K'],d,m)['passed'])

if __name__=='__main__':unittest.main()
