import tempfile
from pathlib import Path
import sys
import unittest

import numpy as np

from experiments.isaac_acquisition_views import (
    camera_choices, load_public_view_queries, saved_state,
    selected_candidate_paths, usd_look_at, validate_capture,
)


class AcquisitionViewTests(unittest.TestCase):
    def test_usd_camera_basis_projects_center_and_is_right_handed(self):
        center=np.array([.01,-.02,.25])
        for azimuth in [-135,-45,45,135,75]:
            pose=usd_look_at(center,azimuth,25,.92)
            np.testing.assert_allclose(pose[:3,:3]@pose[:3,:3].T,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(pose[:3,:3]),1.)
            camera_point=(center-pose[3,:3])@pose[:3,:3].T
            np.testing.assert_allclose(camera_point,[0,0,-.92],atol=1e-12)

    def test_candidate_and_evaluation_camera_roles_and_no_motion_input(self):
        reference={'rest_points_m':np.array([[-.12,-.06,.21],[.12,.06,.29]])}
        choices=camera_choices(reference,1.5)
        self.assertEqual(sum(c['role']=='candidate' for c in choices),4)
        self.assertEqual(sum(c['role']=='evaluation' for c in choices),1)
        self.assertEqual(len({tuple(np.asarray(c['camera_to_world_usd_row']).ravel()) for c in choices}),5)
        self.assertTrue(all(c['time_s']==1.5 for c in choices))

    def test_camera_only_loader_and_acquisition_exclude_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);public=root/'public/e0000';public.mkdir(parents=True)
            np.savez(public/'cameras.npz',view_ids=['candidate00','evaluation00'],
                     roles=['candidate','evaluation'],K=np.eye(3),times_s=[1.5,1.5])
            # No truth directory exists; querying poses must still work.
            queries=load_public_view_queries(root,'e0000')
            self.assertEqual(len(queries['view_ids']),2)
            paths=selected_candidate_paths(root,'e0000','candidate00')
            self.assertEqual(paths['query_index'],0)
            with self.assertRaises(ValueError):selected_candidate_paths(root,'e0000','evaluation00')
            with self.assertRaises(ValueError):selected_candidate_paths(root,'e0000','../evaluation00')

    def test_state_requires_saved_time_and_never_interpolates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'truth').mkdir()
            positions=np.arange(18,dtype=float).reshape(2,3,3)
            np.savez(root/'truth/trajectory.npz',times_s=[0.,1.5],positions_m=positions)
            index,time,points=saved_state(root,1.5)
            self.assertEqual(index,1);self.assertEqual(time,1.5)
            np.testing.assert_array_equal(points,positions[1])
            with self.assertRaises(ValueError):saved_state(root,1.4)

    def test_sensor_audit_accepts_known_plane_and_rejects_stale_depth(self):
        # The independent existing dataset validator is also a CLI module.
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
        try:
            positions=np.array([[-.3,-.2,-1],[.3,-.2,-1],[.3,.2,-1],[-.3,.2,-1]])
            faces=np.array([[0,1,2],[0,2,3]])
            k=np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
            depth=np.zeros((480,640));depth[141:339,171:469]=1.
            rgb=np.zeros((480,640,3),dtype=np.uint8)
            checks=validate_capture(rgb,depth,positions,faces,np.eye(4),k)
            self.assertLess(checks['ray_depth_max_error_m'],1e-12)
            self.assertEqual(checks['depth_rays_tested'],64)
            depth[depth>0]+=.001
            with self.assertRaises(ValueError):validate_capture(rgb,depth,positions,faces,np.eye(4),k)
            with self.assertRaises(ValueError):validate_capture(rgb[:100],depth,positions,faces,np.eye(4),k)
        finally:
            sys.path.pop(0)


if __name__=='__main__':unittest.main()
