import copy
import unittest
import numpy as np
from experiments.campaign_numerical_solver_coordinates import translate_reference,assert_paused_state_unchanged


class TranslationTests(unittest.TestCase):
    def test_paused_capture_guard_detects_state_mutation_without_clock_advance(self):
        state={'q':np.array([1.,2.]),'time':np.array(0.),'steps':np.array(10)}
        self.assertTrue(assert_paused_state_unchanged(state,copy.deepcopy(state)))
        for key in ['q','steps']:
            changed=copy.deepcopy(state);changed[key]=changed[key]+1
            with self.assertRaises(RuntimeError):assert_paused_state_unchanged(state,changed)

    def test_mesh_support_load_targets_offsets_and_camera_projection_equivariant(self):
        points=np.array([[-.1,0,.21],[.1,0,.29],[0,.03,.25]])
        spec={'support_centers_m':[[.1,0,.29]],'camera':{'center_m':[0,0,.25]},
            'load_nodes':[0,2],'load_patches':[[0,1],[1,2]],'fixture_nodes':[1],
            'motion_direction_world':[0,1,0]}
        pose=np.eye(4);pose[3,:3]=[.3,-.8,.7];shift=np.array([0,0,-.25]);before=copy.deepcopy(spec)
        p,s,c=translate_reference(points,spec,pose,shift)
        self.assertEqual(spec,before)
        np.testing.assert_allclose(p-points,shift[None].repeat(3,0),atol=1e-16)
        np.testing.assert_allclose(np.array(s['support_centers_m'])-spec['support_centers_m'],shift[None])
        for node,patch in zip(spec['load_nodes'],spec['load_patches']):
            np.testing.assert_allclose(p[patch]-p[node],points[patch]-points[node],atol=1e-16)
        for t in [0,.5,1,3]:
            motion=.008*np.sin(t)*np.array(spec['motion_direction_world'])
            np.testing.assert_allclose(p[spec['fixture_nodes']]+motion,points[spec['fixture_nodes']]+motion+shift)
        camera_before=np.c_[points,np.ones(3)]@np.linalg.inv(pose)
        camera_after=np.c_[p,np.ones(3)]@np.linalg.inv(c)
        np.testing.assert_allclose(camera_before,camera_after,atol=1e-15)
        np.testing.assert_allclose(camera_before[:,:2]/camera_before[:,2:3],camera_after[:,:2]/camera_after[:,2:3],atol=1e-15)
        # A pixel's world ray origin translates and its direction stays fixed.
        pixel_ray=np.array([.1,-.2,-1.]);np.testing.assert_array_equal(pixel_ray@pose[:3,:3],pixel_ray@c[:3,:3])
        np.testing.assert_allclose(c[3,:3]-pose[3,:3],shift)


if __name__=='__main__':unittest.main()
