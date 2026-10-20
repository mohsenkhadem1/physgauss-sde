import sys
from pathlib import Path
import unittest
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from isaac_dataset import make_plan,geometry,camera_pose,forces_at,boundary_shift,envelope,load_model_inputs
from validate_isaac_dataset import ray_depth,require
from describe_isaac_dataset import visibility_labels


class DatasetTests(unittest.TestCase):
    def test_factorial_plan_and_grouped_splits(self):
        p=make_plan();self.assertEqual(p,make_plan());self.assertEqual(len(p['episodes']),48)
        groups={}
        for case in p['episodes']:groups.setdefault(case['group_id'],[]).append(case)
        self.assertEqual(len(groups),16)
        for cases in groups.values():
            self.assertEqual(len(cases),3);self.assertEqual(len({x['split'] for x in cases}),1)
            self.assertEqual(len({x['seed'] for x in cases}),1)
        for split,count in [('train',24),('validation',12),('test',12)]:
            cases=[c for c in p['episodes'] if c['split']==split]
            self.assertEqual(len(cases),count)
            self.assertEqual(len({c['force_type'] for c in cases}),4)
            self.assertEqual(len({c['boundary_type'] for c in cases}),4)

    def test_all_geometries_have_disjoint_surface_boundaries_and_consistent_pairing(self):
        groups={}
        for case in make_plan()['episodes']:
            p,t,f,s=geometry(case);surface=set(np.unique(f));used=set(s['fixture_nodes'])
            self.assertLessEqual(used,surface)
            for patch in s['load_patches']:
                self.assertEqual(len(patch),9);self.assertFalse(used & set(patch));used.update(patch)
                self.assertLessEqual(set(patch),surface)
            previous=groups.setdefault(case['group_id'],s);self.assertEqual(previous,s)

    def test_camera_is_rigid_smooth_and_moves_independently_of_material(self):
        for case in make_plan()['episodes'][:16]:
            _,_,_,s=geometry(case);poses=np.array([camera_pose(s,t) for t in np.linspace(0,3,61)])
            np.testing.assert_allclose(poses[:,:3,:3]@poses[:,:3,:3].transpose(0,2,1),np.broadcast_to(np.eye(3),(61,3,3)),atol=1e-12)
            self.assertTrue(np.all(np.linalg.det(poses[:,:3,:3])>.999))
            self.assertGreater(np.linalg.norm(poses[-1,3,:3]-poses[0,3,:3]),.9)
            center=np.r_[s['camera']['center_m'],1.]
            for pose in poses:
                local=center@np.linalg.inv(pose);np.testing.assert_allclose(local[:2],0,atol=1e-12);self.assertLess(local[2],0)

    def test_forces_have_correct_normal_tangent_and_zero_net_couple(self):
        for case in make_plan()['episodes'][:16]:
            p,t,f,s=geometry(case);time=np.linspace(0,3,601);force=np.array([forces_at(s,x) for x in time])
            np.testing.assert_allclose(force[[0,-1]],0,atol=1e-12)
            if s['force_type']=='twist_couple':
                np.testing.assert_array_equal(force.sum(1),0)
                torque=np.cross(p[s['load_nodes'][0]]-p[s['load_nodes'][1]],force[:,0])
                self.assertGreater(np.linalg.norm(torque,axis=-1).max(),.001)
            if s['force_type']=='shear_oscillation':
                np.testing.assert_allclose(force[:,0]@s['normal_direction_world'],0,atol=1e-9)
            if case['boundary_type']=='moving_patch':self.assertGreater(np.linalg.norm(boundary_shift(s,1.5)),.007)
            else:np.testing.assert_array_equal(boundary_shift(s,1.5),0)

    def test_input_loader_does_not_need_private_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'observations').mkdir()
            np.savez(root/'reference.npz',rest_points_m=np.zeros((3,3)))
            np.savez(root/'observations/cameras.npz',frame_ids=[0,5])
            # Private files are absent: loading inputs must still work.
            inputs=load_model_inputs(root)
            self.assertEqual(set(inputs),{'reference','cameras','rgb_paths','mask_paths','depth_paths'})
            np.testing.assert_array_equal(inputs['cameras']['frame_ids'],[0,5])

    def test_independent_depth_check_rejects_wrong_camera(self):
        points=np.array([[-1.,-1.,-2.],[1.,-1.,-2.],[0.,1.,-2.]])
        faces=np.array([[0,1,2]]);k=np.array([[100.,0,50],[0,100.,50],[0,0,1.]])
        pose=np.eye(4);xs=np.array([49]);ys=np.array([49])
        np.testing.assert_allclose(ray_depth(points,faces,pose,k,xs,ys),2.)
        pose[3,2]=1.
        with self.assertRaises(ValueError):
            require(abs(ray_depth(points,faces,pose,k,xs,ys)[0]-2.)<.0002,'Wrong camera/depth alignment')

    def test_visibility_leaves_sensor_edges_ambiguous(self):
        p=np.array([[-1.,-1.,-2.],[1.,-1.,-2.],[0.,1.,-2.]])
        faces=np.array([[0,1,2]]);means=np.array([[0.,0.,-2.]])
        pose=np.eye(4);k=np.array([[100.,0,320],[0,100.,240],[0,0,1.]])
        def label(depth):return visibility_labels(means,p,faces,np.array([0]),pose,k,depth)[0]
        self.assertEqual(label(np.full((480,640),2.)),1)
        self.assertEqual(label(np.full((480,640),1.)),0)
        self.assertEqual(label(np.zeros((480,640))),2)


if __name__=='__main__':unittest.main()
