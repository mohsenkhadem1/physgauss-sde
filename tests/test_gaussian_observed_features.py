import sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
from gaussian_observed_features import (slab_depth,sample_depth_bilinear,reference_cuboid_sensor,
    projective_depth_features,load_observed_arrays,ObservedFeatureEncoder)
from isaac_dataset import make_plan,geometry,appearance,camera_pose


class MeasuredFeatureTests(unittest.TestCase):
    def test_parallel_slab_and_axis_depth(self):
        origin=np.array([0.,0.,1.]);rays=np.array([[0,0,-1.],[1,0,0],[0,0,1.]])
        result=slab_depth(origin,rays,[-.1,-.1,.2],[.1,.1,.3])
        self.assertAlmostEqual(result[0],.7)
        self.assertTrue(np.isinf(result[1:]).all())

    def test_depth_interpolation_and_background_rejection(self):
        depth=np.array([[.8,.801,.802],[.801,.802,.803],[.802,.803,.804]])
        value,valid=sample_depth_bilinear(depth,np.array([1.]),np.array([1.]))
        self.assertTrue(valid[0]);self.assertAlmostEqual(value[0],4/(1/.8+2/.801+1/.802))
        depth[0,0]=0;_,valid=sample_depth_bilinear(depth,np.array([1.]),np.array([1.]))
        self.assertFalse(valid[0])

    def test_static_depth_features_have_no_motion(self):
        p,t,f,spec=geometry(make_plan()['episodes'][0]);colors,g=appearance(p,f)
        reference={'rest_points_m':p,'surface_faces':f,**g}
        pose=camera_pose(spec,0);k=np.array([[622.,0,320],[0,622.,240],[0,0,1.]])
        sensor=reference_cuboid_sensor(reference,pose,k,(640,480),supersample=1)
        feature,audit=projective_depth_features(reference,pose,k,sensor['depth_m'])
        self.assertGreater(audit['valid_depth_kernels'],100)
        values=feature.reshape(6,9)
        # Bilinear sampling blends two visibleplanes only immediately at edges.
        self.assertLess(abs(values[:,1]).max(),.01)

    def test_rgb_loader_never_requires_depth_or_truth(self):
        p,t,f,spec=geometry(make_plan()['episodes'][0]);colors,g=appearance(p,f)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'observations/rgb').mkdir(parents=True);(root/'observations/masks').mkdir()
            np.savez(root/'reference.npz',rest_points_m=p,surface_faces=f,**g)
            np.savez(root/'observations/cameras.npz',frame_ids=[0],times_s=[0.],K=np.array([[622.,0,320],[0,622.,240],[0,0,1.]]),camera_to_world_usd_row=camera_pose(spec,0)[None])
            Image.fromarray(np.full((480,640,3),200,np.uint8)).save(root/'observations/rgb/00000.png')
            Image.fromarray(np.full((480,640),255,np.uint8)).save(root/'observations/masks/00000.png')
            result=load_observed_arrays(root,'rgb',(32,32));self.assertNotIn('depth_native_m',result)
            self.assertFalse((root/'truth').exists())
            with self.assertRaises(FileNotFoundError):load_observed_arrays(root,'rgbd',(32,32))

    def test_prior_ignores_observations_and_masked_features_are_ignored(self):
        torch.set_num_threads(2);x=torch.randn(2,3,65);mask=torch.tensor([[1,0,1],[1,0,1]],dtype=torch.bool)
        prior=ObservedFeatureEncoder(65,mode='prior');torch.testing.assert_close(prior(x),prior(x+100))
        model=ObservedFeatureEncoder(65).eval()
        with torch.no_grad():
            first=model(x,mask);x[:,1]=100;second=model(x,mask)
        torch.testing.assert_close(first,second)

if __name__=='__main__':unittest.main()
