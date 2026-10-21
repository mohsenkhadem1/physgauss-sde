import sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
from gaussian_texture_features import (fit_color_curve,invert_color_curve,apparent_tangent_fit,
    texture_geometry,load_texture_features)
from isaac_dataset import make_plan,geometry,appearance,camera_pose

class TextureTests(unittest.TestCase):
    def reference(self):
        p,t,f,s=geometry(make_plan()['episodes'][0]);colors,g=appearance(p,f)
        return {'rest_points_m':p,'surface_faces':f,'vertex_colors':colors,**g},s

    def test_color_curve_inverts_nonlinear_palette(self):
        w=np.linspace(0,1,101)
        rgb=np.stack((.3+.5*w+.05*w*w,.5+.2*w-.02*w*w,.8-.4*w+.02*w**3),-1)
        curve=fit_color_curve(w,rgb)
        recovered,error=invert_color_curve(curve,rgb)
        np.testing.assert_allclose(recovered,w,atol=1e-6)
        self.assertLess(error.max(),1e-6)

    def test_known_checker_has_aperture_nullspace(self):
        reference,_=self.reference();g=texture_geometry(reference)
        for face in range(6):
            gradient=g['gradient_world'][g['face']==face]
            values=np.linalg.eigvalsh(gradient.T@gradient)
            self.assertEqual(int((values>values.max()*1e-6).sum()),1)
        self.assertEqual(np.linalg.matrix_rank(g['gradient_world'],tol=1e-6),2)

    def test_tangent_shift_recovers_observable_component(self):
        # Alternating checker gradients only observe x-y, not x+y.
        gradient=np.tile([[50.,-50.,0.],[-50.,50.,0.]],(20,1))
        physical=np.array([.001,-.0005,0.]);delta=-gradient@physical
        fit=apparent_tangent_fit(gradient,delta,face=5)
        self.assertEqual(fit['rank'],1)
        np.testing.assert_allclose(gradient[:,:2]@fit['shift_tangent_m'],gradient@physical,atol=1e-5)
        self.assertAlmostEqual(float(fit['shift_tangent_m'].sum()),0.,places=10)

    def test_projective_gradient_includes_normal_motion_parallax(self):
        # Exact translated-plane ray intersection, without simulator geometry.
        point=np.array([.03,.02,.01]);eye=np.array([.4,.3,.6])
        ray=point-eye;normal=np.array([0.,0.,1.]);gradient=np.array([50.,-50.,0.])
        jacobian=gradient-(gradient@ray)/(normal@ray)*normal
        for axis in range(3):
            displacement=np.eye(3)[axis]*1e-6
            t=(normal@(point+displacement-eye))/(normal@ray)
            material_point=eye+t*ray-displacement
            delta=gradient@(material_point-point)
            self.assertAlmostEqual(delta,float(-jacobian@displacement),places=12)
        self.assertGreater(abs(jacobian[2]),1.)

    def test_public_prefix_loader_never_needs_depth_truth_or_future_pixels(self):
        reference,spec=self.reference();w=np.linspace(0,1,30)
        rgb=np.stack((.3+.5*w,.5+.2*w,.8-.4*w),-1)
        curve=fit_color_curve(w,rgb);calibration={'global_curve':curve,'face_curves':{}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'observations/rgb').mkdir(parents=True);(root/'observations/masks').mkdir()
            np.savez(root/'reference.npz',**reference)
            np.savez(root/'observations/cameras.npz',frame_ids=[0,40],times_s=[0.,2.],
                K=np.array([[622.,0,320],[0,622.,240],[0,0,1.]]),
                camera_to_world_usd_row=np.array([camera_pose(spec,0),camera_pose(spec,2.)]))
            Image.fromarray(np.full((480,640,3),180,np.uint8)).save(root/'observations/rgb/00000.png')
            Image.fromarray(np.full((480,640),255,np.uint8)).save(root/'observations/masks/00000.png')
            result=load_texture_features(root,calibration,max_time_s=1.)
            self.assertEqual(result['features'].shape,(1,78))
            self.assertEqual(result['frame_ids'].tolist(),[0])
            self.assertFalse((root/'truth').exists());self.assertFalse((root/'observations/depth').exists())

if __name__=='__main__':unittest.main()
