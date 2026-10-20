import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
from gaussian_observation import usd_projection, IsaacGaussianRenderer
from isaac_dataset import make_plan,geometry,appearance,camera_pose


class ObservationTests(unittest.TestCase):
    def test_usd_projection_and_jacobian(self):
        # Identity USD camera looks toward-Z and+Y is image-up.
        points=torch.tensor([[[.2,.1,-2.]]],dtype=torch.float64,requires_grad=True)
        poses=torch.eye(4,dtype=torch.float64)[None]
        k=torch.tensor([[400.,0,320],[0,400,240],[0,0,1]],dtype=torch.float64)
        uv,depth,jac=usd_projection(points,poses,k)
        torch.testing.assert_close(uv,torch.tensor([[[360.,220.]]],dtype=torch.float64))
        torch.testing.assert_close(depth,torch.tensor([[2.]],dtype=torch.float64))
        numeric=torch.autograd.functional.jacobian(lambda p:usd_projection(p,poses,k)[0],points)[0,0,:,0,0,:]
        torch.testing.assert_close(jac[0,0],numeric)

    def test_rest_render_and_nonzero_deformation_gradient(self):
        case=make_plan()['episodes'][0];p,t,f,s=geometry(case);colors,g=appearance(p,f)
        reference={'rest_points_m':p,'tetrahedra':t,**g}
        renderer=IsaacGaussianRenderer(reference,size=(16,12),dtype=torch.float64)
        positions=torch.tensor(p[None],requires_grad=True,dtype=torch.float64)
        pose=np.array([camera_pose(s,0)])
        k=np.array([[640*35/36,0,320],[0,480*35/27,240],[0,0,1.]])
        a=renderer(positions,pose,k)
        self.assertTrue(torch.isfinite(a['rgb']).all())
        self.assertGreater(float(a['alpha'].detach().max()),.5)
        (a['rgb'][...,0]*torch.arange(16)[None,None,:]).sum().backward()
        self.assertTrue(torch.isfinite(positions.grad).all())
        self.assertGreater(float(positions.grad.norm()),1e-3)



class PublicInputTests(unittest.TestCase):
    def test_no_truth_or_depth_required(self):
        import tempfile
        from PIL import Image
        from gaussian_observation import load_rgb_inputs
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'observations/rgb').mkdir(parents=True);(p/'observations/masks').mkdir()
            np.savez(p/'reference.npz',rest_points_m=np.zeros((2,3)))
            np.savez(p/'observations/cameras.npz',frame_ids=np.array([0]),times_s=np.array([0.]),K=np.eye(3),camera_to_world_usd_row=np.eye(4)[None])
            Image.fromarray(np.zeros((8,8,3),dtype=np.uint8)).save(p/'observations/rgb/00000.png')
            Image.fromarray(np.ones((8,8),dtype=np.uint8)*255).save(p/'observations/masks/00000.png')
            inputs=load_rgb_inputs(p,(8,8))
            self.assertEqual(inputs['rgb'].shape,(1,8,8,3))
            self.assertFalse((p/'truth').exists())

    def test_encoder_ignores_masked_frame_and_has_camera_signal(self):
        from gaussian_video_encoder import GaussianVideoEncoder
        torch.manual_seed(10);model=GaussianVideoEncoder().eval()
        images=torch.rand(1,3,4,32,32);poses=torch.eye(4)[None].expand(3,4,4).clone()
        poses[:,3,2]=1.;k=torch.tensor([[622.,0,320],[0,622.,240],[0,0,1.]])
        times=torch.tensor([0.,1.,2.]);rest=torch.rand(8,3)*.2;mask=torch.tensor([[True,False,True]])
        with torch.no_grad():
            first=model(images,poses,k,times,rest,mask)
            changed=images.clone();changed[:,1]=9
            second=model(changed,poses,k,times,rest,mask)
            alteredpose=poses.clone();alteredpose[0,3,0]+=.1
            third=model(images,alteredpose,k,times,rest,mask)
        torch.testing.assert_close(first,second)
        self.assertEqual(first.shape,(1,128))
        self.assertGreater(float((third-first).norm()),1e-4)

class CropTests(unittest.TestCase):
    def test_known_reference_crop_calibration(self):
        from gaussian_observation import project_reference_crops
        case=make_plan()['episodes'][0];p,t,f,spec=geometry(case)
        poses=np.array([camera_pose(spec,0),camera_pose(spec,2.)])
        k=np.array([[622.,0,320],[0,622.,240],[0,0,1.]])
        crop=project_reference_crops({'rest_points_m':p},poses,k,(192,160))
        native,_,_=usd_projection(torch.tensor(np.broadcast_to(p,(2,*p.shape))),torch.tensor(poses),torch.tensor(k))
        for frame in range(2):
            projected,_,_=usd_projection(torch.tensor(p[None]),torch.tensor(poses[frame:frame+1]),torch.tensor(crop['K'][frame]))
            box=crop['crop_boxes_xyxy'][frame]
            expected=(native[frame].numpy()-box[:2])*np.array([192,160])/(box[2:]-box[:2])
            np.testing.assert_allclose(projected[0].numpy(),expected,atol=1e-10)
            self.assertGreaterEqual(expected.min(),0)
            self.assertTrue((expected.max(0)<[192,160]).all())

if __name__=='__main__':unittest.main()
