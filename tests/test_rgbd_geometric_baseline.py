import unittest

import numpy as np
import torch

from experiments.rgbd_geometric_baseline import (
    GeometricConfig,RBFGeometricTrajectory,depth_normals,farthest_point_indices,
    interpolate_knots,normalized_rbf,project_world,projective_correspondence,
    rotation_matrix,sample_image,sample_axis_depth,unproject_depth,volume_ratios,
)


class RGBDGeometricTests(unittest.TestCase):
    def setUp(self):
        self.pose=torch.eye(4);self.k=torch.tensor([[100.,0,20],[0,100.,15],[0,0,1.]])
        self.depth=torch.ones(30,40);self.mask=torch.ones(30,40,dtype=torch.bool)

    def test_projection_unprojection_and_pixel_center_sampling(self):
        uv=torch.tensor([[10.5,12.5],[25.25,16.75]])
        point=unproject_depth(uv,torch.tensor([1.,1.5]),self.pose,self.k)
        found,z=project_world(point,self.pose,self.k)
        torch.testing.assert_close(found,uv);torch.testing.assert_close(z,torch.tensor([1.,1.5]))
        image=torch.arange(30*40,dtype=torch.float32).reshape(30,40)
        torch.testing.assert_close(sample_image(image,uv[:1])[:,0],image[12,10].reshape(1))

    def test_observed_plane_normals_face_camera(self):
        normals,valid=depth_normals(self.depth,self.mask,self.pose,self.k)
        self.assertEqual(int(valid.sum()),26*36)
        torch.testing.assert_close(normals[valid],torch.tensor([0.,0.,1.]).expand(int(valid.sum()),3))

    def test_inverse_depth_sampling_is_exact_for_perspective_plane(self):
        yy,xx=torch.meshgrid(torch.arange(30,dtype=torch.float64)+.5,torch.arange(40,dtype=torch.float64)+.5,indexing='ij')
        depth=1/(1+.01*xx+.02*yy)
        uv=torch.tensor([[10.3,12.7],[25.9,16.1]],dtype=torch.float64)
        expected=1/(1+.01*uv[:,0]+.02*uv[:,1])
        observed,valid=sample_axis_depth(depth,uv)
        self.assertTrue(valid.all());torch.testing.assert_close(observed,expected,rtol=0,atol=1e-14)
        self.assertGreater(float((sample_image(depth,uv)[:,0]-expected).abs().max()),1e-6)

    def test_projective_icp_gradient_and_occlusion_rejection(self):
        normals,valid=depth_normals(self.depth,self.mask,self.pose,self.k)
        points=torch.tensor([[0.,0.,-1.01],[.02,0.,-1.],[0.,.02,-1.04]],requires_grad=True)
        predicted_normal=torch.tensor([[0.,0.,1.],[0.,0.,-1.],[0.,0.,1.]])
        data=projective_correspondence(points,predicted_normal,self.pose,self.k,self.depth,self.mask,normals,valid)
        self.assertEqual(data['valid'].tolist(),[True,False,False])
        loss=data['plane_residual_m'][data['valid']].square().mean()
        loss.backward();self.assertLess(float(points.grad[0,2]),0)
        torch.testing.assert_close(data['axis_residual_m'][0],torch.tensor(.01),atol=1e-6,rtol=0)

    def test_bilinear_support_rejects_depth_background_mixture(self):
        depth=self.depth.clone();depth[:,20:]=0
        normals,valid=depth_normals(depth,depth>0,self.pose,self.k)
        uv=torch.tensor([[19.75,15.5]])
        points=unproject_depth(uv,torch.ones(1),self.pose,self.k)
        data=projective_correspondence(points,torch.tensor([[0.,0.,1.]]),self.pose,self.k,depth,depth>0,normals,valid)
        self.assertFalse(bool(data['valid'][0]))

    def test_nonuniform_interpolation_is_exact_at_knots_and_initially_rest(self):
        knots=torch.tensor([0.,.25,1.,2.,3.]);values=torch.randn(5,2,3,requires_grad=True)
        for mode in ['linear','hermite']:
            found=interpolate_knots(knots,values,knots,mode)
            torch.testing.assert_close(found,values)
        t=torch.tensor([0.],requires_grad=True)
        derivative=torch.autograd.grad(interpolate_knots(knots,values,t).sum(),t)[0]
        torch.testing.assert_close(derivative,torch.zeros_like(derivative))

    def test_rbf_partition_and_initial_geometry_and_rigid_rotation(self):
        points=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=float)
        reference={'rest_points_m':points,'surface_faces':np.array([[0,1,2],[0,2,3]])}
        indices=farthest_point_indices(points,4);self.assertEqual(len(set(indices)),4)
        basis=normalized_rbf(torch.tensor(points),torch.tensor(points),.3)
        torch.testing.assert_close(basis.sum(-1),torch.ones(4,dtype=torch.float64))
        model=RBFGeometricTrajectory(reference,[0.,1.,3.],GeometricConfig(controls=4))
        with torch.no_grad():model.translations[1]=torch.tensor([.1,0,0]);model.rotations[1,2]=torch.pi/2
        result=model(torch.tensor([0.,1.]))
        torch.testing.assert_close(result[0],torch.tensor(points,dtype=torch.float32))
        expected=(model.rest-model.center)@rotation_matrix(torch.tensor([0.,0.,torch.pi/2])).T+model.center+torch.tensor([.1,0,0])
        torch.testing.assert_close(result[1],expected)
        loss=result.square().sum();loss.backward()
        self.assertTrue(torch.isfinite(model.rotations.grad).all())

    def test_inversions_are_reported_not_hidden(self):
        rest=torch.tensor([[0.,0.,0.],[1,0,0],[0,1,0],[0,0,1.]])
        tetra=torch.tensor([[0,1,2,3]])
        reflected=rest.clone();reflected[:,0]*=-1
        ratio=volume_ratios(torch.stack((rest,reflected)),rest,tetra)
        torch.testing.assert_close(ratio,torch.tensor([[1.],[-1.]]))


if __name__=='__main__':unittest.main()
