"""Numerical qualification of the declared linear Gaussian-field mechanics."""
import math
import unittest
import torch
from scripts.joint_field_identification.mechanics import LinearGaussianBody


class TestJointFieldMechanics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.theta=torch.tensor([[math.log(30000.),.4,-.02,math.log(400.),.02]],dtype=torch.float64)

    def test_volume_mass_surface_and_stiffness(self):
        for shape,fraction in [('block',1.),('step',.75)]:
            body=LinearGaussianBody(shape)
            self.assertAlmostEqual(float(body.volumes.sum()),.12*.06*.06*fraction,places=14)
            self.assertAlmostEqual(float(body.mass.sum()),500*.12*.06*.06*fraction,places=13)
            self.assertTrue(bool((body.volumes>0).all()))
            self.assertTrue(torch.allclose(body.surface_interpolation.sum(-1),torch.ones(len(body.surface_points),dtype=torch.float64)))
            op=body.operators(self.theta)
            self.assertLess(float((op['stiffness']-op['stiffness'].transpose(-1,-2)).abs().max()),1e-10)
            self.assertGreater(float(torch.linalg.eigvalsh(op['stiffness']).min()),0.)
            elastic_eigenvalues=torch.linalg.eigvalsh(op['elastic_stiffness'])[0]
            self.assertLess(float(elastic_eigenvalues[:6].abs().max()),1e-9)
            self.assertGreater(float(elastic_eigenvalues[6]),0.)

    def test_rigid_and_affine_strain(self):
        body=LinearGaussianBody()
        translation=body.points.new_tensor([.03,-.04,.01]).expand_as(body.points)
        omega=body.points.new_tensor([.02,.03,-.01]).expand_as(body.points)
        rotation=torch.linalg.cross(omega,body.points)
        for rigid in (translation,rotation):
            self.assertLess(float(body.strain(rigid).abs().max()),1e-14)
            K=body.operators(self.theta)['elastic_stiffness'][0]
            self.assertLess(float((K@rigid.flatten()).abs().max()),1e-10)
        F=body.points.new_tensor([[.01,.02,.03],[-.02,-.01,.005],[.01,.003,.015]])
        u=body.points@F.T
        expected=(F+F.T)/2
        self.assertTrue(torch.allclose(body.strain(u),expected.expand(len(body.elements),-1,-1),atol=1e-14))

    def test_normalized_forces_and_support(self):
        for shape in ('block','step'):
            body=LinearGaussianBody(shape)
            op=body.operators(self.theta)
            total=op['unknown_forces'].reshape(1,body.n,3,2).sum(1)[0]
            self.assertTrue(torch.allclose(total,body.points.new_tensor([[0.,0.],[0.,-1.],[-1.,0.]]),atol=1e-14))
            self.assertAlmostEqual(float(op['support_weights'].sum()),1.,places=14)
            probes=op['probe_forces'].reshape(body.n,3,6).sum(0)
            expected=body.points.new_tensor([[0,0],[0,-1],[-1,0]]).repeat(1,3)
            self.assertTrue(torch.allclose(probes,expected,atol=1e-14))
            self.assertTrue(bool((op['support_weights']>=0).all()))
            self.assertEqual(float(op['support_weights'][:,body.points[:,2]>1e-12].sum()),0.)

    def test_equilibrium_reciprocity_and_batch(self):
        body=LinearGaussianBody()
        batch=torch.cat([self.theta,self.theta+torch.tensor([[.2,-.1,.01,.1,-.01]],dtype=torch.float64)])
        result=body.solve(batch,include_operators=True)
        response=result['unit_response'].reshape(2,body.ndof,2)
        self.assertLess(float((result['stiffness']@response-result['unknown_forces']).abs().max()),1e-12)
        work=result['unknown_forces'].transpose(-1,-2)@response
        self.assertTrue(torch.allclose(work,work.transpose(-1,-2),atol=1e-13))
        for i in range(2):
            single=body.solve(batch[i])['unit_response'][0]
            self.assertTrue(torch.allclose(single,result['unit_response'][i],atol=1e-14))

    def test_interpolation_and_outside_rejection(self):
        for shape in ('block','step'):
            body=LinearGaussianBody(shape)
            q=body.points.new_tensor([[-.043,.007,.049],[.027,-.011,.019],[0.,0.,.03]])
            N=body.interpolation(q)
            self.assertTrue(torch.allclose(N@body.points,q,atol=1e-14))
            self.assertTrue(torch.allclose(N.sum(-1),torch.ones(len(q),dtype=torch.float64),atol=1e-14))
            with self.assertRaises(ValueError):body.interpolation([[.08,0.,.01]])
        with self.assertRaises(ValueError):LinearGaussianBody('step').interpolation([[.04,0.,.05]])

    def test_material_field_mean_and_gradient(self):
        body=LinearGaussianBody()
        theta=self.theta.clone().requires_grad_(True)
        E=body.material_modulus(theta)
        logmean=(E.log()*body.volumes).sum(-1)/body.volumes.sum()
        self.assertTrue(torch.allclose(logmean,theta[:,0],atol=1e-14))
        quantity=body.solve(theta)['unit_response'][0,-1,2,0]
        derivative=torch.autograd.grad(quantity,theta)[0]
        for i in range(5):
            step=1e-6 if i in (2,4) else 1e-5
            d=torch.zeros_like(theta);d[0,i]=step
            fd=(body.solve(theta.detach()+d)['unit_response'][0,-1,2,0]-
                body.solve(theta.detach()-d)['unit_response'][0,-1,2,0])/(2*step)
            self.assertAlmostEqual(float(derivative[0,i]),float(fd),delta=max(1e-8,float(fd.abs())*2e-5))

    def test_custom_probe_and_continuous_material(self):
        body=LinearGaussianBody()
        specs=body.probe_specs+[dict(x=x,mode=m) for x in (-.02,.02) for m in ('top','side')]
        result=body.solve(self.theta,True,probe_specs=specs)
        self.assertEqual(result['probe_response'].shape,(1,body.n,3,10))
        default=body.solve(self.theta)
        self.assertTrue(torch.allclose(result['probe_response'][...,:6],default['probe_response'],atol=1e-14))
        field=body.modulus_at_points(self.theta,body.centers)
        self.assertTrue(torch.allclose(field,result['element_modulus'],atol=1e-10))
        self.assertLess(float((result['stiffness']@result['probe_response'].reshape(1,body.ndof,10)-result['probe_forces']).abs().max()),1e-12)

    def test_mesh_refinement_diagnostic(self):
        # This tests decreasing error for a declared case, not population convergence.
        q=torch.tensor([[-.04,-.03,.06],[-.02,.03,.06],[.02,-.03,.03],[.04,.03,.03]],dtype=torch.float64)
        for shape in ('block','step'):
            response=[]
            for cells in [(4,2,2),(8,4,4),(12,6,6)]:
                body=LinearGaussianBody(shape,cells)
                response.append(body.point_response(body.solve(self.theta)['unit_response'],q)*.04)
            coarse=float((response[1]-response[0]).square().mean().sqrt())
            fine=float((response[2]-response[1]).square().mean().sqrt())
            self.assertLess(fine,.5*coarse)
            self.assertLess(fine,.00005)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cpu_gpu(self):
        cpu=LinearGaussianBody().solve(self.theta)['unit_response']
        gpu=LinearGaussianBody(device='cuda:0').solve(self.theta.to('cuda:0'))['unit_response'].cpu()
        self.assertTrue(torch.allclose(cpu,gpu,atol=1e-12,rtol=1e-10))


if __name__=='__main__':unittest.main()
