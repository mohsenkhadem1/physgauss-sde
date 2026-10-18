"""Independent analytic/full-matrix regressions for the stiff-mode quadrature bug."""
import importlib.util
import math
from pathlib import Path
import unittest
import torch

path=Path(__file__).resolve().parents[1]/'scripts'/'exact_pac_integrals.py'
spec=importlib.util.spec_from_file_location('pac_exact_integrals',path)
exact=importlib.util.module_from_spec(spec);spec.loader.exec_module(exact)


class ExactIntegralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_free_particle_closed_form(self):
        h=1/24;drive,k=exact.integrals(torch.zeros(1,1,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),h)
        torch.testing.assert_close(drive[0,:,0],torch.tensor([h*h/2,h],dtype=drive.dtype),atol=1e-14,rtol=1e-13)
        expected=torch.tensor([[h**3/3,h*h/2],[h*h/2,h]],dtype=drive.dtype)
        torch.testing.assert_close(k[0,0,0],expected,atol=1e-14,rtol=1e-13)

    def test_high_frequency_undamped_closed_form(self):
        h=1/24;w=torch.tensor([2.,600.,1200.],dtype=torch.float64)
        drive,k=exact.integrals(w.square()[None],torch.zeros(1,dtype=w.dtype),h)
        for i,value in enumerate(w):
            expected=torch.stack((torch.stack((h/(2*value**2)-torch.sin(2*value*h)/(4*value**3),torch.sin(value*h)**2/(2*value**2))),
                torch.stack((torch.sin(value*h)**2/(2*value**2),h/2+torch.sin(2*value*h)/(4*value)))))
            torch.testing.assert_close(k[0,i,i],expected,atol=2e-14,rtol=2e-10)
            torch.testing.assert_close(drive[0,i,i],(1-torch.cos(value*h))/value**2,atol=2e-14,rtol=2e-10)
            torch.testing.assert_close(drive[0,3+i,i],torch.sin(value*h)/value,atol=2e-14,rtol=2e-10)

    def test_full_coupled_covariance_matches_independent_van_loan(self):
        h=1/24;omega=torch.tensor([[0.,35.,1000.],[0.,75.,2500.]],dtype=torch.float64)
        damping=torch.tensor([.8,6.],dtype=omega.dtype);b,m=omega.shape
        drive,k=exact.integrals(omega.square(),damping,h)
        rng=torch.Generator().manual_seed(870)
        r=torch.randn(b,3,m,m,dtype=omega.dtype,generator=rng)
        spectral=r@r.transpose(-1,-2)+.0225*torch.eye(m,dtype=omega.dtype)
        actual=exact.covariance(spectral,k)
        a=omega.new_zeros(b,2*m,2*m);a[:,:m,m:]=torch.eye(m,dtype=omega.dtype)
        a[:,m:,:m]=-torch.diag_embed(omega.square());a[:,m:,m:]=-damping[:,None,None]*torch.eye(m,dtype=omega.dtype)
        block=omega.new_zeros(b,3,4*m,4*m);block[:,:,:2*m,:2*m]=a[:,None]
        block[:,:,2*m:,2*m:]=-a[:,None].transpose(-1,-2);block[:,:,m:2*m,3*m:]=spectral
        exp=torch.matrix_exp(block*h);f=torch.matrix_exp(a*h)
        expected=exp[:,:,:2*m,2*m:]@f[:,None].transpose(-1,-2)
        expected=(expected+expected.transpose(-1,-2))/2
        root=torch.linalg.cholesky(expected)
        difference=torch.linalg.solve_triangular(root,actual-expected,upper=False)
        difference=torch.linalg.solve_triangular(root,difference.transpose(-1,-2),upper=False).transpose(-1,-2)
        self.assertLess(float(torch.linalg.eigvalsh((difference+difference.transpose(-1,-2))/2).abs().max()),2e-8)
        affine=omega.new_zeros(b,3*m,3*m);affine[:,:2*m,:2*m]=a;affine[:,m:2*m,2*m:]=torch.eye(m,dtype=omega.dtype)
        torch.testing.assert_close(drive,torch.matrix_exp(affine*h)[:,:2*m,2*m:],atol=2e-12,rtol=2e-8)
        self.assertGreater(float(torch.linalg.eigvalsh(actual).min()),0.)

    def test_parameter_gradient_and_second_derivative(self):
        theta=torch.tensor([.4,-.3],dtype=torch.float64,requires_grad=True)
        def value(p):
            frequency=torch.tensor([[0.,30.,900.]],dtype=p.dtype)*torch.exp(.35*p[0])
            drive,k=exact.integrals(frequency.square(),(.8*torch.exp(.7*p[1])).reshape(1),1/24)
            return drive.square().sum()+k.square().sum()
        g,=torch.autograd.grad(value(theta),theta)
        delta=1e-5
        for i in range(2):
            direction=torch.zeros_like(theta);direction[i]=delta
            difference=(value(theta+direction)-value(theta-direction))/(2*delta)
            torch.testing.assert_close(g[i],difference,atol=1e-10,rtol=2e-5)
        h=torch.autograd.functional.hessian(value,theta,vectorize=True)
        self.assertTrue(bool(torch.isfinite(h).all()))
        torch.testing.assert_close(h,h.T,atol=1e-10,rtol=1e-8)


if __name__=='__main__':unittest.main()
