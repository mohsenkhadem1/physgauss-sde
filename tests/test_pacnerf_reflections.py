from pathlib import Path
import sys,unittest,math
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pacnerf_reflections import harmonic_reflections,logistic_slice_threshold,state_transition,exit_times


class ReflectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_multiple_wall_flow_is_reversible_and_energy_conserving(self):
        rng=torch.Generator().manual_seed(815)
        normals=torch.randn(16,9,3,dtype=torch.float64,generator=rng)
        offset=.2+torch.rand(16,9,dtype=torch.float64,generator=rng)
        q=torch.zeros(16,3,dtype=torch.float64);p=torch.randn(q.shape,dtype=q.dtype,generator=rng)
        a,b,bad,count=harmonic_reflections(q,p,normals,offset,4.1)
        self.assertFalse(bool(bad.any()));self.assertGreater(int(count.sum()),20)
        torch.testing.assert_close(a.square().sum(-1)+b.square().sum(-1),p.square().sum(-1),atol=1e-11,rtol=1e-11)
        back,back_p,bad,_=harmonic_reflections(a,-b,normals,offset,4.1)
        self.assertFalse(bool(bad.any()))
        torch.testing.assert_close(back,q,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(back_p,-p,atol=1e-10,rtol=1e-10)

    def test_orthant_flow_matches_independent_absolute_harmonic_solution(self):
        rng=torch.Generator().manual_seed(819)
        q=torch.rand(12,4,dtype=torch.float64,generator=rng)+.01
        p=2*torch.randn(q.shape,dtype=q.dtype,generator=rng)
        normal=torch.eye(4,dtype=q.dtype).expand(12,-1,-1);offset=torch.zeros(12,4,dtype=q.dtype)
        duration=3.7
        actual,_,bad,_=harmonic_reflections(q,p,normal,offset,duration)
        expected=(q*math.cos(duration)+p*math.sin(duration)).abs()
        self.assertFalse(bool(bad.any()))
        torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-12)

    def test_logistic_auxiliary_and_metropolis_residual_match_independent_quadrature(self):
        rng=torch.Generator().manual_seed(821);dtype=torch.float64
        q=torch.tensor([[-1.],[-.3],[.5],[1.2]],dtype=dtype)
        normal=torch.ones(4,1,1,dtype=dtype);offset=torch.zeros(4,1,dtype=dtype)
        mean=.25;sd=.75;scale=.15
        def residual(x):
            value=-.5*((x[:,0]-mean)/sd).square()+.5*x[:,0].square()
            gradient=-(x-mean)/sd**2+x
            return value,gradient,torch.zeros(4,dtype=torch.bool)
        saved=[];accepted=0
        for i in range(4300):
            q,info=state_transition(q,residual,normal,offset,scale,.45,4,rng)
            self.assertFalse(bool(info['invalid'].any()))
            accepted+=int(info['accepted'].sum())
            if i>=300:saved.append(q.clone())
        sample=torch.stack(saved).numpy().reshape(-1)
        x=np.linspace(-6,6,60001);density=np.exp(-.5*((x-mean)/sd)**2)/(1+np.exp(-x/scale))
        density/=np.trapezoid(density,x)
        expected_mean=np.trapezoid(x*density,x);variance=np.trapezoid((x-expected_mean)**2*density,x)
        self.assertLess(abs(sample.mean()-expected_mean),.035)
        self.assertLess(abs(sample.var()-variance),.035)
        cdf=np.r_[0,np.cumsum((density[1:]+density[:-1])*.5*np.diff(x))]
        expected=np.interp([.05,.5,.95],cdf,x)
        np.testing.assert_allclose(np.quantile(sample,[.05,.5,.95]),expected,atol=.045,rtol=0)
        self.assertGreater(accepted/(4300*4),.8)
