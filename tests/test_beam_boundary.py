import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from beam_boundary import (assemble, candidates, response, ExactBoundaryPosterior,
    observation_design, simulation_audit, mixture_quantile, mixture_cdf)


class BeamBoundaryTests(unittest.TestCase):
    def test_mechanics_refinement_and_balance(self):
        report=simulation_audit(candidates())
        self.assertTrue(report['pass'],report)

    def test_free_beam_rigid_modes_and_energy(self):
        ke,ks,f=assemble(candidates()[0],20)
        translation=np.tile([1.,0.],21)
        rotation=np.empty(42);rotation[0::2]=np.linspace(0,1,21);rotation[1::2]=1
        self.assertLess(np.linalg.norm(ke@translation),1e-8)
        self.assertLess(np.linalg.norm(ke@rotation),1e-8)
        self.assertGreater(np.linalg.eigvalsh(ke+ks).min(),0)
        u=np.linalg.solve(ke+ks,f)
        # Stationarity of independently finite-differenced total potential.
        rng=np.random.default_rng(8);v=rng.normal(size=len(u));eps=1e-4
        energy=lambda q:.5*q@(ke+ks)@q-q@f
        self.assertLess(abs((energy(u+eps*v)-energy(u-eps*v))/(2*eps)),1e-6)

    def test_uniform_cantilever_matches_closed_form(self):
        elements=20;h=1/elements
        ke,_,_=assemble(candidates()[0],elements)
        load=np.zeros(2*(elements+1))
        for e in range(elements):load[2*e:2*e+4]+=np.array([h/2,h*h/12,h/2,-h*h/12])
        u=np.zeros_like(load);u[2:]=np.linalg.solve(ke[2:,2:],load[2:])
        x=np.linspace(0,1,elements+1)
        expected=x*x*(6-4*x+x*x)/(24*.8)
        np.testing.assert_allclose(u[0::2],expected,rtol=1e-8,atol=1e-10)

    def test_dense_conditioning_matches_marginalized_solver(self):
        engine=ExactBoundaryPosterior(candidates()[:3],observation_design('sparse'))
        y=np.random.default_rng(12).normal(size=len(engine.design))
        weights,means=engine.fit(y)
        scores=[]
        for i,h in enumerate(engine.h):
            s=h@engine.prior@h.T+engine.sigma**2*np.eye(len(y))
            expected=engine.prior@h.T@np.linalg.solve(s,y)
            cov=engine.prior-engine.prior@h.T@np.linalg.solve(s,h@engine.prior)
            np.testing.assert_allclose(means[0,i],expected,rtol=1e-9,atol=1e-11)
            np.testing.assert_allclose(engine.cov[i],cov,rtol=1e-8,atol=1e-12)
            scores.append(-.5*(y@np.linalg.solve(s,y)+np.linalg.slogdet(s)[1]))
        scores=np.exp(np.array(scores)-max(scores));scores/=scores.sum()
        np.testing.assert_allclose(weights[0],scores,rtol=1e-9,atol=1e-10)

    def test_mixture_quantiles_include_between_boundary_uncertainty(self):
        weights=np.array([[.5,.5]]); means=np.array([[-4.,4.]]);var=np.array([[.01,.01]])
        lo=mixture_quantile(.05,weights,means,var);hi=mixture_quantile(.95,weights,means,var)
        self.assertLess(lo[0],-4);self.assertGreater(hi[0],4)
        np.testing.assert_allclose(mixture_cdf(lo,weights,means,var),.05,atol=1e-12)

    def test_support_and_material_are_distinct_parameters(self):
        b=candidates()[0];x=np.linspace(0,1,21)
        self.assertGreater(np.linalg.norm(response(b,x)-response(b,x,rigidity=(1.6,1.6,1.6))),.01)


if __name__=='__main__':unittest.main()
