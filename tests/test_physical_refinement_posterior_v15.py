import numpy as np
from scripts.physical_refinement_v15.belief import Query,observe_once
from scripts.physical_refinement_v15.posterior import Likelihood,posterior


def test_evidence_update_matches_sequential_conditioning_with_correlated_noise():
    q=Query('q','rgbd',np.array([[0.,0.],[1.,2.],[3.,4.]]),np.array([[1.,.4],[.4,2.]]))
    weights=np.array([.2,.3,.5]);observations=[np.array([.5,1.]),np.array([1.,1.5])]
    likelihood=Likelihood(q)
    actual,z=posterior(weights,[likelihood.log_density(y) for y in observations])
    expected=weights
    for y in observations:expected,_=observe_once(expected,q,y)
    np.testing.assert_allclose(actual,expected,rtol=1e-12)
    assert np.isfinite(z)


def test_empty_observation_preserves_nonuniform_quadrature_prior():
    prior=np.array([.1,.3,.6])
    actual,z=posterior(prior,[])
    np.testing.assert_allclose(actual,prior)
    assert abs(z)<1e-14
