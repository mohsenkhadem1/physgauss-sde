import numpy as np
from scipy.stats import truncnorm
from scripts.phystwin.uncertainty_posterior import (
    Posterior, conditional_components, conformal_scale, discrepancy,
    marginal_quantiles, Z90,
)


def test_trapezoidal_quantiles_uniform_density():
    axis = np.linspace(-2, 4, 65)
    mass = np.ones((1, 65)); mass[:, [0, -1]] = .5; mass /= mass.sum()
    assert np.allclose(marginal_quantiles(mass, axis), [[-1.7, 1., 3.7]], atol=1e-12)


def test_joint_discrepancy_conditioning_matches_gaussian_formula():
    fields = np.array([[1., 3.], [2., 5.]])
    C = np.array([[4., 1.], [1., 2.]])
    fobs, adjusted, gain, var, chol = conditional_components(fields, np.array([.5, -.2]), C, [0], [1], 1.)
    assert np.allclose(gain, [[.2]])
    assert np.allclose(var, [1.8])
    assert np.allclose(chol, [[np.sqrt(5.)]])
    y=2.
    assert np.allclose(adjusted + gain*y, fields[:, 1:]-.2 + .2*(y-fobs))


def test_quadrature_matches_independent_truncated_normal_solution():
    axis=np.linspace(0, 1, 129)
    nodes=np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1,2)
    w=np.ones((129,129));w[[0,-1],:]*=.5;w[:,[0,-1]]*=.5;w=w.ravel()/w.sum()
    fields=np.c_[nodes[:,0], nodes[:,0]+nodes[:,1]]
    p=Posterior(fields,nodes,w,np.zeros(2),np.zeros((2,2)),[0],[1],.15,'cpu')
    result=p.infer(np.array([[.35]]))
    mean,var=truncnorm.stats((0-.35)/.15,(1-.35)/.15,loc=.35,scale=.15,moments='mv')
    assert abs(result['mean'][0,0]-(mean+.5))<2e-5
    assert abs(result['sd'][0,0]-np.sqrt(var+1/12))<2e-5


def test_discrepancy_covariance_positive_and_floor_retained():
    residuals=np.arange(24).reshape(8,3)*1e-6
    bias,C=discrepancy(residuals)
    assert np.allclose(bias,residuals.mean(0))
    assert np.linalg.eigvalsh(C).min()>=1e-12*(1-1e-10)


def test_conformal_rank_uses_objects_and_all_budgets():
    truth=np.zeros((80,2));sd=np.ones_like(truth)/Z90
    error=np.repeat(np.arange(1,21),4)[:,None]*np.ones((1,2))
    predictions=[dict(mean=error,sd=sd) for _ in range(3)]
    scale,scores=conformal_scale(predictions,truth,4)
    assert len(scores)==20 and np.isclose(scale,19)
