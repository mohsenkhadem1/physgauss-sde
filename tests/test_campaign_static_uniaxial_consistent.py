import numpy as np
import pytest
from experiments.campaign_static_uniaxial_consistent import ConsistentPrismaticBar
from experiments.campaign_static_uniaxial_bar import SIZE


def test_exact_consistent_triangle_spring_integrates_affine_square_not_lumped():
    model=ConsistentPrismaticBar();u=np.zeros_like(model.rest);a,b,c=.004,.05,-.03
    u[:,0]=a+b*model.reference_local[:,1]+c*model.reference_local[:,2]
    energy=.5*u.ravel()@(model.spring_matrix@u.ravel())
    exact=.5*model.spring*(a*a+b*b*SIZE[1]**2/12+c*c*SIZE[2]**2/12)
    assert energy==pytest.approx(exact,abs=2e-18)
    assert abs(.5*np.sum(model.Kdiagonal*u**2)-exact)>1e-5
    np.testing.assert_allclose(np.asarray(model.spring_matrix.sum(1)).ravel(),model.Kdiagonal.ravel(),atol=2e-14,rtol=1e-14)
    assert model.spring_matrix.sum()==pytest.approx(model.spring)


def test_consistent_spring_full_static_gradient_and_tangent_finite_differences():
    model=ConsistentPrismaticBar(force=3.);u,_=model.oracle_field();rng=np.random.default_rng(2026121201)
    u+=rng.normal(size=u.shape)*1e-4;direction=rng.normal(size=u.shape);direction/=np.linalg.norm(direction);eps=2e-7
    state=model.evaluate(u,True);plus=model.evaluate(u+eps*direction);minus=model.evaluate(u-eps*direction)
    assert (plus['potential_J']-minus['potential_J'])/(2*eps)==pytest.approx(np.sum(state['residual']*direction),abs=2e-10,rel=2e-7)
    np.testing.assert_allclose((state['tangent']@direction.ravel()).reshape(u.shape),(plus['residual']-minus['residual'])/(2*eps),atol=3e-8,rtol=2e-7)


def test_consistent_oracle_has_zero_residual_and_same_finite_spring_energy():
    model=ConsistentPrismaticBar(force=3.);u,exact=model.oracle_field();state=model.evaluate(u)
    assert np.linalg.norm(state['residual'])<2e-12
    assert state['spring_J']==pytest.approx(exact['spring_energy_J'],abs=2e-15)
    assert np.max(abs(u[model.mask]))==0
