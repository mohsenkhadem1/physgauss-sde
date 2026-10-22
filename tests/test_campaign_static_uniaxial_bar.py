import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from experiments.campaign_static_uniaxial_bar import PrismaticBar,analytic_solution,SIZE


def test_independent_finite_uniaxial_oracle_lateral_free_force_and_spring():
    for force in (1.,3.):
        exact=analytic_solution(force)
        assert .02<exact['stretch']-1<.08
        assert exact['lateral_stretch']<1 and exact['J']>0
        assert abs(exact['lateral_Piola_Pa'])<1e-10
        assert exact['first_Piola_axial_Pa']*SIZE[1]*SIZE[2]==pytest.approx(force,abs=2e-13)
        assert exact['left_translation_m']==force/200


def test_static_rest_exact_constraint_elimination_and_no_inertia():
    model=PrismaticBar(force=0.)
    answer=model.solve()
    np.testing.assert_array_equal(answer['displacement_local_m'],np.zeros_like(model.rest))
    assert answer['iterations']==0 and answer['potential_J']==0
    assert np.all(model.Kdiagonal[:,1:]==0) and not hasattr(model,'mass')
    assert model.left_weights.sum()==pytest.approx(1.) and model.right_weights.sum()==pytest.approx(1.)


def test_sparse_static_energy_gradient_and_tangent_independent_differences():
    model=PrismaticBar(force=1.);u,exact=model.oracle_field()
    rng=np.random.default_rng(2026121200);u+=rng.normal(size=u.shape)*1e-4
    direction=rng.normal(size=u.shape);direction/=np.linalg.norm(direction)
    state=model.evaluate(u,True);eps=2e-7
    plus=model.evaluate(u+eps*direction);minus=model.evaluate(u-eps*direction)
    gradient=(plus['potential_J']-minus['potential_J'])/(2*eps)
    assert gradient==pytest.approx(np.sum(state['residual']*direction),abs=2e-10,rel=2e-7)
    Hd=(state['tangent']@direction.ravel()).reshape(u.shape)
    np.testing.assert_allclose(Hd,(plus['residual']-minus['residual'])/(2*eps),atol=3e-8,rtol=2e-7)
    np.testing.assert_allclose(state['tangent'].toarray(),state['tangent'].toarray().T,atol=2e-12,rtol=0)


def test_passive_finite_frame_equivariance_including_load_fixture_constraints():
    rotation=Rotation.from_rotvec([.41,-.23,.37]).as_matrix();base=PrismaticBar(force=3.)
    frame=PrismaticBar(force=3.,rotation=rotation,translation=np.array([.4,-.2,.3]))
    u,_=base.oracle_field();a=base.evaluate(u,True);b=frame.evaluate(u,True)
    assert a['potential_J']==pytest.approx(b['potential_J'],abs=2e-15)
    np.testing.assert_allclose(a['residual'],b['residual'],atol=1e-13,rtol=1e-10)
    np.testing.assert_allclose(a['tangent'].toarray(),b['tangent'].toarray(),atol=2e-12,rtol=1e-10)
    np.testing.assert_allclose(b['positions'],a['positions']@rotation.T+frame.translation,atol=1e-16)
    np.testing.assert_array_equal(base.mask,frame.mask)


def test_oracle_stress_energy_and_nodal_residual_independent_of_sparse_solve():
    model=PrismaticBar(force=3.);u,oracle=model.oracle_field();state=model.evaluate(u)
    F,P=model.independent_stress(state['positions'])
    np.testing.assert_allclose(P,np.broadcast_to(np.diag([oracle['first_Piola_axial_Pa'],0.,0.]),P.shape),atol=3e-11,rtol=1e-12)
    assert state['elastic_J']==pytest.approx(oracle['elastic_energy_J'],abs=2e-15)
    assert state['spring_J']==pytest.approx(oracle['spring_energy_J'],abs=2e-15)
    assert np.linalg.norm(state['residual'])<2e-12


def test_inversion_and_incompatible_exact_initial_constraints_fail():
    model=PrismaticBar();u=np.zeros_like(model.rest);u[:,0]=-2*model.reference_local[:,0]
    with pytest.raises(ValueError,match='J'):model.evaluate(u)
    u[:]=0;u[model.mask]=1e-3
    with pytest.raises(ValueError,match='symmetry'):model.solve(u)
