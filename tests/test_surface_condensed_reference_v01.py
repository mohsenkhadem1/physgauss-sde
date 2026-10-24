import numpy as np
from scipy import sparse
from experiments.campaign_surface_condensed_reference_v01 import CondensedStaticReference


def test_exact_interior_elimination_with_interior_forces_and_queries():
    rng=np.random.default_rng(4262);R=rng.normal(size=(15,15));A=R.T@R+np.eye(15)
    C=rng.normal(size=(5,5));B=np.zeros((15,15));B[:5,:5]=C.T@C
    f=rng.normal(size=(15,3));P=rng.normal(size=(6,15));model=CondensedStaticReference(A,B,f,P)
    theta=np.array([[2.,3.],[10.,.1],[.3,11.]])
    expected=np.stack([P@np.linalg.solve(E*A+K*B,f)for E,K in theta])
    np.testing.assert_allclose(model(theta),expected,rtol=2e-12,atol=1e-13)
    assert model.full_free_DOF==15 and len(model.boundary)==5
    assert model.diagnostics['exact_particular_1_over_E_term']
    assert not model.diagnostics['interior_basis_truncated']


def test_P2_all_surface_loads_and_queries_against_full_sparse_operator():
    from experiments import campaign_surface_p2_static_v01 as p2
    from experiments.campaign_surface_force_study_v01 import observation_points
    from experiments.campaign_surface_p2_study_design_v01 import load_catalog
    forward=p2.SparseStaticForward(2);loads=np.stack(list(load_catalog(2).values()))*1e-4
    P=p2.point_query_matrix(forward,observation_points())
    model=CondensedStaticReference(forward.Kunit_free,forward.W_free,loads.reshape(10,-1).T[forward.free],P)
    theta=np.array([[2000.,5.],[8000.,30.],[4300.,17.]])
    expected=p2.field_queries(2,forward.staticforward(theta,loads),observation_points()).transpose(0,2,3,1).reshape(3,24,10)
    np.testing.assert_allclose(model(theta),expected,rtol=2e-9,atol=2e-17)
    assert model.full_free_DOF==1125 and model.modes==720
    assert model.diagnostics['interior_force_exact_zero'] and model.diagnostics['interior_query_exact_zero']
    assert not model.diagnostics['exact_particular_1_over_E_term']
