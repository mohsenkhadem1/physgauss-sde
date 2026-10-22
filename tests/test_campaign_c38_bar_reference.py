import numpy as np
from experiments.campaign_c38_bar_reference import P2SpringBlock,oracle_positions,material_quadrature,prior_physical,endpoint_displacements
from experiments.campaign_static_uniaxial_bar import analytic_solution,SIZE


def test_p2_uniform_solution_full_constitutive_energy_and_force():
    for E,K,F in ((4500.,100.,1.),(9000.,400.,2.),(6000.,220.,2.)):
        model=P2SpringBlock(young=E,spring=K,force=F);d=model.verify_oracle()
        assert d['original_free_residual_N']<2e-11
        np.testing.assert_allclose(d['spring_force_N'],[F,0,0],rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(d['external_force_N'],[F,0,0],rtol=1e-13,atol=1e-13)
        assert d['elastic_energy_error_J']<1e-13 and d['spring_energy_error_J']<1e-15 and d['minimum_J']>.9
    # Independent sparse P2 Newton starts at rest, not the analytical answer.
    answer=model.solve(force_tolerance_N=1e-10)
    expected=oracle_positions(np.array([[E,K]]),F,model.rest)[0]
    np.testing.assert_allclose(answer['positions'],expected,atol=2e-11,rtol=0)


def test_oracle_brent_zero_prior_boundary_and_positive_material_quadrature():
    theta=prior_physical(np.array([[0.,0.],[0,1.],[1.,0],[1.,1.],[.5,.5]]));X=material_quadrature()
    assert X.shape==(8,3)
    np.testing.assert_allclose(oracle_positions(theta,0.,X),np.broadcast_to(X,(len(theta),*X.shape)),atol=1e-16)
    for E,K in theta:
        for F in (1.,1.5,2.):
            a=analytic_solution(F,young=E,spring=K);p=endpoint_displacements(np.array([[E,K]]),[F])[0,0]
            np.testing.assert_allclose(p,[a['left_translation_m'],a['right_translation_m']],rtol=0,atol=2e-14)
    # Integrates uniform volume moments, not a hand-picked visible subset.
    np.testing.assert_allclose(X.mean(0),[SIZE[0]/2,0,0],atol=1e-17)
    np.testing.assert_allclose((X-X.mean(0)).T@(X-X.mean(0))/len(X),np.diag(np.array(SIZE)**2/12),atol=1e-17)
