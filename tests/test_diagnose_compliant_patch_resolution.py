"""Analytic checks for the bounded secondary NUMERICAL diagnosis only."""
import numpy as np
import pytest
from experiments import diagnose_compliant_patch_resolution as diagnosis


def test_nested_positive_cc_integrates_uniform_prior_and_polynomials():
    previous=None
    for order in diagnosis.ORDERS:
        x,w=diagnosis.cc_nodes_weights(order)
        assert np.all(w>0) and w.sum()==pytest.approx(1.,abs=1e-15)
        for degree in (0,1,2,3,4):
            exact=0. if degree%2 else 1/(degree+1)
            assert w@x**degree==pytest.approx(exact,abs=1e-15)
        if previous is not None:
            np.testing.assert_array_equal(x[::2],previous)
        previous=x
    theta,mass=diagnosis.grid(9)
    assert mass.sum()==pytest.approx(1.,abs=1e-15) and len(theta)==9**4
    assert mass@theta[:,0]==pytest.approx(6000/np.log(4),rel=1e-11)
    with pytest.raises(ValueError):diagnosis.cc_nodes_weights(6)


def test_sup_cdf_and_step_quantiles_detect_unresolved_concentration():
    fine=dict(values=[0.,1.,2.],cdf=[.2,.6,1.])
    coarse=dict(values=[1.],cdf=[1.])
    assert diagnosis.cdf_distance(coarse,fine)==pytest.approx(.4)
    theta=np.array([[2000.,.005,.018,5.],[4000.,.035,.034,17.5],[8000.,.065,.05,30.]])
    marginal=diagnosis.marginals(theta,np.array([.01,.98,.01]))[0]
    assert marginal['quantiles_05_50_95']==[4000.,4000.,4000.]
    assert marginal['grid_nodes_in_90_percent_interval']==1


def test_predictive_agreement_cannot_override_unresolved_material_cdf():
    theta=np.array([[2000.,.005,.018,5.],[4000.,.035,.034,17.5],[8000.,.065,.05,30.]])
    moments={k:(np.zeros((61,5,3)),np.full((61,5,3),1e-4)) for k in ('A','B')}
    a=dict(order=5,material_marginals=diagnosis.marginals(theta,np.array([.01,.98,.01])),_moments=moments)
    b=dict(a,order=9)
    result=diagnosis.compare(a,b)
    assert all(x['passed'] for x in result['parameters'])
    assert all(x['passed'] for x in result['targets'].values())
    assert not result['E_interval_spans_at_least3_nodes'] and not result['passed']


def test_constant_likelihood_keeps_nonuniform_cc_prior_weights():
    theta,mass=diagnosis.grid(5)
    post,_=diagnosis.original.condition(np.zeros((len(theta),1)),np.zeros(1),mass)
    np.testing.assert_allclose(post,mass,atol=2e-16,rtol=1e-14)
    assert np.ptp(post)>0
