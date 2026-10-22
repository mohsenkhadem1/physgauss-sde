import json
import numpy as np
import pytest
from scipy.integrate import quad,cubature
from scipy.special import logsumexp
from experiments import campaign_c38_beta_importance as m


def parameters(a,b):
    return dict(logits=np.zeros((1,16)),alpha=np.broadcast_to(np.asarray(a),(1,16,2)),beta=np.broadcast_to(np.asarray(b),(1,16,2)))


def test_normalized_mixture_proposal_and_nested_sobol_prefix():
    pars=parameters([3.,6.],[5.,4.]);result=cubature(lambda u:np.exp(m.proposal_log_density(u,pars)),np.zeros(2),np.ones(2),rule='gk21',rtol=1e-9)
    assert result.status=='converged' and abs(result.estimate-1)<1e-9
    large=m.sobol_points(271,power=12);small=m.sobol_points(271,power=10);np.testing.assert_array_equal(large[:1024],small)
    draw=m.sample_proposal(large,pars);np.testing.assert_allclose(draw.mean(0),.5*.5+.5*np.array([3/8,6/10]),atol=3e-4)
    uniform=parameters([1.,1.],[1.,1.]);np.testing.assert_array_equal(m.sample_proposal(large,uniform),large[:,2:])


def test_exact_target_reweighting_ignores_neural_posterior_as_prior():
    pars=parameters([3.,6.],[5.,4.]);draw=m.sample_proposal(m.sobol_points(456,power=14),pars)
    # An uninformative likelihood must recover the UNIFORM physical prior,
    # rather than retain the concentrated neural proposal as a changed prior.
    lw=-m.proposal_log_density(draw,pars);q=np.tile(draw[:,0,None],(1,24));s=m.summarize_cloud(draw,lw,q,np.array([.25,.5,.75]))
    assert abs(s['log_evidence'])<3e-4 and s['ESS']>8000
    np.testing.assert_allclose(s['CDF_grid'],[[.25,.5,.75]]*2,atol=8e-4)
    np.testing.assert_allclose(s['field_mean']-m.material_quadrature(),.5,atol=3e-4)


def test_real_likelihood_evidence_and_CDF_against_independent_scalar_integration():
    obs=np.array([.005]);u=m.sample_proposal(m.sobol_points(789,power=14));lw,q=m.evaluate_cloud(u,obs,[1.],[0]);s=m.summarize_cloud(u,lw,q,[.25,.5,.75])
    def density(k):return float(np.exp(-.5*((np.exp(-m.LOWER[1]-k*(m.UPPER[1]-m.LOWER[1]))-.005)/m.SIGMA_M)**2))
    Z=quad(density,0,1,epsabs=1e-12,epsrel=1e-10)[0];expected=np.array([quad(density,0,v,epsabs=1e-12,epsrel=1e-10)[0]/Z for v in (.25,.5,.75)])
    assert abs(np.expm1(s['log_evidence']-np.log(Z)))<2e-5
    np.testing.assert_allclose(s['CDF_grid'][1],expected,atol=3e-4)
    # Left-only acquisition leaves E exactly at its independent prior.
    np.testing.assert_allclose(s['CDF_grid'][0],[.25,.5,.75],atol=3e-4)
    # Both raw readings enter once; this is not a posterior-times-likelihood.
    value=m.exact_log_likelihood(u[:10],np.array([.005,.005]),np.ones(2),np.zeros(2,dtype=int))
    np.testing.assert_allclose(value,2*m.exact_log_likelihood(u[:10],obs,[1.],[0]),atol=1e-12)


def test_inverse_cdf_precision_failure_preserved_and_TEST_role_forbidden(tmp_path):
    pars=parameters([1e-8,1.],[1.,1.]);u=np.array([[.9,.2,.1,.5]])
    with pytest.raises(FloatingPointError,match='no clipping'):m.sample_proposal(u,pars)
    p=dict(schema='c38-beta-fixed-importance-v01',budgets=list(m.BUDGETS),scramble_seeds=list(m.SCRAMBLES),fit_seeds=list(m.frozen.FIT_SEEDS),object_indices=list(range(16)),comparison=m.CRITERIA,roles={'TEST':{}},source_sha256={n:'0'*64 for n in m.source_names()})
    path=tmp_path/'protocol.json';path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='Exact prospective'):m.validate(path,m.frozen.sha(path))
