"""Independent finite-bank Gaussian controls; no physics/data/RNG calls."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from experiments import campaign_layout_mixture_decision_v01 as m

def bank():return m.PredictionBank([.2,.3,.5],[0,0,1],[[-.8,.1],[.2,.1],[.9,.1]],[[-2.,1.],[.5,1.],[2.,-1.]])
def test_exact_category_evidence_normalization_and_layout_variance():
    b=m.combine_layouts([np.log(2),np.log(3)],[.5,.5],[[.4,.6],[1.]],[np.array([[-.8],[.2]]),np.array([[.9]])],[np.array([[-2.,1.],[.5,1.]]),np.array([[2.,-1.]])])
    np.testing.assert_allclose(b.probability,[.16,.24,.6]);v=m.variance_decomposition(b);assert v['between_layout']>0
    assert v['total']==pytest.approx(v['within_layout']+v['between_layout'],abs=1e-14)
    np.testing.assert_allclose(m.layout_probabilities([1001.,1000.],[.5,.5]),m.layout_probabilities([1.,0.],[.5,.5]))
def test_streamed_centered_conditional_risk_equals_direct_dense_and_large_offset():
    b=bank();ys=np.array([-.5,.4,1.2]);lp,v=m.outcome_risk(b,0,ys,.3,latent_chunk=1)
    for j,y in enumerate(ys):
        logs=np.log(b.probability)-.5*((b.candidate[:,0]-y)/.3)**2-np.log(.3*np.sqrt(2*np.pi));q=np.exp(logs-logsumexp(logs));mu=q@b.target
        assert lp[j]==pytest.approx(logsumexp(logs),abs=1e-14);assert v[j]==pytest.approx(np.mean(q@((b.target-mu)**2)),abs=1e-14)
    shifted=m.PredictionBank(b.probability,b.layout,b.candidate,b.target+1e7);_,w=m.outcome_risk(shifted,0,ys,.3,latent_chunk=2);np.testing.assert_allclose(w,v,rtol=1e-9,atol=1e-9)
def test_positive_outcome_GL_matches_independent_adaptive_and_generative_GH():
    b=bank();sigma=.4;got=m.expected_risk(b,0,sigma,order=12)
    def fn(y):
        logs=np.log(b.probability)-.5*((b.candidate[:,0]-y)/sigma)**2-np.log(sigma*np.sqrt(2*np.pi));q=np.exp(logs-logsumexp(logs));mu=q@b.target;return np.exp(logsumexp(logs))*np.mean(q@((b.target-mu)**2))
    reference=quad(fn,-8,8,epsabs=1e-11,epsrel=1e-11)[0];assert got['expected_target_variance']==pytest.approx(reference,abs=1e-10)
    nodes,weights=np.polynomial.hermite.hermgauss(128);generative=0.
    for i,p in enumerate(b.probability):
        ys=b.candidate[i,0]+sigma*np.sqrt(2)*nodes;v=[]
        for y in ys:
            logs=np.log(b.probability)-.5*((b.candidate[:,0]-y)/sigma)**2;posterior=np.exp(logs-logsumexp(logs));mu=posterior@b.target;v.append(np.mean(posterior@((b.target-mu)**2)))
        generative+=p*np.dot(weights,v)/np.sqrt(np.pi)
    assert got['expected_target_variance']==pytest.approx(generative,abs=1e-8);assert abs(1-got['predictive_mass'])<=got['omitted_mass_bound']+1e-12
    assert 0<=got['expected_target_variance']<got['prior_target_variance']
def test_uninformative_probe_has_no_value_but_informative_layout_probe_does():
    b=bank();a=m.expected_risk(b,0,.25);z=m.expected_risk(b,1,.25)
    assert z['variance_reduction']==pytest.approx(0,abs=1e-12);assert a['variance_reduction']>.5
    result=m.screen_actions(b,.25,orders=(8,16));assert result['qualified']and result['minimum_risk_action']==0

def test_deterministic_action_ranking_crossover_without_truth_argument():
    # Two distinct prefix posterior states on the SAME three-category mechanism.
    targets=np.array([[-1.],[0.],[1.]]);actions=np.array([[1.,0.],[0.,0.],[0.,1.]])
    left=m.PredictionBank([.49,.49,.02],[0,1,2],actions,targets);right=m.PredictionBank([.02,.49,.49],[0,1,2],actions,targets)
    l=m.screen_actions(left,.2,orders=(8,16));r=m.screen_actions(right,.2,orders=(8,16))
    assert l['qualified']and r['qualified']and l['minimum_risk_action']==0 and r['minimum_risk_action']==1

def test_zero_variance_and_exact_zero_probability_are_not_dropped_prior_mass():
    b=m.PredictionBank([1.,0.],[0,1],[[0.],[100.]],[[3.,4.],[3.,4.]])
    got=m.expected_risk(b,0,1.);assert got['expected_target_variance']==0 and got['omitted_risk_bound']==0
    with pytest.raises(ValueError):m.layout_probabilities([0.,0.],[.2,.2])
    with pytest.raises(ValueError):m.PredictionBank([1.],[0],[[np.nan]],[[1.]])


def test_numeric_ties_use_predeclared_order_without_claiming_unique_winner():
    b=m.PredictionBank([.5,.5],[0,1],[[-1.,-1.],[1.,1.]],[[-1.],[1.]])
    result=m.screen_actions(b,.3,orders=(8,16),tie_order=[1,0])
    assert result['qualified']and result['unique_best_action']is None and result['selected_action']==1
    assert result['numerically_indistinguishable_actions']==[0,1]
    with pytest.raises(ValueError,match='tie-order'):m.screen_actions(b,.3,tie_order=[0,0])


def test_outcome_budget_refuses_before_large_allocation(monkeypatch):
    b=bank();monkeypatch.setattr(np.polynomial.legendre,'leggauss',lambda *a:pytest.fail('No allocation after budget refusal'))
    with pytest.raises(MemoryError,match='node budget'):m.expected_risk(b,0,1e-10,maximum_outcome_nodes=1000)


def test_overflowing_positive_weight_sum_fails_closed():
    with pytest.raises(ValueError,match='Finite positive total'):m.PredictionBank([1e308,1e308],[0,1],[[0.],[1.]],[[0.],[1.]])
    with pytest.raises(ValueError,match='Finite positive total'):m.combine_layouts([0.],[1.],[[1e308,1e308]],[np.array([[0.],[1.]])],[np.array([[0.],[1.]])])
