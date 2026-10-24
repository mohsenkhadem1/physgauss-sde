"""Controls use synthetic parameters and observations only, no posterior reads."""
import numpy as np
import pytest
from experiments import campaign_compliant_force_challenge_v01 as challenge
from experiments import campaign_compliant_patch_inverse as original


def test_arbitrary_force_scalar_batch_and_independent_first_order_recurrence():
    forward=challenge.ArbitraryForceForward()
    theta=np.array([[4300,.029,.031,16.5],[2100,.011,.019,6.]])
    forces=challenge.force_programs()
    for force in forces.values():
        out=forward(theta,force)
        for i in range(len(theta)):
            np.testing.assert_allclose(out[i],challenge.independent_numpy_forward(theta[i],force),atol=5e-14,rtol=2e-11)
        np.testing.assert_array_equal(out[:,:25],forward(theta,'A')[:,:25])
    stacked=np.stack(list(forces.values())[:2]);out=forward(theta,stacked)
    for i in range(2):np.testing.assert_array_equal(out[i],forward(theta[i:i+1],stacked[i])[0])
    np.testing.assert_allclose(forward(theta,original.known_force('A')),forward(theta,'A'),atol=5e-15,rtol=2e-12)


def test_force_budget_causal_prefix_and_changed_future():
    original_a=original.known_force('A')
    for force in challenge.force_programs().values():
        np.testing.assert_array_equal(force[:24],original_a[:24]);assert not np.array_equal(force[24:],original_a[24:])
    pokes=list(challenge.poke_programs().values())
    amplitudes=[np.linalg.norm(f.sum(1),axis=1)for f in pokes]
    for a in amplitudes[1:]:np.testing.assert_allclose(a,amplitudes[0],atol=1e-17,rtol=2e-15)


def test_gaussian_target_variance_analytic_and_no_truth_selection():
    x=np.array([-1.,1.,-1.,1.]);target=x[:,None];pokes=np.stack([x[:,None],np.zeros((4,1))])
    scores=challenge.gaussian_target_reduction(target,pokes,sigma=1)
    np.testing.assert_allclose(scores,[.5,0.],atol=1e-15)
    rng=np.random.default_rng(17);unit=rng.normal(size=(100,4));pokes=rng.normal(size=(3,100,2));target=rng.normal(size=(100,7))
    first,_=challenge.choose_actions(target,pokes,unit,sigma=1)
    second,_=challenge.choose_actions(target,pokes,unit,sigma=1)
    assert first==second


def test_exact_weights_and_positive_centered_moments():
    x=np.array([[0.],[1.],[2.]])
    w,check=challenge.exact_reweight(x,np.array([1.]),sigma=1)
    expected=np.exp(-.5*(x[:,0]-1)**2);expected/=expected.sum()
    np.testing.assert_allclose(w,expected,atol=1e-15)
    mean,var=challenge.weighted_prediction(1e8+x,w)
    np.testing.assert_allclose(mean,1e8+1);np.testing.assert_allclose(var,[2*expected[0]],atol=1e-15)
    assert not check['importance_gate_passed']
    for invalid in [np.zeros((59,5,3)),np.full((60,5,3),np.nan)]:
        with pytest.raises(ValueError):challenge.ArbitraryForceForward()(original.TRUTH[None],invalid)
