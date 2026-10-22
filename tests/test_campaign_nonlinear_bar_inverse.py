import numpy as np
import pytest
from experiments import campaign_nonlinear_bar_inverse as problem


def test_vectorized_nonlinear_oracle_vs_independent_nested_brent_prior_and_boundaries():
    rng=np.random.default_rng(2026121298)
    unit=np.vstack([rng.uniform(size=(64,2)),[[0.,0.],[1.,1.],[0.,1.],[1.,0.]]])
    theta=problem.physical(unit)
    np.testing.assert_allclose(problem.predict(theta,(0.,1.,2.,3.)),problem.independent_predict(theta,(0.,1.,2.,3.)),atol=3e-14,rtol=2e-12)
    np.testing.assert_array_equal(problem.predict(theta,(0.,)),np.zeros((len(theta),2)))
    np.testing.assert_allclose(problem.predict(problem.TRUTH[None]),problem.independent_predict(problem.TRUTH[None]),atol=3e-14,rtol=2e-12)


def test_loguniform_prior_and_separate_leftshift_extension_identifiability():
    np.testing.assert_allclose(problem.physical([[0.,0.],[1.,1.]]),[[4500.,100.],[9000.,400.]],rtol=1e-14)
    theta=np.array([problem.TRUTH,problem.TRUTH*[1.01,1.],problem.TRUTH*[1.,1.01]])
    y=problem.predict(theta);np.testing.assert_array_equal(y[1,[0,2]],y[0,[0,2]])
    np.testing.assert_allclose((y[2]-y[0])[[0,2]],(y[2]-y[0])[[1,3]],atol=1e-17)
    jac=np.column_stack([(y[1]-y[0])/.01,(y[2]-y[0])/.01])
    assert np.linalg.matrix_rank(jac)==2 and np.linalg.svd(jac,compute_uv=False)[-1]>1e-3


def test_likelihood_uses_only_four_raw_channels_and_known_forces():
    unit=np.array([[.3,.6],[.5,.5]]);observed=problem.predict(problem.physical(unit[:1]))[0]
    likelihood=problem.log_likelihood(unit,observed)
    assert likelihood[0]==pytest.approx(-4*np.log(problem.SIGMA*np.sqrt(2*np.pi)))
    assert likelihood[1]<likelihood[0]
    with pytest.raises(ValueError):problem.log_likelihood(unit,np.zeros(6))
    with pytest.raises(ValueError):problem.physical([[-.1,.5]])


def test_finite_strain_response_differs_from_linear_tangent_at_truth():
    nonlinear=problem.predict(problem.TRUTH[None],(1.,2.,3.))[0]
    E,K=problem.TRUTH;linear=[]
    for force in (1.,2.,3.):
        left=force/K;linear.extend([left,left+problem.SIZE[0]*force/(E*problem.SIZE[1]*problem.SIZE[2])])
    np.testing.assert_array_equal(nonlinear[[0,2,4]],np.asarray(linear)[[0,2,4]])
    assert nonlinear[-1]-linear[-1]>problem.SIGMA


def test_preregistered_numeric_comparison_rejects_CDF_and_held_errors():
    from experiments.run_nonlinear_bar_inverse import compare,safe
    import json
    reference=dict(unit_CDF=np.array([[0.,.5,1.],[0.,.5,1.]]),unit_quantiles=np.array([[.3,.5,.7],[.3,.5,.7]]),held_mean_m=np.array([.01,.02]),held_std_m=np.array([.0001,.0002]))
    assert compare(reference,reference)['passed']
    changed={**reference,'unit_CDF':reference['unit_CDF']+.051}
    assert not compare(changed,reference)['passed']
    changed={**reference,'held_mean_m':reference['held_mean_m']+1e-5}
    assert not compare(changed,reference)['passed']
    assert json.loads(json.dumps(safe(compare(reference,reference))))['passed'] is True
