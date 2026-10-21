import inspect
import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss
from experiments.censored_range_observation import CensoredRanges,CensoredRangeModel
from experiments.campaign_point_active_views import TargetVariance,score_censored_views,select_point_active_view,update_after_view


def test_centered_moments_match_direct_variances_without_rest_origin_cancellation():
    targets=np.array([[0.,1.,2.],[2.,-1.,0.],[4.,3.,-2.]])+2**30
    weights=np.array([[.2,.3,.5],[1.,0.,0.],[0.,.5,.5]])
    m=TargetVariance(targets,feature_chunk_size=2)
    direct=[]
    for w in weights:
        relative=targets-targets[0];mu=w@relative
        direct.append(float(np.sum(w[:,None]*(relative-mu)**2)/targets.shape[1]))
    np.testing.assert_allclose(m.risks(weights),direct,atol=2e-15,rtol=1e-14)
    assert m._tensor is None and m.centered.shape==targets.shape


def test_gaussian_law_of_total_variance_with_proper_likelihood_quadrature():
    theta,prior_weights=hermegauss(64);prior_weights/=np.sqrt(2*np.pi)
    y,observation_weights=hermegauss(48);observation_weights/=np.sqrt(2*np.pi);y*=np.sqrt(2)
    model=CensoredRangeModel(np.eye(1),.01,background_std_m=2.,outlier_probability=0)
    predictions=CensoredRanges(np.ones((64,1),bool),theta[:,None]);posteriors=[]
    for value in y:
        w,_=model.condition(predictions,CensoredRanges([True],[value]),prior_weights);posteriors.append(w)
    m=TargetVariance(theta[:,None],prior_weights)
    posterior=np.array(posteriors);within=float(observation_weights@m.risks(posterior));means=posterior@theta
    between=float(observation_weights@(means**2)-(observation_weights@means)**2)
    assert within==pytest.approx(.5,abs=2e-10)
    assert between==pytest.approx(.5,abs=2e-10)
    assert within+between==pytest.approx(m.risks(prior_weights[None])[0],abs=2e-10)


def test_discrete_visibility_law_of_total_variance_no_fake_missingrange():
    # Background and foreground sameN(0,1); only visibility carries information.
    model=CensoredRangeModel(np.eye(1),.1,background_std_m=1.,outlier_probability=0)
    prediction=CensoredRanges([[False],[True]],[[np.nan],[0.]])
    result=score_censored_views(np.array([[0.],[2.]]),[prediction],model)
    assert result['prior_target_variance_m2']==1
    assert result['expected_target_variance_m2'][0]==pytest.approx(.36,abs=2e-15)
    w0,_=model.condition(prediction,CensoredRanges([False],[np.nan]))
    w1,_=model.condition(prediction,CensoredRanges([True],[0.]))
    means=np.array([w0@np.array([0.,2.]),w1@np.array([0.,2.])])
    assert .36+np.var(means)==pytest.approx(1.)


def test_uninformative_views_leave_risk_and_weights_unchanged_stable_tie():
    model=CensoredRangeModel(np.eye(2)*.01,.1,background_std_m=.4,outlier_probability=.02)
    prediction=CensoredRanges(np.ones((5,2),bool),np.zeros((5,2)))
    targets=np.arange(15.).reshape(5,3)/1000
    r=score_censored_views(targets,[prediction]*4,model)
    assert r['selected_index']==0 and r['fantasies']==32 and r['seed']==2026116000
    np.testing.assert_allclose(r['expected_target_variance_m2'],r['prior_target_variance_m2'],atol=1e-20)
    np.testing.assert_allclose(r['expected_effective_sample_size'],5,atol=1e-13)


def test_actual_update_uses_proper_censored_likelihood_and_keeps_all_particles():
    model=CensoredRangeModel(np.eye(36)*(.0002**2),.005,.05,0)
    visible=np.ones((2048,36),bool);visible[1024:,0]=False
    ranges=np.zeros((2048,36));ranges[:1024,0]=.0001
    prediction=CensoredRanges(visible,ranges);values=np.zeros(36);values[0]=.0003
    obs=CensoredRanges(np.ones(36,bool),values)
    weights,d=update_after_view(prediction,obs,model)
    hit=.995*np.exp(-.5*((.0003-.0001)/.0002)**2)/.0002
    miss=.005*np.exp(-.5*(.0003/.05)**2)/.05
    assert weights[:1024].sum()==pytest.approx(hit/(hit+miss),abs=1e-14)
    assert weights.shape==(2048,) and (weights>0).all() and weights.sum()==pytest.approx(1)
    assert d['effective_sample_size']==pytest.approx(1/(weights@weights))
    # A missing observation is not an artificial range value.
    observation_visible=np.ones(36,bool);observation_visible[0]=False
    first,_=update_after_view(prediction,CensoredRanges(observation_visible,np.zeros(36)),model)
    changed=np.zeros(36);changed[0]=1e3
    third,_=update_after_view(prediction,CensoredRanges(observation_visible,changed),model)
    np.testing.assert_array_equal(first,third)
    assert not d['resampled_or_replaced'] and d['all_particle_entries_retained']


def test_validation_and_selection_api_cannot_take_actualobs_or_truth():
    assert 'actual_observation' not in inspect.signature(select_point_active_view).parameters
    assert 'truth' not in inspect.signature(score_censored_views).parameters
    with pytest.raises(ValueError):TargetVariance(np.array([[np.nan]]))
    with pytest.raises(ValueError):TargetVariance(np.ones((2,1))).risks([[-1.,2.]])
    model=CensoredRangeModel(np.eye(36)*(.0002**2),.005,.05,0)
    with pytest.raises(ValueError):select_point_active_view(np.zeros((2,1)),[],model)
    with pytest.raises(ValueError):update_after_view(CensoredRanges(np.zeros((2,36),bool),np.zeros((2,36))),CensoredRanges(np.zeros(36,bool),np.zeros(36)),model)


def test_informative_visibility_view_wins_and_equal_helpful_views_tie_lowest():
    model=CensoredRangeModel(np.eye(1),.1,background_std_m=1.,outlier_probability=0)
    uninformative=CensoredRanges([[False],[False]],[[np.nan],[np.nan]])
    informative=CensoredRanges([[False],[True]],[[np.nan],[0.]])
    result=score_censored_views(np.array([[0.],[2.]]),[uninformative,informative,informative],model)
    assert result['selected_index']==1
    np.testing.assert_allclose(result['expected_target_variance_m2'],[1.,.36,.36],atol=2e-15)
    assert result['selected_advantage_mc_se_m2'][2]==0
