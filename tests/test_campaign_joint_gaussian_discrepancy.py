import numpy as np
import pytest
from experiments.campaign_joint_gaussian_discrepancy import JointGaussianDiscrepancy
from experiments.censored_range_observation import CensoredRanges,CensoredRangeModel


def test_grouped_lowrank_matches_dense_marginal_and_target_conditionals_with_censoring():
    rng=np.random.default_rng(91);mo=rng.normal(0,1e-4,6);bo=rng.normal(0,2e-4,(6,3));mt=rng.normal(0,1e-3,5);bt=rng.normal(0,.002,(5,3))
    model=JointGaussianDiscrepancy(mo,bo,mt,bt);rows=np.array([2,5,0,4]);pred=rng.normal(0,1e-3,(5,4));h=np.array([[1,1,1,0],[1,0,1,0],[0,1,0,1],[0,0,0,0],[1,1,1,0]],bool);v=np.array([1,0,1,1],bool);y=rng.normal(0,1e-3,4)
    predictions=CensoredRanges(h,pred);observation=CensoredRanges(v,y);post=model.condition(predictions,observation,rows)
    target=rng.normal(0,.01,(5,5));moments=model.conditional_target_moments(target,post,feature_chunk_size=2);C=post.covariance_for_particles()
    for i in range(5):
        a=v&h[i];b=v&~h[i];B=bo[rows][a];r=y[a]-pred[i,a]-mo[rows][a];S=np.eye(a.sum())*.0002**2+B@B.T
        ll=np.sum(np.where(h[i]==v,np.log(.995),np.log(.005)))
        ll+=-.5*(np.sum(y[b]**2)/.05**2+b.sum()*np.log(2*np.pi*.05**2))
        if a.any():
            ll+=-.5*(r@np.linalg.solve(S,r)+np.linalg.slogdet(S)[1]+a.sum()*np.log(2*np.pi))
            expected_mean=B.T@np.linalg.solve(S,r);expected_C=np.eye(3)-B.T@np.linalg.solve(S,B)
        else:expected_mean=np.zeros(3);expected_C=np.eye(3)
        assert post.log_likelihood[i]==pytest.approx(ll,abs=2e-11)
        np.testing.assert_allclose(post.latent_mean[i],expected_mean,atol=2e-14,rtol=1e-12)
        np.testing.assert_allclose(C[i],expected_C,atol=3e-15,rtol=1e-12)
        np.testing.assert_allclose(moments.mean[i],target[i]+mt+bt@expected_mean,atol=1e-16)
        np.testing.assert_allclose(moments.variance_for_particles()[i],np.diag(bt@expected_C@bt.T),atol=1e-18)
    assert len(post.covariance_groups)<5 and not moments.boundary_projection_applied


def test_extra_view_uses_joint_over_base_not_independent_likelihood():
    model=JointGaussianDiscrepancy(np.zeros(2),np.full((2,1),.0002),np.zeros(1),np.ones((1,1)))
    prediction=CensoredRanges(np.ones((2,1),bool),[[0.],[.0004]]);observation=CensoredRanges([True],[.0002])
    base=model.condition(prediction,observation,[0]);joint,increment=model.condition_extra_view(base,prediction,observation,[1])
    np.testing.assert_allclose(joint.covariance_for_particles()[:,0,0],1/3,atol=1e-15)
    np.testing.assert_allclose(joint.latent_mean[:,0],[2/3,-2/3],atol=1e-15)
    expected=[]
    for i in range(2):
        mean=prediction.residual_m[i,0]+.0002*base.latent_mean[i,0];variance=.0002**2*(1+base.covariance_for_particles()[i,0,0]);residual=.0002-mean
        expected.append(np.log(.995)-.5*(residual**2/variance+np.log(2*np.pi*variance)))
    np.testing.assert_allclose(increment,expected,atol=2e-14)
    independent=model.marginal_log_likelihood(prediction,observation,[1])
    assert np.max(np.abs(increment-independent))>.1
    assert joint.observation_rows==(0,1) and base.observation_rows==(0,)
    with pytest.raises(ValueError,match='NEWdistinct'):model.condition_extra_view(base,prediction,observation,[0])


def test_absent_ranges_do_not_condition_discrepancy_and_background_is_independent():
    model=JointGaussianDiscrepancy(np.ones(2),np.ones((2,1)),np.zeros(1),np.ones((1,1)))
    predictions=CensoredRanges([[True,False],[False,False]],[[.1,np.nan],[np.nan,np.nan]])
    missing=model.condition(predictions,CensoredRanges([False,False],[np.nan,1e99]),[0,1])
    np.testing.assert_array_equal(missing.latent_mean,np.zeros((2,1)))
    np.testing.assert_allclose(missing.covariance_for_particles(),np.ones((2,1,1)))
    absent=CensoredRanges([[False,False]],[[np.nan,np.nan]]);y=np.array([.03,-.02])
    p=model.condition(absent,CensoredRanges([True,True],y),[0,1])
    expected=2*np.log(.005)-.5*(np.sum(y*y)/.05**2+2*np.log(2*np.pi*.05**2))
    assert p.log_likelihood[0]==pytest.approx(expected)
    np.testing.assert_array_equal(p.latent_mean,[[0.]])


def test_rankzero_reduces_to_existing_censored_sensor_without_discrepancy():
    model=JointGaussianDiscrepancy(np.zeros(3),np.empty((3,0)),np.zeros(2),np.empty((2,0)))
    pred=CensoredRanges([[True,False,True],[False,True,True]],[[.001,0,-.001],[0,.002,.003]])
    obs=CensoredRanges([True,False,True],[.0005,np.nan,-.0002])
    expected=CensoredRangeModel(np.eye(3)*.0002**2,.005,.05,0).log_prob(pred,obs)
    post=model.condition(pred,obs,[0,1,2]);np.testing.assert_allclose(post.log_likelihood,expected,atol=2e-13)
    target=np.ones((2,2));moments=model.conditional_target_moments(target,post)
    np.testing.assert_array_equal(moments.mean,target);np.testing.assert_array_equal(moments.variance_for_particles(),np.zeros((2,2)))


def test_fantasies_sample_conditional_shared_z_and_new_range_distribution():
    model=JointGaussianDiscrepancy(np.zeros(2),np.full((2,1),.0002),np.zeros(1),np.ones((1,1)))
    pred=CensoredRanges([[True]],[[0.]])
    base=model.condition(pred,CensoredRanges([True],[.0002]),[0])
    fantasy,info=model.fantasize_view(base,pred,[1],[1.],count=20000,seed=930)
    z=info['conditional_latent_z'][:,0];values=fantasy.residual_m[fantasy.visible]
    assert z.mean()==pytest.approx(.5,abs=.015) and z.var()==pytest.approx(.5,abs=.02)
    assert values.mean()==pytest.approx(.0001,abs=5e-6)
    assert values.var()==pytest.approx(1.5*.0002**2,rel=.035)
    assert fantasy.visible.mean()==pytest.approx(.995,abs=.002)
    repeat,rep_info=model.fantasize_view(base,pred,[1],[1.],count=20000,seed=930)
    np.testing.assert_array_equal(fantasy.visible,repeat.visible);np.testing.assert_array_equal(fantasy.residual_m,repeat.residual_m)
    np.testing.assert_array_equal(z,rep_info['conditional_latent_z'][:,0])


def test_conditioning_additional_measurement_obeys_gaussian_total_variance():
    # BaseC=.5, extraB=.0002, noise=.0002 => extraC=1/3.
    # E[Var(z|base,view)] + Var(E[z|base,view]) =1/3+1/6=.5.
    model=JointGaussianDiscrepancy(np.zeros(2),np.full((2,1),.0002),np.zeros(1),np.ones((1,1)))
    pred=CensoredRanges([[True]],[[0.]])
    base=model.condition(pred,CensoredRanges([True],[0.]),[0]);nodes,weights=np.polynomial.hermite_e.hermegauss(16);weights/=np.sqrt(2*np.pi)
    values=nodes*np.sqrt(1.5*.0002**2);means=[];variances=[]
    for y in values:
        joint,_=model.condition_extra_view(base,pred,CensoredRanges([True],[y]),[1]);means.append(joint.latent_mean[0,0]);variances.append(joint.covariance_groups[0,0,0])
    assert weights@variances+weights@np.square(means)-(weights@means)**2==pytest.approx(.5,abs=1e-14)


def test_rows_rank_noise_and_cross_model_states_are_guarded_without_jitter():
    with pytest.raises(ValueError):JointGaussianDiscrepancy([0.],np.zeros((1,17)),[0.],np.zeros((1,17)))
    with pytest.raises(ValueError):JointGaussianDiscrepancy([0.],np.zeros((1,1)),[0.],np.zeros((1,1)),isotropic_std_m=.0001)
    a=JointGaussianDiscrepancy([0.,0.],np.zeros((2,1)),[0.],np.ones((1,1)),isotropic_std_m=.0003)
    pred=CensoredRanges([[True,True]],[[0.,0.]])
    with pytest.raises(ValueError):a.condition(pred,CensoredRanges([True,True],[0.,0.]),[0,0])
    post=a.condition(pred,CensoredRanges([True,True],[0.,0.]),[0,1])
    b=JointGaussianDiscrepancy([0.,0.],np.zeros((2,1)),[0.],np.ones((1,1)))
    with pytest.raises(ValueError,match='SAMEjointmodel'):b.conditional_target_moments([[0.]],post)
    with pytest.raises(AttributeError):a.isotropic_std_m=.1
    with pytest.raises(ValueError):a.observation_loading_m[0,0]=1.


def test_particle_indexing_preserves_shared_conditionals_without_reweighting():
    model=JointGaussianDiscrepancy([0.,0.],np.array([[.0002],[.0001]]),[0.],np.ones((1,1)),isotropic_std_m=.0003)
    p=CensoredRanges([[True,True],[False,True],[True,False]],[[0.,.0002],[0.,-.0001],[.0003,0.]])
    obs=CensoredRanges([True,True],[.0001,-.0002]);full=model.condition(p,obs,[0,1]);indices=np.array([2,0,2])
    selected=full.take(indices);direct=model.condition(CensoredRanges(p.visible[indices],p.residual_m[indices]),obs,[0,1])
    np.testing.assert_allclose(selected.log_likelihood,direct.log_likelihood)
    np.testing.assert_allclose(selected.latent_mean,direct.latent_mean)
    np.testing.assert_allclose(selected.covariance_for_particles(),direct.covariance_for_particles())
    assert np.array_equal(selected.predictions.visible,p.visible[indices])
    assert full.observation_rows==selected.observation_rows==(0,1)
    with pytest.raises(ValueError):full.take([3])
