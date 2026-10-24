"""Controls for fixed cohort, ratio MCSE, aggregation and multi-RHS ordering."""
import numpy as np
import pytest
from experiments import campaign_diverse_force_cohort_v01 as model
from experiments import run_diverse_force_cohort_v01 as run


def test_cohort_fixed_stratified_and_truth_free_conversion():
 u=model.cohort_units();assert u.shape==(12,4);np.testing.assert_array_equal(u,model.cohort_units())
 for j in range(4):assert sorted(np.floor(u[:,j]*12).astype(int))==list(range(12))
 np.testing.assert_allclose(model.physical(u),model.old.base.physical(u,(0,1,2,3)),rtol=2e-15)
 with pytest.raises(ValueError):model.physical(u[:,:2])


def test_noise_is_paired_within_object_and_independent_across_objects():
 seeds=run.DESIGN['poke_noise_seeds'];assert np.shape(seeds)==(12,16)
 assert len(set(np.array(seeds).reshape(-1)))==192
 assert not set(np.array(seeds).reshape(-1))&set(run.DESIGN['acquisition_noise_seeds']+run.DESIGN['MH_seeds']+run.DESIGN['random_policy_seeds'])


def test_weighted_ratio_against_direct_influence():
 rng=np.random.default_rng(17);x=rng.normal(size=(8,1600,3));w=np.exp(.1*x[:,:,0]);r=model.weighted_ratio_mcse(x,w);mu=np.einsum('cn,cnd->d',w,x)/w.sum();center=x-mu;v=np.einsum('cn,cnd->d',w,center**2)/w.sum();sd=np.sqrt(v)
 mean_se=[];sd_se=[]
 for k in (*model.BATCHES,1600):
  z=(w[:,:,None]*center/w.mean()).reshape(8,1600//k,k,3).mean(2).reshape(-1,3)
  s=(w[:,:,None]*(center**2-v)/w.mean()/(2*sd)).reshape(8,1600//k,k,3).mean(2).reshape(-1,3)
  mean_se.append(z.std(0,ddof=1)/np.sqrt(len(z)));sd_se.append(s.std(0,ddof=1)/np.sqrt(len(s)))
 np.testing.assert_allclose(r['mean'],mu,atol=1e-14);np.testing.assert_allclose(r['variance'],v,rtol=1e-13)
 np.testing.assert_allclose(r['mean_MCSE'],np.max(mean_se,axis=0),rtol=1e-12);np.testing.assert_allclose(r['SD_MCSE'],np.max(sd_se,axis=0),rtol=1e-12)
 assert r['cluster_means'].shape==(24,3)


def test_nonfinite_and_zero_weight_blocks_rejected():
 x=np.zeros((8,1600,1));w=np.ones((8,1600));w[0,:100]=0
 with pytest.raises(ValueError):model.weighted_ratio_mcse(x,w)
 x[0,0,0]=np.nan
 with pytest.raises(ValueError):model.weighted_ratio_mcse(x,np.ones((8,1600)))


def test_translation_invariance():
 rng=np.random.default_rng(18);x=rng.normal(size=(8,1600,2));w=np.exp(.1*x[:,:,0]);a=model.weighted_ratio_mcse(x,w);b=model.weighted_ratio_mcse(x+13,w)
 for k in ['variance','mean_MCSE','SD_MCSE']:np.testing.assert_allclose(a[k],b[k],atol=1e-13)


def test_all_failures_retained_and_superiority_suppressed():
 rows=[]
 for i in range(12):
  rows.append(dict(completed=True,index=i,passed=i!=3,policy_RMSE_m={name:[float(j+1)]*16 for j,name in enumerate(run.DESIGN['policies'])},baseline_RMSE_m={'nominal':2.},empirical_numerical_envelope_m={name:0. for name in run.DESIGN['policies'][1:]},choices={'target_variance':0}))
 r=run.aggregate(rows);assert r['objects']==12 and r['numerically_passed_objects']==11 and r['all_objects_retained']
 assert not any(x['superiority_supported']for x in r['comparisons'].values())
 with pytest.raises(ValueError):run.aggregate(rows[:-1])
 rows[-1]['index']=0
 with pytest.raises(ValueError):run.aggregate(rows)


def test_empirical_envelope_blocks_advantage():
 rows=[]
 for i in range(12):rows.append(dict(completed=True,index=i,passed=True,policy_RMSE_m={name:[float(j+1)]*16 for j,name in enumerate(run.DESIGN['policies'])},baseline_RMSE_m={'nominal':2.},empirical_numerical_envelope_m={name:10. for name in run.DESIGN['policies'][1:]},choices={'target_variance':0}))
 r=run.aggregate(rows);assert not any(x['superiority_supported']for x in r['comparisons'].values())


def test_multi_rhs_against_independent_numpy_and_scalar_forward():
 """Small explicit finite-model controls; run only after root review release."""
 import torch
 torch.set_num_threads(2)
 f=model.old.ArbitraryForceForward();u=np.array([[.2,.3,.4,.5],[.8,.7,.6,.4]]);theta=model.physical(u)
 target,y,functions=model.joint_features(f,theta,chunk=1,with_functions=True)
 expected_functions=np.concatenate([model.ridge.fixture_functions(theta,f.areas,f.X),f(theta,'A')[:,-1,1],f(theta,'B')[:,-1,4]],axis=1)
 np.testing.assert_allclose(functions,expected_functions,atol=1e-13,rtol=1e-12)
 programs=model.old.force_programs();pokes=model.old.poke_programs()
 expected=np.concatenate([f(theta,force)[:,model.TARGET_FRAMES].reshape(2,-1)for force in programs.values()],axis=1)
 np.testing.assert_allclose(target,expected,atol=1e-13,rtol=1e-12)
 oracle=np.concatenate([model.old.independent_numpy_forward(theta[0],force)[list(model.TARGET_FRAMES)].reshape(-1)for force in programs.values()])
 np.testing.assert_allclose(target[0],oracle,atol=1e-13,rtol=1e-12)
 ey=np.stack([model.old.original.selected_observations(f(theta,force))for force in pokes.values()]);np.testing.assert_allclose(y,ey,atol=1e-13,rtol=1e-12)
 for force in programs.values():
  np.testing.assert_array_equal(force[:24],model.old.original.known_force('A')[:24])
  np.testing.assert_allclose(f(theta,force)[:,:25],f(theta,'A')[:,:25],atol=1e-13,rtol=1e-12)


def test_half_mass_concentration_gate_is_separate_from_MCSE():
 row=dict(variance=np.ones(3),mean_MCSE=np.zeros(3),SD_MCSE=np.zeros(3),half_block_weight_ESS=7.)
 assert not model.moment_gate(row)['passed']
 row['half_block_weight_ESS']=8.;assert model.moment_gate(row)['passed']
