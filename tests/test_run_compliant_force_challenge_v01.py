"""Synthetic equality and failure controls; never read actual posterior arrays."""
import numpy as np
from experiments import run_compliant_force_challenge_v01 as runner


def test_streamed_covariance_matches_direct(tmp_path):
 rng=np.random.default_rng(41);x=rng.normal(size=(96,19))*1e-5;y=rng.normal(size=(3,96,12))*2e-5
 target=np.lib.format.open_memmap(tmp_path/'target.npy',mode='w+',shape=x.shape,dtype=float);target[:]=x
 mean,var,scores=runner.streamed_moments(target,y,width=13)
 np.testing.assert_allclose(mean,x.mean(0),atol=1e-20);np.testing.assert_allclose(var,x.var(0),atol=1e-24)
 expected=runner.helper.gaussian_target_reduction(x,y)
 np.testing.assert_allclose(scores,expected,rtol=1e-13)


def test_streamed_weighted_updates_and_original_contiguous_halves(tmp_path):
 rng=np.random.default_rng(42);x=rng.normal(size=(96,19))*1e-5;w=rng.uniform(size=(96,7));w/=w.sum(0)
 target=np.lib.format.open_memmap(tmp_path/'target.npy',mode='w+',shape=x.shape,dtype=float);target[:]=x
 mean,var,halves,mass,halfvar=runner.streamed_updates(target,w,blocks=16,width=5)
 np.testing.assert_allclose(mean,w.T@x,atol=1e-20)
 np.testing.assert_allclose(var,w.T@(x*x)-mean*mean,atol=1e-24)
 for block in range(16):
  s=slice(block*6,(block+1)*6)
  np.testing.assert_allclose(mass[block],w[s].sum(0),atol=1e-15)
  np.testing.assert_allclose(halves[block],w[s].T@x[s]/mass[block,:,None],atol=1e-20)


def test_fisher_linear_exact_jacobian_and_step_refinement():
 matrix=np.arange(48).reshape(12,4)*1e-7
 def forward(theta,force):
  intrinsic=theta.copy();intrinsic[:,[0,2,3]]=np.log(intrinsic[:,[0,2,3]])
  unit=(intrinsic-runner.helper.original.LOWER)/(runner.helper.original.UPPER-runner.helper.original.LOWER)
  result=np.zeros((len(theta),61,5,3));values=unit@matrix.T
  for j,frame in enumerate((12,24)):
   for k,node in enumerate((1,4)):result[:,frame,node]=values[:,j*6+k*3:j*6+k*3+3]
  return result*force
 cov=np.diag([.03,.02,.04,.01]);mean=np.full(4,.5)
 scores=runner.fisher_scores(forward,mean,cov,dict(one=1.,two=2.),1e-4)
 expected=[np.trace(cov)-np.trace(np.linalg.inv(np.linalg.inv(cov)+(matrix*f).T@(matrix*f)/runner.helper.original.SIGMA**2))for f in (1,2)]
 np.testing.assert_allclose(scores,expected,rtol=1e-10,atol=1e-15)
 np.testing.assert_allclose(scores,runner.fisher_scores(forward,mean,cov,dict(one=1.,two=2.),5e-5),rtol=1e-10)


def test_bootstrap_paired_exact_ties():
 values=np.arange(16)*1e-6;result=runner.bootstrap_paired(values,values)
 assert result['mean_difference_m']==0 and result['noise_only_95pct_CI_m']==[0.,0.]


def test_degenerate_half_importance_is_not_admitted():
 x=np.zeros((512,3));x[:32]=1e-4;y=np.zeros((2,512,12));truth=np.zeros(3);pokes=np.zeros((2,12))
 result=runner.evaluate_updates(x,y,truth,pokes,dict(target_variance=0,fixed_first=1))
 assert result['minimum_ESS']==512
 assert not result['passed']  # raw ESS alone does not establish mean precision


def test_future_only_target_and_no_policy_truth_access():
 def forward(theta,force):return np.broadcast_to(np.arange(61)[None,:,None,None],(len(theta),61,5,3))
 values=runner.target_prediction(forward,np.ones((2,4)),dict(a=1,b=2,c=3))
 assert values.shape==(2,1620) and values.min()==25 and values.max()==60


def test_anchored_variance_resolves_large_common_offset():
 rng=np.random.default_rng(57);x=1e3+rng.normal(size=(96,5))*1e-5;y=rng.normal(size=(2,96,12))*2e-5
 mean,var,scores=runner.streamed_moments(x,y,width=11)
 np.testing.assert_allclose(var,np.var(x-x[0],axis=0),rtol=1e-12)
 weights=np.full((96,1),1/96);_,variance,_,_,_=runner.streamed_updates(x,weights,blocks=16,width=5)
 np.testing.assert_allclose(variance[0],var,rtol=1e-12)


def test_standard_deviation_only_half_drift_not_admitted():
 x=np.zeros((512,3));x[:32:2]=1e-4;x[1:32:2]=-1e-4
 y=np.zeros((2,512,12));result=runner.evaluate_updates(x,y,np.zeros(3),np.zeros((2,12)),dict(target_variance=0))
 assert result['maximum_split_half_difference_m']<1e-18
 assert result['maximum_split_half_SD_difference_m']>2e-6 and not result['passed']
