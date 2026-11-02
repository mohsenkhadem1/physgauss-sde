import copy
import numpy as np
import pytest
import torch
from scripts.compositional_neural_mechanics.joint_inference import JointMAP,neutral_parameters
from scripts.compositional_neural_mechanics.gauss_newton_inference import GaussNewtonMAP
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.mechanics_uncertainty import local_covariance,public_position_jacobian

def estimator():
 points=np.array([[.2,.3,.4],[.5,.6,.7],[.8,.4,.2]]);pose=np.zeros((2,6));pose[1,0]=.0008
 return JointMAP(GeometrySpec(),bulk_family='mooney_rivlin',interface_family=0,heterogeneity=0,observations=[dict(pose=pose,points=points,displacement=np.zeros_like(points))],position_sigma=50e-6,n=1,continuation=(1.,),gravity=(0,0,0))

def test_real_gn_bridge_prior_once_scale_null_and_truth_independence():
 torch.set_num_threads(1);e=estimator();theta=neutral_parameters();gn=GaussNewtonMAP(e)
 r,J,_=gn.residual_jacobian(theta)
 result=local_covariance(gn,theta,fit_status={'incomplete':True,'optimizer_success':False})
 np.testing.assert_allclose(result['precision'],np.diag(1/e.std**2)+J[:-15].T@J[:-15],rtol=1e-12,atol=1e-10)
 null=np.zeros(15);null[[0,3,5]]=1;null/=np.linalg.norm(null)
 np.testing.assert_allclose(J[:-15]@null,0,atol=1e-8)
 np.testing.assert_allclose(result['covariance']@null,null,atol=1e-8)
 assert result['removed_prior_rows']==15 and not result['calibrated']
 command=dict(name='public_stretch',pose=e.observations[0]['pose'],points=e.observations[0]['points'])
 public=public_position_jacobian(e,theta,[command])
 np.testing.assert_allclose(public['whitened_jacobian'],J[:-15],atol=1e-10)
 np.testing.assert_allclose(public['position_jacobian_m'],J[:-15]*e.sigmas[0],atol=1e-12)
 # Mutating even allowed-template outcomes cannot enter candidate prediction.
 e.observations[0]['displacement'][:]=12345
 again=public_position_jacobian(e,theta,[command])
 np.testing.assert_array_equal(public['predicted_displacement_m'],again['predicted_displacement_m'])
 np.testing.assert_array_equal(public['whitened_jacobian'],again['whitened_jacobian'])
 with pytest.raises(ValueError,match='truth'):public_position_jacobian(e,theta,[{**command,'true_displacement':np.zeros((3,3))}])

def test_prior_tail_validation_and_bound_flags():
 from types import SimpleNamespace
 from scripts.compositional_neural_mechanics.joint_inference import default_bounds
 theta=neutral_parameters();theta[0]=default_bounds()[0][0]
 e=SimpleNamespace(mean=neutral_parameters(),std=np.arange(1,16,dtype=float))
 class Fake:
  estimator=e
  def residual_jacobian(self,value):return np.r_[0.,(value-e.mean)/e.std],np.vstack([np.zeros((1,15)),np.diag(1/e.std)]),[]
 result=local_covariance(Fake(),theta)
 assert 'logscale'in result['near_bound_parameters']
 np.testing.assert_allclose(result['covariance'],np.diag(e.std**2))
 class Bad(Fake):
  def residual_jacobian(self,value):
   r,J,d=super().residual_jacobian(value);J[-1,-1]*=2;return r,J,d
 with pytest.raises(ValueError,match='prior rows'):local_covariance(Bad(),theta)
