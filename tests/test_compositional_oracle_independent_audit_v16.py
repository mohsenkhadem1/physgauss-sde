import numpy as np
from scripts.compositional_neural_mechanics_reporting.oracle_independent_audit import metrics,nonrigid,bootstrap,rest_points

def test_independent_metric_units_rigid_removal_and_force_torque_split():
 rest=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
 R=np.array([[0,-1,0],[1,0,0],[0,0,1]],float)
 moved=rest@R.T+[.2,-.3,.1];u=moved-rest
 assert np.max(np.abs(nonrigid(rest,u)))<1e-14
 ref=dict(displacement=np.zeros_like(rest),wrenches=np.zeros((2,6)))
 pred=dict(displacement=u,wrenches=np.tile([1.,2.,3.,4.,5.,6.],(2,1)))
 out=metrics(pred,ref,rest)
 np.testing.assert_allclose(out['displacement_coordinate_rmse_m'],np.sqrt(np.mean(u**2)))
 assert out['canonical_nonrigid_coordinate_rmse_m']<1e-14
 np.testing.assert_allclose(out['force_coordinate_rmse_N'],np.sqrt(14/3))
 np.testing.assert_allclose(out['torque_coordinate_rmse_Nm'],np.sqrt(77/3))

def test_paired_object_bootstrap_constant_and_geometry_map():
 result=bootstrap(np.arange(6.),np.arange(6.)+2)
 assert result['objects']==6 and result['wins']==6 and result['percentile95_interval']==[2.,2.]
 q=np.array([[0,.5,.5],[.5,.5,.5],[1,.5,.5]])
 points=rest_points(q,dict(dimensions=[.2,.08,.08],taper=.2,arch=.01))
 np.testing.assert_allclose(points,[[0,0,0],[.1,0,.01],[.2,0,0]])
