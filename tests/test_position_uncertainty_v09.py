import numpy as np
from scripts.position_ood.uncertainty import sensitivity,covariance,probe_utilities
from scripts.position_ood.differentiable import DensePositionFEM
from scripts.position_ood.inverse import physical
from scripts.position_ood.fixture import controls

def test_implicit_sensitivity_predicts_perturbed_equilibrium():
    m=DensePositionFEM(1,'cpu');z=np.r_[np.log([20000,5000,8000]),.1,-.1,.5,.3,0,0,0,0];p=controls('translate_y');u,J=sensitivity(m,z,p,1)
    d=np.random.default_rng(29).normal(size=11)*1e-4
    shifted=m.predict(physical(m.tensor(z+d)),m.tensor(p),1).detach().numpy()
    np.testing.assert_allclose(shifted,u+J@d,atol=5e-8,rtol=1e-4)

def test_probe_uses_information_about_target_not_parameter_variance():
    J=np.zeros((10,2,3,2));J[9,:,:,0]=1;J[4,:,:,1]=100;J[5,:,:,0]=1
    C=covariance(np.zeros((3,2,3,2)));plan=probe_utilities(J,C,np.array([0,1]),candidates=(4,5));assert plan['selected']==5
    assert plan['expected_variance_reduction_m2'][0]==0
