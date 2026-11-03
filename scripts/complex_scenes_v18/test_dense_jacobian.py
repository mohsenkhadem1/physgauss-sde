import numpy as np
from scripts.complex_scenes_v18.test_contact import body
from scripts.complex_scenes_v18.dense_jacobian import DenseJacobianAudit


def test_dense_orientation_affine_and_quadratic_mapping():
    m=body();audit=DenseJacobianAudit(m,5);F=np.array([[.8,.2,0],[0,1.1,.1],[0,0,1.2]])
    u=m.rest@(F-np.eye(3)).T
    np.testing.assert_allclose(audit.minimum(u),np.linalg.det(F),atol=1e-12)
    u=np.zeros_like(m.rest);u[:,0]=-10*m.rest[:,0]**2
    np.testing.assert_allclose(audit.minimum(u),1-20*.12,atol=1e-12)
