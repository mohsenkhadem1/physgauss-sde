import numpy as np
from scipy.spatial.transform import Rotation
from scripts.compositional_dynamics_v17.matched_gaussian_states import covariance_parameters

def test_full_covariance_native_parameterization_preserves_affine_ellipsoid():
    rng=np.random.default_rng(914);F=rng.normal(size=(12,3,3));cov=F@F.swapaxes(-1,-2)+np.eye(3)[None]*.01
    scales,q=covariance_parameters(cov);R=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    restored=np.einsum('nij,nj,nkj->nik',R,scales**2,R)
    np.testing.assert_allclose(restored,cov,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R),1.,atol=1e-12)
