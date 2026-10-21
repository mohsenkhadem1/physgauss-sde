import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from experiments.campaign_point_rotation_retraction import retract_point_rotation


def rest():
    return np.array([[0.,0.,0.],[.2,0,0],[0,.3,0],[0,0,.1],[-.2,0,0],[0,-.3,0],[0,0,-.1],[.1,.2,.1]],dtype=np.float64)


def test_pure_infinitesimal_rotation_retracts_to_exact_SO3_with_exactpin_and_lengths():
    r=rest();omega=np.array([.1,-.2,.05]);u=np.cross(omega,r)
    actual=retract_point_rotation(r,u,0);R=Rotation.from_rotvec(omega).as_matrix();expected=r@R.T-r
    np.testing.assert_allclose(actual['fitted_infinitesimal_rotation_rad'],omega,atol=1e-16,rtol=1e-14)
    np.testing.assert_allclose(actual['displacement_m'],expected,atol=1e-16,rtol=1e-13)
    np.testing.assert_array_equal(actual['displacement_m'][0],np.zeros(3))
    q=r+actual['displacement_m'];np.testing.assert_allclose(np.linalg.norm(q[:,None]-q[None],axis=-1),np.linalg.norm(r[:,None]-r[None],axis=-1),atol=2e-16)
    np.testing.assert_allclose(actual['fitted_rotation_matrix'],R,atol=1e-16)


def test_firstorder_limit_and_unchanged_radial_elastic_residual():
    r=rest();omega=np.array([.2,-.1,.05]);norms=[]
    for scale in [.1,.05,.025]:
        u=np.cross(scale*omega,r);a=retract_point_rotation(r,u,0);norms.append(np.linalg.norm(a['correction_m']))
    np.testing.assert_allclose(np.array(norms[:-1])/norms[1:],4,atol=.001)
    # Isotropic radialstretch is orthogonal to every anchoredrotation, so zero
    # fittedrotation produces no artificial correction of this elasticfield.
    u=r*.01;actual=retract_point_rotation(r,u,0)
    np.testing.assert_allclose(actual['fitted_infinitesimal_rotation_rad'],0,atol=1e-17)
    np.testing.assert_allclose(actual['displacement_m'],u,atol=1e-18)


def test_batched_frames_zeros_and_inputs_unchanged():
    r=rest();omegas=np.array([[0.,0.,0.],[.02,.03,-.01]]);u=np.cross(omegas[:,None,:],r);original=u.copy()
    actual=retract_point_rotation(r,u,0)
    np.testing.assert_allclose(actual['fitted_infinitesimal_rotation_rad'],omegas,atol=1e-17)
    np.testing.assert_array_equal(actual['displacement_m'][0],u[0]);np.testing.assert_array_equal(u,original)
    np.testing.assert_array_equal(actual['displacement_m'][:,0],np.zeros((2,3)))


def test_tinyangle_correction_retains_secondorder_term_without_identity_cancellation():
    r=rest();omega=np.array([1e-12,2e-12,-1e-12]);u=np.cross(omega,r)
    actual=retract_point_rotation(r,u,0);leading=.5*np.cross(omega,np.cross(omega,r))
    assert np.linalg.norm(actual['correction_m'])>0
    np.testing.assert_allclose(actual['correction_m'],leading,atol=1e-36,rtol=1e-10)


def test_invalid_pin_drift_precision_and_singular_geometry_abort_without_repair():
    r=rest();u=np.zeros_like(r);u[0,0]=1e-15
    with pytest.raises(ValueError,match='exact'):retract_point_rotation(r,u,0)
    with pytest.raises(ValueError):retract_point_rotation(r.astype(np.float32),np.zeros_like(r),0)
    line=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    with pytest.raises(ValueError,match='Nondegenerate'):retract_point_rotation(line,np.zeros_like(line),0)
