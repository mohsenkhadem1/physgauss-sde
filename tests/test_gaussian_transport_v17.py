import numpy as np
import pytest
from scripts.compositional_dynamics_v17.gaussian_transport import bind_gaussians,transport_gaussians
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec,Q2Mechanics


def test_affine_transport_exact_including_outside_gaussian_offsets():
    model=Q2Mechanics(GeometrySpec(dimensions=(.12,.06,.06)),n=1)
    centers=np.array([[.04,.01,.02],[.13,.04,-.04],[-.01,0,.035]])
    cov=np.broadcast_to(np.diag([1e-6,2e-6,3e-6]),(3,3,3)).copy()
    binding=bind_gaussians(model.geometry,centers,cov)
    F=np.array([[1.1,.2,0],[0,.9,.1],[0,0,1.05]]);shift=np.array([.03,-.01,.02])
    u=model.rest@F.T+shift-model.rest
    result=transport_gaussians(model,u,binding)
    np.testing.assert_allclose(result['centers_m'],centers@F.T+shift,atol=1e-14)
    np.testing.assert_allclose(result['covariance_m2'],F@cov@F.T,atol=1e-17)


def test_rigid_rotation_preserves_covariance_spectrum_and_material_rotation():
    model=Q2Mechanics(GeometrySpec(taper=.2,arch=.01),n=1)
    centers=model.geometry.map(np.array([[.3,.2,.7],[.7,.4,.1]]))
    cov=np.broadcast_to(np.diag([1e-6,2e-6,4e-6]),(2,3,3)).copy()
    b=bind_gaussians(model.geometry,centers,cov)
    angle=.8;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    result=transport_gaussians(model,model.rest@R.T-model.rest,b)
    np.testing.assert_allclose(result['centers_m'],centers@R.T,atol=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(result['covariance_m2']),np.linalg.eigvalsh(cov),atol=1e-17)
    np.testing.assert_allclose(result['material_rotation'],np.broadcast_to(R,(2,3,3)),atol=1e-14)


def test_invalid_covariance_and_inverted_gradient_rejected():
    model=Q2Mechanics(n=1)
    with pytest.raises(ValueError):bind_gaussians(model.geometry,[[.1,0,0]],[np.diag([1.,0.,1.])])
    b=bind_gaussians(model.geometry,[[.1,0,0]],[np.eye(3)*1e-6])
    u=np.zeros_like(model.rest);u[:,0]=-2*model.rest[:,0]
    with pytest.raises(ValueError):transport_gaussians(model,u,b)
