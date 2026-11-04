import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.affine_observation import affine_observation_matrix


def test_affine_rigid_transport_and_exact_reference():
    rng=np.random.default_rng(71);nodes=rng.normal(size=(30,3));points=rng.normal(size=(12,3))*.7
    A,diag=affine_observation_matrix(nodes,points)
    B=rng.normal(size=(3,3));translation=rng.normal(size=3)
    np.testing.assert_allclose(A@(nodes@B+translation),points@B+translation,atol=1e-12)
    assert diag['maximum_geometry_error']<1e-12
    R,_=np.linalg.qr(B);R[:,0]*=np.linalg.det(R)
    moved,_=affine_observation_matrix(nodes@R+translation,points@R+translation)
    np.testing.assert_allclose(moved,A,atol=1e-12)


def test_extrapolation_negative_weights_and_amplification_bound():
    nodes=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]])
    A,diag=affine_observation_matrix(nodes,np.array([[1.,1.,1.]]))
    np.testing.assert_allclose(A,[[-2,1,1,1]],atol=1e-12)
    assert diag['maximum_row_l1']==pytest.approx(5)
    assert diag['maximum_row_l2']==pytest.approx(np.sqrt(7))
    with pytest.raises(ValueError):affine_observation_matrix(nodes,np.array([[1.,1.,1.]]),max_l1=4)


def test_planar_span_is_allowed_but_off_plane_refused():
    nodes=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
    A,_=affine_observation_matrix(nodes,np.array([[.3,.2,0]]))
    np.testing.assert_allclose(A@nodes,[[.3,.2,0]],atol=1e-12)
    with pytest.raises(ValueError):affine_observation_matrix(nodes,np.array([[.3,.2,.1]]))


def test_exact_nodes_and_scale_equivariance():
    rng=np.random.default_rng(32);nodes=rng.normal(size=(20,3));points=rng.normal(size=(10,3))
    identity,_=affine_observation_matrix(nodes,nodes)
    np.testing.assert_array_equal(identity,np.eye(len(nodes)))
    A,_=affine_observation_matrix(nodes,points)
    scaled,_=affine_observation_matrix(nodes*1e-5,points*1e-5)
    np.testing.assert_allclose(A,scaled,atol=1e-12)


def test_nearly_planar_extrapolation_cannot_hide_instability():
    nodes=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1e-10]])
    with pytest.raises(ValueError):affine_observation_matrix(nodes,np.array([[.3,.2,.1]]))
