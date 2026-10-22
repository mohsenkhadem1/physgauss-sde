from math import factorial
import numpy as np
import pytest
from experiments.campaign_p2_jacobian_certificate import (
    MULTISETS, determinant_bernstein_bounds, whole_p2_jacobian_bounds)
from experiments.campaign_p2_geometry import P2Geometry


def evaluate_coefficients(coefficients, bary):
    answer = np.zeros(len(bary))
    for value, picks in zip(coefficients, MULTISETS):
        count = np.bincount(picks, minlength=4)
        multinomial = factorial(3)/np.prod([factorial(int(c)) for c in count])
        answer += value*multinomial*np.prod(bary**count, axis=1)
    return answer


def test_degree_three_identity_against_independent_determinants():
    rng = np.random.default_rng(2026102207)
    F = np.eye(3)+rng.normal(0, .4, (5, 4, 3, 3))
    bary = rng.dirichlet(np.ones(4), 401)
    result = determinant_bernstein_bounds(F)
    for i in range(5):
        direct = np.linalg.det(np.einsum('nv,vij->nij', bary, F[i]))
        polynomial = evaluate_coefficients(result['coefficients'][i], bary)
        np.testing.assert_allclose(direct, polynomial, rtol=2e-14, atol=2e-14)
        assert direct.min() >= result['lower'][i]
        assert direct.max() <= result['upper'][i]


def test_constant_and_positive_corners_with_negative_interior():
    A = np.array([[1.1, .2, -.1], [.1, .9, .3], [0, .1, 1.2]])
    result = determinant_bernstein_bounds(np.repeat(A[None], 4, axis=0))
    exact = np.linalg.det(A)
    np.testing.assert_allclose(result['coefficients'], exact, atol=1e-14)
    assert result['lower'] <= exact <= result['upper']
    F = np.repeat(np.eye(3)[None], 4, axis=0)
    F[1] = np.diag([-2., -.5, 1.])
    assert (np.linalg.det(F) > 0).all()
    assert np.linalg.det((F[0]+F[1])/2) < 0
    assert determinant_bernstein_bounds(F)['lower'] < 0


def test_whole_p2_positive_nonlinear_map_chunk_parity_and_rejection():
    m = P2Geometry((2, 2, 2), pattern='axial_pull')
    X = m.rest
    q = X+np.column_stack([2*X[:, 0]**2, X[:, 1]**2, 3*X[:, 2]**2])
    a = whole_p2_jacobian_bounds(m, q, chunk_size=7)
    b = whole_p2_jacobian_bounds(m, q, chunk_size=4096)
    np.testing.assert_array_equal(a['lower'], b['lower'])
    assert a['all_elements_positive'] and a['minimum_lower_bound'] > .6
    inverted = X.copy(); inverted[:, 0] *= -1
    assert not whole_p2_jacobian_bounds(m, inverted)['all_elements_positive']
    with pytest.raises(ValueError):
        determinant_bernstein_bounds(np.zeros((3, 3)))
