import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from experiments.campaign_generalized_stability_v01 import generalized_stability


def run(H, A):
    LU = splu(csc_matrix(A))
    return generalized_stability(H, A, LU.solve)


def test_dense_independent_generalized_oracle():
    rng = np.random.default_rng(4401)
    B = rng.normal(size=(12, 12)); A = B.T @ B + np.eye(12)
    C = rng.normal(size=(12, 12)); H = C.T @ C + .2*np.eye(12)
    result = run(H, A)
    truth = eigh(H, A, eigvals_only=True)[0]
    assert result['passed']
    assert abs(result['smallest_generalized_Ritz_ratio']-truth) < 1e-11
    assert result['relative_pair_residual'] < 1e-8
    assert result['units'].startswith('dimensionless')
    assert not result['rigorous_global_eigenvalue_bound']


def test_antisymmetric_negative_mode_not_missed():
    H = np.diag([4., 5., 6., 7.]); H[:2,:2] = [[1,2],[2,1]]
    A = np.diag([2.,3.,4.,5.])
    result = run(H, A)
    assert not result['passed']
    assert result['smallest_generalized_Ritz_ratio'] < 0
    assert abs(result['smallest_generalized_Ritz_ratio']-eigh(H,A,eigvals_only=True)[0]) < 1e-12
    assert np.linalg.eigvalsh(H)[0] < 0


def test_exact_zero_rejected_without_shift():
    H = np.diag([0., 1., 2., 3.])
    result = run(H, np.eye(4))
    assert not result['passed']
    assert np.array_equal(H, np.diag([0.,1.,2.,3.]))


def test_unchanged_inputs_and_invalid_symmetry_rejected():
    A=np.diag([1.,2.,3.,4.]); H=A.copy(); old=H.copy()
    assert run(H,A)['passed']
    assert np.array_equal(H,old)
    H[0,1]=.1
    with pytest.raises(ValueError):run(H,A)
