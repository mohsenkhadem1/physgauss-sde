"""Future-only generalized inertia diagnostic for an exact SPD rest operator.

The caller supplies an independently justified SPD A0 and its exact factor solve.
No stiffness shifts, clipping or matrix repairs are performed. A converged
smallest generalized Ritz pair is numerical evidence, not an interval-arithmetic
certificate that ARPACK has found the global smallest eigenvalue.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from experiments.campaign_exact_rest_pcg_v01 import _symmetric_matrix

DESIGN = dict(schema='generalized-stability-v01', seed=2026102491,
              eigenpair_relative_tolerance=1e-6, normalization_tolerance=1e-6,
              eigsh_tolerance=1e-10, maximum_iterations=5000,
              eigenvalue_units='dimensionless generalized stiffness ratio',
              rest_SPD='Externally justified assumption; LU is not a certificate',
              matrix_modified=False, rigorous_global_eigenvalue_bound=False)


def generalized_stability(tangent, rest, rest_solve):
    """Smallest H v=lambda A0 v, using an already factored exact SPD A0.

    The explicit A0-dual residual divided by the A0 norm bounds distance to
    *some* generalized eigenvalue. Its positive lower endpoint and SA search
    support an inertia diagnostic; they are not a rigorous global enclosure.
    """
    H = _symmetric_matrix(tangent)
    A = _symmetric_matrix(rest, shape=H.shape)
    if H.shape[0] < 2:
        raise ValueError('At least two free coordinates required')
    inverse = LinearOperator(A.shape, matvec=rest_solve, dtype=np.float64)
    rng = np.random.default_rng(DESIGN['seed'])
    value, vectors = eigsh(H, M=A, Minv=inverse, k=1, which='SA',
                           v0=rng.normal(size=H.shape[0]),
                           tol=DESIGN['eigsh_tolerance'],
                           maxiter=DESIGN['maximum_iterations'])
    lam = float(value[0]); v = vectors[:, 0]
    Av = A @ v; Hv = H @ v
    norm2 = float(v @ Av)
    if not np.isfinite(lam) or not np.isfinite(v).all() or not np.isfinite(norm2) or norm2 <= 0:
        raise FloatingPointError('Nonfinite pair or nonpositive rest norm')
    r = Hv - lam * Av
    inverse_r = np.asarray(rest_solve(r), float)
    if inverse_r.shape != r.shape or not np.isfinite(inverse_r).all():
        raise FloatingPointError('Invalid exact rest inverse residual')
    solve_error = float(np.linalg.norm(A @ inverse_r - r))
    solve_tolerance = max(1e-13, 1e-11 * float(np.linalg.norm(r)))
    dual2 = float(r @ inverse_r)
    if not np.isfinite(dual2) or dual2 < 0:
        raise FloatingPointError('Nonpositive dual quadratic form; no clipping permitted')
    residual = float(np.sqrt(dual2 / norm2))
    relative = residual / max(abs(lam), np.finfo(float).tiny)
    lower = lam - residual
    normalization_ok = abs(norm2 - 1.) <= DESIGN['normalization_tolerance']
    accepted = bool(normalization_ok and solve_error <= solve_tolerance
                    and relative <= DESIGN['eigenpair_relative_tolerance'] and lower > 0)
    return dict(passed=accepted, smallest_generalized_Ritz_ratio=lam,
                rest_norm_squared=norm2, normalization_passed=normalization_ok,
                explicit_physical_pair_residual_norm=float(np.linalg.norm(r)),
                generalized_dual_residual=residual, relative_pair_residual=relative,
                dimensionless_Ritz_lower_endpoint=lower,
                exact_rest_solve_residual=solve_error,
                exact_rest_solve_tolerance=solve_tolerance,
                units=DESIGN['eigenvalue_units'], matrix_modified=False,
                rest_SPD_externally_assumed=True,
                rigorous_global_eigenvalue_bound=False)
