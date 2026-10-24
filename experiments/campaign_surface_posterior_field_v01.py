"""Exact weighted mean displacement fields for static prediction videos.

All boundary modes and the exact interior particular solution are retained.
This reconstructs the response of supplied positive quadrature hypotheses;
it does not itself establish posterior convergence or physical admission.
No data, rendering, automatic execution or file writes are embedded.
"""
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import splu
from experiments.campaign_exact_rest_pcg_v01 import _symmetric_matrix


class PosteriorMeanField:
    def __init__(self, elastic, foundation, forces):
        A = _symmetric_matrix(elastic)
        B = _symmetric_matrix(foundation, shape=A.shape).tocsr()
        B.eliminate_zeros()
        f = np.asarray(forces, float)
        if f.ndim != 2 or f.shape[0] != A.shape[0] or not np.isfinite(f).all():
            raise ValueError('Finite forces[free_dof,loads] required')
        self.boundary = np.flatnonzero(np.diff(B.indptr) > 0)
        self.interior = np.setdiff1d(np.arange(len(f)), self.boundary)
        b, i = self.boundary, self.interior
        if not len(b) or not len(i) or B[i].nnz or B[:, i].nnz:
            raise ValueError('Exact nonempty foundation boundary and free interior required')
        Aii = A[i][:, i]; Aib = A[i][:, b]; Abi = A[b][:, i]
        factor = splu(Aii)
        self.M = factor.solve(Aib.toarray())
        self.particular = factor.solve(f[i])
        scale = max(np.linalg.norm(Aib.data), np.finfo(float).tiny)
        residual = np.linalg.norm(Aii @ self.M - Aib.toarray()) / scale
        if residual > 1e-11:
            raise FloatingPointError('Interior elimination residual failed')
        S = A[b][:, b].toarray() - Abi @ self.M
        Bb = B[b][:, b].toarray()
        if not np.allclose(S, S.T, rtol=1e-12, atol=1e-13):
            raise FloatingPointError('Unmodified Schur matrix symmetry failed')
        eigenvalues, V = linalg.eigh(Bb, S, driver='gvd')
        spectral_residual = np.linalg.norm(Bb @ V - (S @ V) * eigenvalues)
        spectral_residual /= max(np.linalg.norm(Bb) * np.linalg.norm(V), np.finfo(float).tiny)
        orthogonality = np.max(abs(V.T @ S @ V - np.eye(len(b))))
        if spectral_residual > 1e-10 or orthogonality > 1e-8:
            raise FloatingPointError('Full boundary spectral controls failed')
        self.eigenvalues = eigenvalues
        self.V = V
        self.projected_force = V.T @ (f[b] - Abi @ self.particular)
        self.shape = f.shape
        self.diagnostics = dict(full_free_DOF=len(f), boundary_modes=len(b),
            eliminated_interior_DOF=len(i), all_modes_retained=True,
            eigenvalues_clipped=False, interior_elimination_relative_residual=float(residual),
            generalized_relative_residual=float(spectral_residual),
            maximum_metric_orthogonality_error=float(orthogonality),
            exact_interior_particular=True)

    def mean(self, physical, probability, *, chunk=512):
        """Return [loads,free_dof], preserving exact quadrature probabilities.

        Averages inverse stiffness responses, rather than using mean E or K.
        The caller separately binds these hypotheses to a qualified posterior.
        """
        theta = np.asarray(physical, float); p = np.asarray(probability, float)
        if theta.ndim != 2 or theta.shape[1] != 2 or len(theta) == 0:
            raise ValueError('Physical hypotheses[E,K] required')
        if p.shape != (len(theta),) or not np.isfinite(theta).all() or np.any(theta <= 0):
            raise ValueError('Finite positive physical hypotheses required')
        if not np.isfinite(p).all() or np.any(p < 0) or abs(p.sum() - 1) > 1e-12:
            raise ValueError('Normalized finite nonnegative quadrature probabilities required')
        if type(chunk) is not int or chunk < 1:
            raise ValueError('Positive bounded chunk required')
        modal_mean = np.zeros_like(self.eigenvalues); inverse_E_mean = 0.
        for start in range(0, len(theta), chunk):
            t = theta[start:start+chunk]; w = p[start:start+chunk]
            denominator = t[:, 0, None] + t[:, 1, None] * self.eigenvalues
            if not np.isfinite(denominator).all() or np.any(denominator <= 0):
                raise FloatingPointError('Nonpositive full modal denominator')
            modal_mean += w @ (1 / denominator)
            inverse_E_mean += float(w @ (1 / t[:, 0]))
        boundary = self.V @ (modal_mean[:, None] * self.projected_force)
        field = np.empty(self.shape)
        field[self.boundary] = boundary
        field[self.interior] = self.particular * inverse_E_mean - self.M @ boundary
        if not np.isfinite(field).all():
            raise FloatingPointError('Nonfinite posterior mean displacement')
        return field.T
