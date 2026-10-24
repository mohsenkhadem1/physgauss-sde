"""Sparse static compressible neo-Hookean reference, with exact hard clamps.

No experiments run on import. The P1 constitutive evaluator is reused unchanged
from the independently controlled analytic nonlinear reference. Fixed reference
surface weights define dead loads and a quadratic compliant foundation.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from experiments.campaign_sparse_nonlinear_gaussian import SparseNonlinearGaussian

GATES = dict(absolute_residual_N=1e-10, relative_residual=1e-8,
             minimum_J=.1, maximum_iterations=60, maximum_backtracks=40,
             armijo=1e-4)


class StaticFailure(RuntimeError):
    def __init__(self, message, history):
        super().__init__(message)
        self.history = history


class NonlinearReference:
    """All geometry, fixed nodes, and surface support weights are explicit.

    support_matrix must be the SAME consistent integrated reference-space
    surface operator used by the matched linear model. Units: E Pa, K N/m, f N.
    No dynamic damping, gravity, or contact is implicitly added.
    """
    elastic = SparseNonlinearGaussian.elastic

    def __init__(self, points, tets, fixed_nodes, support_matrix, *, young,
                 support_K, poisson=.35):
        self.rest = np.asarray(points, dtype=np.float64).copy()
        raw_tets = np.asarray(tets)
        if raw_tets.dtype.kind not in 'iu':
            raise ValueError('Integer tetrahedra required')
        self.tets = raw_tets.astype(np.int64, copy=True)
        self.young, self.poisson = float(young), float(poisson)
        self.support_K = float(support_K)
        if self.rest.ndim != 2 or self.rest.shape[1] != 3 or not np.isfinite(self.rest).all():
            raise ValueError('Finite rest[N,3] required')
        if self.tets.ndim != 2 or self.tets.shape[1] != 4 or not len(self.tets) or np.any(self.tets < 0) or np.any(self.tets >= len(self.rest)):
            raise ValueError('Nonempty valid tetrahedra required')
        if not np.isfinite([young, support_K, poisson]).all() or young <= 0 or support_K < 0 or not -1 < poisson < .5:
            raise ValueError('Finite valid elastic/support parameters required')
        fixed = np.asarray(fixed_nodes)
        if fixed.ndim != 1 or fixed.dtype.kind not in 'iu' or not len(fixed) or len(np.unique(fixed)) != len(fixed) or np.any(fixed < 0) or np.any(fixed >= len(self.rest)):
            raise ValueError('Explicit distinct fixed node indices required')
        self.fixed = fixed.astype(np.int64)
        self.mask = np.zeros_like(self.rest, dtype=bool)
        self.mask[self.fixed] = True
        self.free = np.flatnonzero(~self.mask.ravel())
        if not len(self.free):
            raise ValueError('At least one free degree of freedom required')
        W = sparse.csr_matrix(support_matrix, dtype=float)
        if W.shape != (len(self.rest), len(self.rest)) or not np.isfinite(W.data).all():
            raise ValueError('Finite consistent scalar surface matrix [N,N] required')
        difference = W-W.T
        if np.max(np.abs(difference.data), initial=0) > 1e-13 or np.any(W.data < -1e-15) or abs(float(W.sum())-1) > 1e-6:
            raise ValueError('Symmetric nonnegative normalized consistent surface matrix required')
        # Nonnegative entries do not imply PSD. Reject a materially indefinite
        # supplied operator rather than repairing it. Shared surface assembly is
        # a positive weighted sum of N N.T and thus PSD by construction.
        if len(self.rest) <= 500 and np.linalg.eigvalsh(W.toarray())[0] < -1e-12:
            raise ValueError('Surface operator must be positive semidefinite')
        self.W = W.copy()
        dm = np.swapaxes(self.rest[self.tets[:, 1:]]-self.rest[self.tets[:, :1]], -1, -2)
        det = np.linalg.det(dm)
        if not np.isfinite(det).all() or np.any(det <= 0):
            raise ValueError('Positive reference tetrahedra required')
        self.volumes = det/6
        inv = np.linalg.inv(dm)
        self.gradients = np.concatenate([-inv.sum(axis=1)[:, None], inv], axis=1)
        dofs = (self.tets[:, :, None]*3+np.arange(3)).reshape(-1, 12)
        self.rows = np.repeat(dofs, 12, axis=1).ravel()
        self.cols = np.tile(dofs, (1, 12)).ravel()
        self.minimum_J = GATES['minimum_J']
        self.foundation = support_K*sparse.kron(self.W, sparse.eye(3), format='csr')

    def evaluate(self, displacement, force, *, tangent=False):
        u, f = np.asarray(displacement, float), np.asarray(force, float)
        if u.shape != self.rest.shape or f.shape != u.shape or not np.isfinite(u).all() or not np.isfinite(f).all():
            raise ValueError('Explicit finite displacement and nodal force [N,3] required')
        e, r, h, J = self.elastic(self.rest+u, tangent=tangent)
        ku = (self.foundation @ u.ravel()).reshape(u.shape)
        energy = e + .5*np.sum(u*ku)-np.sum(f*u)
        residual = r+ku-f
        if not np.isfinite(energy) or not np.isfinite(residual).all():
            raise ValueError('Nonfinite physical energy/residual')
        return float(energy), residual, h+self.foundation if tangent else None, J

    def solve(self, force, *, initial_displacement=None):
        f = np.asarray(force, float)
        if f.shape != self.rest.shape or not np.isfinite(f).all():
            raise ValueError('Finite nodal dead force [N,3] required')
        u = np.zeros_like(self.rest) if initial_displacement is None else np.asarray(initial_displacement, float).copy()
        if u.shape != self.rest.shape or not np.isfinite(u).all() or np.any(u[self.mask] != 0):
            raise ValueError('Initial displacement must respect exact zero clamps')
        tolerance = max(GATES['absolute_residual_N'], GATES['relative_residual']*np.linalg.norm(f))
        history = []
        for iteration in range(GATES['maximum_iterations']+1):
            energy, residual, hessian, J = self.evaluate(u, f, tangent=True)
            r = residual.ravel()[self.free]
            norm = float(np.linalg.norm(r))
            history.append(dict(iteration=iteration, free_residual_N=norm, minimum_J=float(J.min()), energy_J=energy))
            if norm <= tolerance:
                break
            if iteration == GATES['maximum_iterations']:
                raise StaticFailure('Original free residual failed iteration cap', history)
            try:
                direction = splu(hessian[self.free][:, self.free].tocsc()).solve(-r)
            except RuntimeError as exc:
                raise StaticFailure('Unmodified physical tangent factorization failed', history) from exc
            slope = float(r@direction)
            if not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0:
                raise StaticFailure('Unmodified Newton direction is not descent', history)
            for backtrack in range(GATES['maximum_backtracks']):
                alpha = 2.**(-backtrack)
                trial = u.copy()
                trial.ravel()[self.free] += alpha*direction
                try:
                    e2, r2, _, _ = self.evaluate(trial, f)
                except ValueError:
                    continue
                # Roundoff-only acceptance still requires strict residual decrease.
                window = 128*np.finfo(float).eps*max(abs(energy), self.young*self.volumes.sum(), 1e-30)
                if e2 <= energy+GATES['armijo']*alpha*slope or (abs(e2-energy) <= window and np.linalg.norm(r2.ravel()[self.free]) < norm):
                    u = trial
                    history[-1]['step_length'] = alpha
                    break
            else:
                raise StaticFailure('Positive-J line search exhausted', history)
        _, residual, _, J = self.evaluate(u, f)
        free_norm = float(np.linalg.norm(residual.ravel()[self.free]))
        if free_norm > tolerance or np.any(u[self.mask] != 0):
            raise StaticFailure('Final original residual or exact clamp failed', history)
        # residual on fixed nodes is the clamp reaction on the body.
        clamp = np.where(self.mask, residual, 0.)
        support = -(self.foundation@u.ravel()).reshape(u.shape)
        total_force = f+clamp+support
        balance = total_force.sum(axis=0)
        moment = np.cross(self.rest+u, total_force).sum(axis=0)
        force_balance_tolerance = np.sqrt(len(self.rest))*tolerance+1e-12*max(np.linalg.norm(f), 1.)
        moment_balance_tolerance = np.max(np.linalg.norm(self.rest+u, axis=1))*force_balance_tolerance
        if np.linalg.norm(balance) > force_balance_tolerance or np.linalg.norm(moment) > moment_balance_tolerance:
            raise StaticFailure('Reaction force/moment balance failed', history)
        return u, dict(passed=True, free_residual_N=free_norm, tolerance_N=float(tolerance),
                      minimum_J=float(J.min()), iterations=iteration,
                      clamp_reaction_N=clamp.sum(axis=0).tolist(),
                      support_reaction_N=support.sum(axis=0).tolist(),
                      force_balance_error_N=float(np.linalg.norm(balance)),
                      force_balance_tolerance_N=float(force_balance_tolerance),
                      moment_balance_error_Nm=float(np.linalg.norm(moment)),
                      moment_balance_tolerance_Nm=float(moment_balance_tolerance),
                      clamp_maximum_error_m=float(np.max(abs(u[self.mask]))),
                      history=history, tangent_shift=0.)


def from_shared(n, *, young, support_K):
    """Build from exactly the linear campaign's mesh and integrated support."""
    from experiments.campaign_surface_traction_static_v01 import SparseStaticForward
    linear = SparseStaticForward(n)
    return NonlinearReference(linear.X, linear.tets, linear.fixed, linear.W_scalar,
                              young=young, support_K=support_K)


def design():
    return dict(schema='clamped-surface-nonlinear-v01',
                law='mu/2*(tr(F.T F)-3)-mu*log(J)+lambda/2*log(J)^2',
                mesh_source='campaign_surface_traction_static_v01.geometry',
                traction_source='campaign_surface_traction_static_v01.surface_traction',
                support_source='SparseStaticForward.W_scalar; consistent integral g Ni Nj / exact integral g; shared quadrature qualification required; never renormalized',
                loads='Shared qualification_loads: smooth sin² top normal−.10N; paired top/bottom axial+.24N each (net+.48N), zero reference eccentric moment',
                geometry_m=[.36,.08,.08],both_end_faces_fixed=True,
                material_prior_E_Pa=[2000,8000],poisson=.35,
                support_prior_K_N_m=[5,30],support_center_m=[.24,.08,.04],support_width_m=.025,
                gates=GATES.copy(), tangent='exact analytic sparse; no diagonal shifts',
                scope='Static nonlinear finite-element reference; continuum and dynamic qualification separate',
                no_automatic_experiments=True)
