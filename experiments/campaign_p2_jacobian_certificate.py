"""Whole-tetrahedron Bernstein bounds for the determinant of affine F.

For straight-reference T10 elements, F is affine in barycentric coordinates.
det(F) is a degree-three polynomial: its 20 Bernstein coefficients bound its
range on the entire element. No point sampling or determinant clipping occurs.
The arithmetic margin covers coefficient evaluation from the supplied FP64 F
corners. It is not an interval enclosure of reference-geometry inversion.
"""
from itertools import combinations_with_replacement, permutations
import numpy as np

MULTISETS = tuple(combinations_with_replacement(range(4), 3))
PERMUTATIONS = tuple(tuple(sorted(set(permutations(x)))) for x in MULTISETS)


def determinant_bernstein_bounds(corner_F):
    """Return conservative polynomial bounds and coefficients, (...,4,3,3)."""
    F = np.asarray(corner_F, dtype=np.float64)
    if F.shape[-3:] != (4, 3, 3) or not np.isfinite(F).all():
        raise ValueError('Four finite affine-F corner matrices required')
    coefficients, margins = [], []
    eps = np.finfo(np.float64).eps
    # Six triple products, their alternating sum, permutation sum/division.
    # 128 eps is greater than the gamma bound for these <=6 permutations.
    gamma = 128 * eps / (1 - 128 * eps)
    for picks in PERMUTATIONS:
        values, absolute = [], []
        for i, j, k in picks:
            a, b, c = F[..., i, :, 0], F[..., j, :, 1], F[..., k, :, 2]
            terms = np.stack((a[..., 0]*b[..., 1]*c[..., 2],
                              a[..., 1]*b[..., 2]*c[..., 0],
                              a[..., 2]*b[..., 0]*c[..., 1],
                              a[..., 2]*b[..., 1]*c[..., 0],
                              a[..., 1]*b[..., 0]*c[..., 2],
                              a[..., 0]*b[..., 2]*c[..., 1]), axis=-1)
            values.append(terms[..., :3].sum(-1)-terms[..., 3:].sum(-1))
            absolute.append(abs(terms).sum(-1))
        coefficients.append(np.sum(values, axis=0)/len(picks))
        margins.append(gamma*np.sum(absolute, axis=0)/len(picks))
    coefficients, margins = np.stack(coefficients, -1), np.stack(margins, -1)
    if not np.isfinite(coefficients).all() or not np.isfinite(margins).all():
        raise FloatingPointError('Nonfinite determinant coefficient; no repair')
    lower = np.nextafter(np.min(coefficients-margins, axis=-1), -np.inf)
    upper = np.nextafter(np.max(coefficients+margins, axis=-1), np.inf)
    return dict(lower=lower, upper=upper, coefficients=coefficients,
                coefficient_roundoff_margin=margins)


def whole_p2_jacobian_bounds(model, positions, *, chunk_size=4096):
    """Bound each straight P2 element's computed affine-F polynomial.

    Reference gradients are those of the supplied immutable geometry. The
    corner-F construction has an additional FP64 arithmetic enclosure; the
    geometry gradients themselves remain supplied inputs, not interval inverses.
    """
    q = np.asarray(positions, dtype=np.float64)
    X = np.asarray(model.rest, dtype=np.float64)
    if q.shape != X.shape or q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError('Finite matching P2 positions required')
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Positive integer element chunk required')
    from experiments.campaign_static_p2_block import shape_gradients
    lower, upper, min_coefficient = [], [], []
    # F error from q-X and a ten-term sum of triple products. Reference
    # gradients are treated as supplied FP64 coefficients. Propagate each
    # component's absolute error to determinant triple products conservatively.
    eps = np.finfo(np.float64).eps
    gamma = 64*eps/(1-64*eps)
    for start in range(0, len(model.elements), chunk_size):
        ids = model.elements[start:start+chunk_size]
        g = model.linear_gradients[start:start+chunk_size]
        grad = np.stack([shape_gradients(b[None], g) for b in np.eye(4)], axis=1)
        u = q[ids]-X[ids]
        F = np.eye(3)+np.einsum('eai,eqaj->eqij', u, grad)
        eF = gamma*(np.eye(3)+np.einsum('eai,eqaj->eqij', abs(q[ids])+abs(X[ids]), abs(grad)))
        bound = determinant_bernstein_bounds(F)
        # Multilinearity bounds det(F+dF)-det(F) for all |dF|<=eF,
        # including mixed-column Bernstein polarizations.
        A = np.max(abs(F), axis=1)
        D = np.max(eF, axis=1)
        perm = tuple(permutations(range(3)))
        uncertainty = np.zeros(len(ids))
        for p in perm:
            a = np.stack([A[:, p[j], j] for j in range(3)], -1)
            d = np.stack([D[:, p[j], j] for j in range(3)], -1)
            uncertainty += (d[:, 0]*a[:, 1]*a[:, 2]+a[:, 0]*d[:, 1]*a[:, 2]+
                            a[:, 0]*a[:, 1]*d[:, 2]+d[:, 0]*d[:, 1]*a[:, 2]+
                            d[:, 0]*a[:, 1]*d[:, 2]+a[:, 0]*d[:, 1]*d[:, 2]+
                            d.prod(-1))
        uncertainty *= 1+gamma
        lower.append(np.nextafter(bound['lower']-uncertainty, -np.inf))
        upper.append(np.nextafter(bound['upper']+uncertainty, np.inf))
        min_coefficient.append(bound['coefficients'].min(-1))
    lo, hi = np.concatenate(lower), np.concatenate(upper)
    return dict(lower=lo, upper=hi,
                minimum_coefficient=np.concatenate(min_coefficient),
                minimum_lower_bound=float(lo.min()),
                all_elements_positive=bool(np.all(lo > 0)),
                scope='Whole computed affine-F polynomial; FP64 arithmetic margins; reference inverse not interval-enclosed')
