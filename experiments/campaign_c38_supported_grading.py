"""Prospective boundary grading; same physical block, supports and known loads.

Only the reference mesh changes. The fixed, hypothesis-independent mapping
clusters corners at x=0,.06,.18,.24 and the four y/z surfaces. T10 edge nodes
are rebuilt AFTER corner mapping: each reference tet remains affine. There is
no mechanical accuracy/admission claim and no solver/GPU call in this module.
"""
from __future__ import annotations
import numpy as np
from experiments.boundary_block import block_mesh
from experiments.campaign_static_p2_block import enrich, shapes, EDGES
from experiments.campaign_p2_geometry import P2Geometry
from experiments.campaign_p2_exact_quadrature import exact_degree6_rule
from experiments.campaign_p2_keast_quadrature import positive_keast24
from experiments.campaign_block_surface_controls import SIZE, support_groups
from experiments.campaign_c38_static_family import (
    DENSITY, GRAVITY, POISSON, E_BOUNDS, SUPPORTS,
    compile_known_forces, mask_trace_certificate, array_record,
)

MAPPING = 'fixed_piecewise_symmetric_quadratic_boundary_grade_v01'


def symmetric_grade(u):
    """Strictly monotone [0,1] map, quadratic toward BOTH interval ends."""
    u = np.asarray(u, dtype=np.float64)
    if not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError('Finite unit reference coordinates required')
    return np.where(u <= .5, 2*u*u, 1-2*(1-u)*(1-u))


def graded_axes(cells):
    """Public geometry ONLY; quarter edges remain exact mesh planes."""
    if (len(cells) != 3 or any(type(n) is not int or n < 2 or n % 2 for n in cells)
            or cells[0] % 4):
        raise ValueError('Even positive transverse counts and nx multiple4 required')
    nx, ny, nz = cells
    anchors = (0., .06, .18, .24)
    counts = (nx//4, nx//2, nx//4)
    parts = []
    for lo, hi, n in zip(anchors[:-1], anchors[1:], counts):
        a = lo+(hi-lo)*symmetric_grade(np.arange(n+1, dtype=float)/n)
        a[0], a[-1] = lo, hi
        parts.append(a[:-1])
    axes = [np.r_[*parts, anchors[-1]]]
    for n in (ny, nz):
        a = -.04+.08*symmetric_grade(np.arange(n+1, dtype=float)/n)
        a[0], a[-1] = -.04, .04
        axes.append(a)
    if any(np.any(np.diff(a) <= 0) for a in axes):
        raise ValueError('Strictly positive graded reference spacings required')
    return tuple(axes)


class SupportedBoundaryGradedGeometry(P2Geometry):
    """Same polynomial stable-NH operator inputs with rebuilt affine T10 mesh.

    Supports affect only exact constraints; E affects only the material. Neither
    selects the grading. Both integration choices retain the old80+10 J guards.
    Full consistent body forces include constrained-node reaction contributions.
    """
    def __init__(self, cells, young, support, Fz, *, rule='keast24'):
        axes = graded_axes(cells)
        if (not np.isfinite(young) or not E_BOUNDS[0] <= young <= E_BOUNDS[1]
                or support not in SUPPORTS or rule not in ('keast24', 'exact80')):
            raise ValueError('Declared family material/support/integration required')
        self.young, self.poisson, self.density = float(young), POISSON, DENSITY
        self.minimum_J = .1
        self.mapping = MAPPING
        self.rotation, self.translation = np.eye(3), np.zeros(3)
        _, self.tets, self.faces = block_mesh(SIZE, tuple(cells), (SIZE[0]/2, 0, 0))
        corners = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        self.rest, self.elements = enrich(corners, self.tets)
        self.reference_local = self.rest
        tet = corners[self.tets]
        Dm = np.swapaxes(tet[:, 1:]-tet[:, :1], -1, -2)
        determinant = np.linalg.det(Dm)
        if not np.isfinite(determinant).all() or np.any(determinant <= 0):
            raise ValueError('Positive affine reference tetrahedra required')
        inverse = np.linalg.inv(Dm)
        self.linear_gradients = np.concatenate([-inverse.sum(1)[:, None], inverse], 1)
        self.volumes = determinant/6
        if not np.isclose(self.volumes.sum(), np.prod(SIZE), rtol=1e-13, atol=1e-16):
            raise ValueError('Same physical solid volume required')
        edges = np.stack([np.linalg.norm(tet[:, i]-tet[:, j], axis=-1) for i, j in EDGES])
        self.maximum_reference_tet_edge_m = float(edges.max())
        self.minimum_reference_tet_edge_m = float(edges.min())
        self.cell_volume_ratio = float(self.volumes.max()/self.volumes.min())
        old_bary, old_weights = exact_degree6_rule()
        self.check_bary = np.concatenate([
            np.eye(4), np.array([(np.eye(4)[i]+np.eye(4)[j])/2 for i,j in EDGES]), old_bary])
        if rule == 'keast24':
            self.bary, self.weights = positive_keast24()
            self.bulk_rule = 'Keast24_degree6_positive_with_old80_J_guards'
        else:
            self.bary, self.weights = old_bary, old_weights
            self.bulk_rule = 'positive_Duffy_5x4x4_degree6_exact'
        self.N = shapes(self.bary)
        self.order = self.bulk_rule
        groups, self.support_active = support_groups(self.rest, support)
        self.support_masks = groups
        self.mask = np.repeat(groups.any(0)[:, None], 3, 1)
        self.free = np.flatnonzero(~self.mask.ravel())
        compiled = compile_known_forces(self.rest, self.tets, self.faces, self.elements, Fz)
        self.load = compiled['total_N']
        self.body_integral_weights = compiled['body_integral_weights_m3']
        self.force_metadata = compiled['metadata']
        self.public_force_components = compiled
        self.mask_trace = mask_trace_certificate(self.rest, self.elements, self.faces, self.mask, support)
        self.grading_metadata = dict(mapping=MAPPING, axes_m=[a.tolist() for a in axes],
            corners_mapped_before_true_midpoints=True, identical_across_material_support_and_load=True,
            reference_geometry=array_record(self.rest), cells=list(cells),
            mechanical_convergence_qualified=False, domain_or_dataset_admitted=False)
        self.base_geometry_array_bytes = sum(a.nbytes for a in [self.rest, self.tets,
            self.faces, self.elements, self.linear_gradients, self.volumes, self.mask,
            self.load, self.body_integral_weights])
        self.estimated_peak_memory_bytes = None
