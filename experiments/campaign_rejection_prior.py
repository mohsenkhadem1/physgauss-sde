"""BC-uniform physical prior with joint face/UV rejection on PUBLIC hardware.

This is a NEW designed prior, not the seeded generator's conditional law.
For each fixed BC, propose a uniform face and continuous support coordinates,
then reject only exact compact-patch/attachment-clearance violations. ALL valid
support positions are included, without the conservative atlas restriction.

The normalized density is kernel/Z_BC(H); Z_BC is unknown. Accordingly there is
deliberately NO log_prob method. log_kernel ratios are valid only for moves
preserving BC (and fixed public hardware). A genuine whole-prior independence
proposal cancels the entire prior. Other proposals need their actual ratios.

Ideal uncapped rejection samples the stipulated prior exactly. A bounded call
can fail and raises rather than redraw a BC, omit a particle, or return partial
results. Its conservative failure-probability bound is reported; conditioning
on successful bounded calls must not silently be called mathematically exact.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np

try:
    from campaign_physical_prior import (
        ConditionalPhysicalPrior, FACE_BASES, FACE_NORMALS, SHEAR_DIRECTIONS,
        INTRINSIC_NAMES,
    )
except ImportError:
    from experiments.campaign_physical_prior import (
        ConditionalPhysicalPrior, FACE_BASES, FACE_NORMALS, SHEAR_DIRECTIONS,
        INTRINSIC_NAMES,
    )


@dataclass(frozen=True)
class RejectionPriorState:
    coordinates: np.ndarray          # [N,12]; support entries4:8 are GLOBAL UV
    boundary_type: np.ndarray        # [N], four BC types
    force_type: np.ndarray           # [N], four original load families
    support_face: np.ndarray         # [N], distribution weighted by valid area
    load_faces: np.ndarray           # [N,2], common incident public patch faces

    def __len__(self):
        return len(self.coordinates)

    def take(self, indices):
        ids = np.atleast_1d(indices)
        return type(self)(**{key: np.asarray(value)[ids].copy() for key, value in vars(self).items()})

    def with_coordinates(self, coordinates):
        return replace(self, coordinates=np.asarray(coordinates))


class SamplingBudgetExceeded(RuntimeError):
    """No state batch was returned; never replace failed BCs or particles."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('Fixed-BC rejection budget exhausted; entire sampling operation incomplete')


class ConditionalRejectionPrior:
    """Material AND unknown BC/support/load prior with full public attachments.

    Material marginals, force conventions, moving direction/amplitude and finite
    inactive coordinates follow campaign_physical_prior. The important change
    is that support coordinates are global face tangents, and face probability
    conditional on BC is proportional to its exact feasible support area.
    Both original handles remain present under known reset control queries.

    Certified atlas cells are used ONLY as positive-volume witnesses and to
    lower-bound acceptance probability, never to restrict samples or density.
    """

    dimension = 12
    unknown_normalizer_stratum = 'boundary_type with fixed public hardware'

    def __init__(self, reference_points, handle_centers_m, handle_patch_node_ids,
                 *, witness_grid=8, support_uv_limit=.18,
                 support_handle_clearance_m=.027, motion_amplitude_bounds_m=(.006, .010),
                 surface_tolerance_m=1e-7, max_attempts_per_particle=10000):
        if handle_patch_node_ids is None:
            raise ValueError('Full public2x9attachment node IDs are mandatory; centers are insufficient')
        if (isinstance(max_attempts_per_particle, bool) or
                int(max_attempts_per_particle) != max_attempts_per_particle or max_attempts_per_particle < 1):
            raise ValueError('Positive integer rejection budget required')
        # This frozen helper validates public geometry/hardware and supplies a
        # conservative, rigorous positive-volume witness for EACH BC. Failure
        # aborts the context; no BC is removed or renormalized away.
        self._witness = ConditionalPhysicalPrior(reference_points, handle_centers_m,
            grid=witness_grid, support_uv_limit=support_uv_limit,
            support_handle_clearance_m=support_handle_clearance_m,
            motion_amplitude_bounds_m=motion_amplitude_bounds_m,
            surface_tolerance_m=surface_tolerance_m,
            handle_patch_node_ids=handle_patch_node_ids)
        p = self._witness
        self.reference_points = p.reference_points
        self.handle_centers_m = p.handle_centers_m
        self.handle_patch_node_ids = p.handle_patch_node_ids
        self.center, self.size = p.center, p.size
        self.support_uv_limit = p.support_uv_limit
        self.support_handle_clearance_m = p.support_handle_clearance_m
        self.max_attempts_per_particle = int(max_attempts_per_particle)
        self.lower, self.upper = p.lower.copy(), p.upper.copy()
        self.lower[4:8], self.upper[4:8] = -support_uv_limit, support_uv_limit
        self.incident_faces, self.couple_basis = p.incident_faces, p.couple_basis
        self._face_nodes, self._node_clearance = [], []
        all_hardware_points = self.reference_points[self.handle_patch_node_ids].reshape(-1, 3)
        for face in range(6):
            ids = np.flatnonzero(np.isclose(self.reference_points[:, face % 3],
                (p.minimum if face < 3 else p.maximum)[face % 3], atol=surface_tolerance_m, rtol=0))
            self._face_nodes.append(ids)
            self._node_clearance.append(np.linalg.norm(
                self.reference_points[ids, None]-all_hardware_points[None], axis=-1).min(-1))
        g2 = p.grid**2
        self.acceptance_lower_bound = np.array([
            sum(len(p.feasible_cells[bc, face])/g2 *
                (len(p.feasible_cells[bc, (face+3) % 6])/g2 if bc == 2 else 1.)
                for face in range(6))/6 for bc in range(4)])
        if not (self.acceptance_lower_bound > 0).all():
            raise ValueError('No certified positive-volume support witness for every BC')
        self.last_sampling_diagnostics = None
        # rho/damping are PHYSICAL-uniform even though their chart entries are
        # logarithms. This base density includes their exp Jacobians below.
        self._base_log_constant = (-np.log(self.upper-self.lower).sum()
            +np.log(self.upper[1]-self.lower[1])-np.log(450)
            +np.log(self.upper[3]-self.lower[3])-np.log(.5)
            -2*np.log(4)-np.log(6)-sum(np.log(len(f)) for f in self.incident_faces))

    @staticmethod
    def _shapes(state):
        n = len(state.coordinates)
        shapes = dict(coordinates=(n, 12), boundary_type=(n,), force_type=(n,),
                      support_face=(n,), load_faces=(n, 2))
        if any(np.asarray(getattr(state, key)).shape != shape for key, shape in shapes.items()):
            raise ValueError('Incompatible RejectionPriorState batch shapes')
        return n

    def _category_valid(self, state):
        n = self._shapes(state)
        valid = np.ones(n, dtype=bool)
        for name in ['boundary_type', 'force_type', 'support_face', 'load_faces']:
            v = np.asarray(getattr(state, name))
            if n:
                valid &= (np.isfinite(v)&(v == np.floor(v))).reshape(n, -1).all(-1)
        for values, top in [(state.boundary_type, 4), (state.force_type, 4), (state.support_face, 6)]:
            valid &= (np.asarray(values) >= 0)&(np.asarray(values) < top)
        for j in range(2):
            valid &= np.isin(state.load_faces[:, j], self.incident_faces[j])
        return valid

    def feasible_support(self, state):
        """Exact deterministic static fixture predicate, no response/solver QC.

        Same stable k-nearest face-node selection and compact radius as the
        generator, with distance to ALL18 known attachment nodes. The inactive
        second support is ignored, and all continuous centers remain possible.
        """
        n = self._shapes(state)
        x = np.asarray(state.coordinates)
        bc, face = np.asarray(state.boundary_type), np.asarray(state.support_face)
        first_valid = np.isfinite(x[:, 4:6]).all(-1)&(np.abs(x[:, 4:6]) <= self.support_uv_limit).all(-1)
        second_valid = np.isfinite(x[:, 6:8]).all(-1)&(np.abs(x[:, 6:8]) <= self.support_uv_limit).all(-1)
        valid = (first_valid&((bc != 2)|second_valid)&np.isin(bc, range(4))&np.isin(face, range(6)))
        for b in range(4):
            count = [1, 16, 9, 16][b]
            for f in range(6):
                ids = np.flatnonzero(valid&(bc == b)&(face == f))
                if not len(ids):
                    continue
                for j in range(2 if b == 2 else 1):
                    actual_face = (f+3*j) % 6
                    uv = x[ids, 4+2*j:6+2*j]
                    center = (self.center+FACE_NORMALS[actual_face]*self.size/2
                        +uv@FACE_BASES[actual_face]*self.size)
                    nodes = self.reference_points[self._face_nodes[actual_face]]
                    distance = np.linalg.norm(nodes[None]-center[:, None], axis=-1)
                    chosen = np.argsort(distance, axis=1, kind='stable')[:, :count]
                    compact = np.take_along_axis(distance, chosen, axis=1).max(-1) <= .09
                    clear = self._node_clearance[actual_face][chosen].min(-1) > self.support_handle_clearance_m
                    valid[ids] &= compact&clear
        return valid

    def sample(self, n, rng=None):
        state, diagnostics = self.sample_with_diagnostics(n, rng)
        self.last_sampling_diagnostics = diagnostics
        return state

    def sample_with_diagnostics(self, n, rng=None):
        """Hold each sampled BC fixed until joint(face,UV) acceptance or abort.

        A budget exception returns NO partial state. It must not be caught to
        drop failed categories/particles, count a null MH move, or retry a new
        BC. The probability bound concerns this entire bounded sampling call.
        """
        if isinstance(n, bool) or int(n) != n or n < 0:
            raise ValueError('Nonnegative integer sample count required')
        n = int(n);rng = np.random.default_rng() if rng is None else rng
        x = rng.uniform(self.lower, self.upper, (n, 12))
        x[:, 1] = np.log(rng.uniform(400, 850, n));x[:, 3] = np.log(rng.uniform(.35, .85, n))
        bc, family = rng.integers(4, size=(2, n))
        face = np.zeros(n, dtype=np.int64)
        loads = np.stack([rng.choice(f, n) for f in self.incident_faces], -1)
        state = RejectionPriorState(x, bc, family, face, loads)
        attempts = np.zeros(n, dtype=np.int64);pending = np.ones(n, dtype=bool)
        for _ in range(self.max_attempts_per_particle):
            ids = np.flatnonzero(pending)
            if not len(ids):
                break
            # BC, material, force-family and all other parameters are retained.
            face[ids] = rng.integers(6, size=len(ids))
            x[ids, 4:8] = rng.uniform(-self.support_uv_limit, self.support_uv_limit, (len(ids), 4))
            attempts[ids] += 1
            pending[ids] = ~self.feasible_support(state.take(ids))
        # Union bound from known certified feasible area, conditional on the
        # FIXED drawn BC counts. It requires no estimated acceptance normalizer.
        counts = np.bincount(bc, minlength=4)
        def probability_bound(counts):
            active = counts > 0
            if not active.any():
                return 0., -math.inf
            with np.errstate(divide='ignore'):
                terms = np.log(counts[active])+self.max_attempts_per_particle*np.log1p(-self.acceptance_lower_bound[active])
            peak = terms.max()
            if np.isneginf(peak):
                return 0., -math.inf
            log_bound = min(0., float(peak+np.log(np.exp(terms-peak).sum())))
            # Do not publish an underflowed0 as a rigorous probability bound.
            return max(np.finfo(float).tiny, math.exp(log_bound)), log_bound
        failure_bound, log_failure_bound = probability_bound(counts)
        unconditional_bound, log_unconditional_bound = probability_bound(
            np.full(4, n) * (self.acceptance_lower_bound == self.acceptance_lower_bound.min()) /
            np.sum(self.acceptance_lower_bound == self.acceptance_lower_bound.min()))
        diagnostics = dict(particles=n, attempts_per_particle=attempts.tolist(),
            proposals=int(attempts.sum()), accepted=int((~pending).sum()),
            empirical_acceptance=n/attempts.sum() if attempts.sum() else None,
            fixed_boundary_counts=counts.tolist(), failed_particle_indices=np.flatnonzero(pending).tolist(),
            acceptance_lower_bound_by_boundary=self.acceptance_lower_bound.tolist(),
            whole_call_failure_probability_upper_bound=failure_bound,
            log_failure_probability_upper_bound_given_fixed_BCs=log_failure_bound,
            unconditional_worst_BC_failure_probability_upper_bound=unconditional_bound,
            log_unconditional_failure_probability_upper_bound=log_unconditional_bound,
            completed=bool(not pending.any()),
            qualification='Ideal uncapped rejection is exact; budget failure aborts operation; never redraw/drop BC or treat failure as a null MH proposal')
        self.last_sampling_diagnostics = diagnostics
        if pending.any():
            raise SamplingBudgetExceeded(diagnostics)
        return state, diagnostics

    def log_kernel(self, state):
        """Log p_base*Ifeasible, MISSING the BC-dependent -logZ_BC(H).

        Not a normalized density, not valid across different-BC target ratios,
        not a flow importance weight, ELBO prior term, or BC Bayes factor.
        Local BC-preserving MH ratios are valid; whole-prior proposals cancel.
        """
        self._shapes(state)
        x = np.asarray(state.coordinates, dtype=float)
        valid = (self._category_valid(state)&np.isfinite(x).all(-1)&
            (x >= self.lower).all(-1)&(x <= self.upper).all(-1)&(x[:, 9:11] < math.pi).all(-1))
        valid &= self.feasible_support(state)
        return np.where(valid, self._base_log_constant+x[:, 1]+x[:, 3], -np.inf)

    def normalization_stratum(self, state):
        """Local kernel-ratio proposals MUST keep this unchanged."""
        self._shapes(state)
        return np.asarray(state.boundary_type).copy()

    def decode(self, state, known_control=None):
        """Existing25D surrogate input; full hardware context stays separate."""
        if not np.isfinite(self.log_kernel(state)).all():
            raise ValueError('Cannot decode state outside exact feasible prior support')
        x = np.asarray(state.coordinates)
        bc, family, face = [np.asarray(v, dtype=np.int64) for v in
                            [state.boundary_type, state.force_type, state.support_face]]
        loads = np.asarray(state.load_faces, dtype=np.int64)
        def point(uv, f):
            return self.center+FACE_NORMALS[f]*self.size/2+np.einsum('ni,nij->nj', uv, FACE_BASES[f])*self.size
        first = point(x[:, 4:6], face)
        second = np.where((bc == 2)[:, None], point(x[:, 6:8], (face+3) % 6), first)
        normal = FACE_NORMALS[loads[:, 0]]
        couple = np.cos(x[:, 9:10])*self.couple_basis[0]+np.sin(x[:, 9:10])*self.couple_basis[1]
        direction = np.where((family == 0)[:, None], -normal, normal)
        direction = np.where((family == 2)[:, None], SHEAR_DIRECTIONS[loads[:, 0]], direction)
        direction = np.where((family == 3)[:, None], couple, direction)
        force0 = (x[:, 8]*np.where(bc == 0, .25, 1.))[:, None]*direction
        force1 = np.where((family == 3)[:, None], -force0, 0.)
        motion = np.einsum('ni,nij->nj', np.stack([np.cos(x[:, 10]), np.sin(x[:, 10])], -1), FACE_BASES[face])*x[:, 11:12]
        motion = np.where((bc == 3)[:, None], motion, 0.)
        control = self._witness._control(known_control)
        if control is not None:
            family = np.full(len(x), control[0], dtype=np.int64)
            force0 = np.broadcast_to(control[1], (len(x), 3))
            force1 = -force0 if control[0] == 3 else np.zeros_like(force0)
        physical = np.concatenate([x[:, :4], first, second,
            np.broadcast_to(self.handle_centers_m.reshape(1, 6), (len(x), 6)), force0, force1, motion], -1)
        return dict(physical=physical, boundary_type=bc.copy(), force_type=family.copy())

    def hardware_descriptor(self):
        return self._witness.hardware_descriptor()

    def metadata(self):
        spec = dict(prior='designed-fixed-BC-joint-face-UV-rejection-v01',
            continuous_names=[*INTRINSIC_NAMES[:4], 'support0_global_u', 'support0_global_v',
                'support1_global_u', 'support1_global_v', *INTRINSIC_NAMES[8:]],
            lower=self.lower.tolist(), upper=self.upper.tolist(),
            material_marginals=self._witness.metadata()['material_marginals'],
            reference_points_sha256=hashlib.sha256(self.reference_points.tobytes()).hexdigest(),
            public_hardware=self.hardware_descriptor(),
            bc_prior='uniform4; fixed through rejection',
            support_face_prior='withinBC proportional to feasibleUVarea; NOTuniform-feasible-face',
            support_proposal='uniform6faces and two independent uniform[-.18,.18]^2faceUV; secondUVnuisanceexcepttwo_patches',
            normalized_density_available=False,
            unknown_normalizer='Z_BC(publichardware)',
            exact_kernel_ratio_contract='FixedBC+fixedhardware only; whole-exact-prior independence cancels all; no other category-changing kernelratio justified',
            witness_grid=self._witness.grid,
            acceptance_lower_bound_by_boundary=self.acceptance_lower_bound.tolist(),
            max_attempts_per_particle=self.max_attempts_per_particle,
            qualification='NEWdesignedprior, notseededgeneratorconditional; no25DLebesguedensity; no simulatorfidelityclaim',
            hardware_qualification='Full descriptor conditionsprior but existing25Dforwarddoesnotconsumeattachmentgeometry',
            cap_policy='Abortwholeoperation; neverreturnpartialbatch/redrawBC/dropfailure/nullMHstep; reportfinite-capfailurebound')
        raw = json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()
        return dict(spec, prior_sha256=hashlib.sha256(raw).hexdigest())
