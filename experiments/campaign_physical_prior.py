"""An explicit NEW conditional physical prior for numerical/amortized inference.

Only public reference points and the two known hardware centers enter this
module. No recipes, labels, videos, checkpoints or simulator are read. This is
a designed distribution, NOT the exact conditional of the seeded generator or
the former learned joint flow. Its density is with respect to the 12 continuous
coordinates below and counting measure on the declared categorical variables;
there is no 25D physical-space Lebesgue density.

Support positions have an exactly normalized, finite rectangular atlas. Cells
are conservatively certified against the PUBLIC handle centers (or all public
attachment nodes when explicitly supplied), without stochastic rejection or an
unknown normalizing constant. This changes the support prior and may exclude
simulator-valid configurations. Centers alone do not determine attachments.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Mapping

import numpy as np


BOUNDARY_TYPES = ('point_pin', 'fixed_patch', 'two_patches', 'moving_patch')
FORCE_TYPES = ('push_pulse', 'pull_hold', 'shear_oscillation', 'twist_couple')
INTRINSIC_NAMES = (
    'log_E_Pa', 'log_rho_kg_m3', 'nu', 'log_drag_s_inverse',
    'support0_cell_u', 'support0_cell_v', 'support1_cell_u', 'support1_cell_v',
    'base_force_amplitude_N', 'couple_angle_rad', 'moving_angle_rad',
    'moving_amplitude_m',
)
FACE_AXES = np.array([[1, 2], [0, 2], [0, 1]] * 2, dtype=np.int64)
FACE_NORMALS = np.concatenate([-np.eye(3), np.eye(3)])
FACE_BASES = np.eye(3)[FACE_AXES]
SHEAR_DIRECTIONS = np.cross(FACE_NORMALS, np.eye(3)[np.abs(FACE_NORMALS).argmin(-1)])


@dataclass(frozen=True)
class PriorState:
    """Batch of intrinsic states. Resampling preserves every nuisance variable."""

    coordinates: np.ndarray                 # [N,12], including inactive entries
    boundary_type: np.ndarray               # [N], 0..3
    force_type: np.ndarray                  # [N], 0..3, original passive load
    support_face: np.ndarray                # [N], -x,-y,-z,+x,+y,+z
    support_cells: np.ndarray               # [N,2], row-major atlas cell indices
    load_faces: np.ndarray                  # [N,2], incident hardware faces

    def __len__(self):
        return len(self.coordinates)

    def take(self, indices):
        ids = np.atleast_1d(indices)
        return PriorState(**{k: np.asarray(v)[ids].copy() for k, v in self.__dict__.items()})

    def with_coordinates(self, coordinates):
        return replace(self, coordinates=np.asarray(coordinates))


class ConditionalPhysicalPrior:
    """Known-cuboid prior p(theta | public reference, public hardware centers).

    Material marginals match the fresh simulator design: E is log-uniform
    [1500,20000]Pa; rho, nu and damping are uniform [400,850], [.28,.42],
    [.35,.85] in their PHYSICAL units. BC/load families are independently
    uniform over four types. Moving amplitude is a deliberately uncertain
    uniform [6,10]mm, around the generator's fixed8mm; not a fitted distribution.

    Each feasible support face is equiprobable, then each certified equal-area
    cell is equiprobable, then its local coordinates are uniform. This is not
    area-uniform across faces. The second support is on the opposite face for
    two-patches, and is a proper inactive nuisance otherwise. Edges/corners have
    multiple incident load faces, which are explicitly marginalized categories.
    """

    dimension = 12

    def __init__(self, reference_points, handle_centers_m, *, grid=8,
                 support_uv_limit=.18, support_handle_clearance_m=.027,
                 motion_amplitude_bounds_m=(.006, .010), surface_tolerance_m=1e-7,
                 handle_patch_node_ids=None):
        points = np.array(reference_points, dtype=np.float64, copy=True)
        handles = np.array(handle_centers_m, dtype=np.float64, copy=True)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or
                not np.isfinite(points).all()):
            raise ValueError('Finite public reference points [M,3] required')
        if handles.shape != (2, 3) or not np.isfinite(handles).all():
            raise ValueError('Both finite public hardware centers [2,3] required')
        if isinstance(grid, bool) or int(grid) != grid or grid < 1:
            raise ValueError('Positive integer atlas grid required')
        if (not 0 < support_uv_limit < .5 or not np.isfinite(support_handle_clearance_m)
                or support_handle_clearance_m < 0 or not 0 < surface_tolerance_m < .001):
            raise ValueError('Invalid atlas, clearance or surface tolerance')
        motion = np.asarray(motion_amplitude_bounds_m, dtype=float)
        if motion.shape != (2,) or not np.isfinite(motion).all() or not 0 < motion[0] < motion[1]:
            raise ValueError('Positive increasing motion-amplitude bounds required')
        self.minimum, self.maximum = points.min(0), points.max(0)
        self.size = self.maximum-self.minimum
        self.center = (self.maximum+self.minimum)/2
        if (self.size <= 0).any():
            raise ValueError('Reference must span a metric cuboid in three dimensions')
        self.reference_points, self.handle_centers_m = points, handles
        self.grid, self.support_uv_limit = int(grid), float(support_uv_limit)
        self.support_handle_clearance_m = float(support_handle_clearance_m)
        self.surface_tolerance_m = float(surface_tolerance_m)
        self.handle_patch_node_ids = None
        if handle_patch_node_ids is not None:
            ids = np.asarray(handle_patch_node_ids)
            if (ids.shape != (2, 9) or ids.dtype.kind not in 'iu' or
                    (ids < 0).any() or (ids >= len(points)).any() or
                    len(np.unique(ids)) != 18):
                raise ValueError('Public handle patches require18distinct reference-node IDs shaped[2,9]')
            if any(not np.any(np.linalg.norm(points[patch]-h, axis=-1) <= surface_tolerance_m)
                   for patch, h in zip(ids, handles)):
                raise ValueError('Each public handle patch must include its declared hardware center')
            self.handle_patch_node_ids = ids.astype(np.int64, copy=True)
        self._clearance_points = (handles if self.handle_patch_node_ids is None else
                                  points[self.handle_patch_node_ids].reshape(-1, 3))
        if np.linalg.norm(handles[1]-handles[0]) <= surface_tolerance_m:
            raise ValueError('Distinct hardware centers required for a force-couple basis')
        if ((handles < self.minimum-surface_tolerance_m) |
                (handles > self.maximum+surface_tolerance_m)).any():
            raise ValueError('Hardware center outside public cuboid')
        self.incident_faces = []
        for j, handle in enumerate(handles):
            incident = np.flatnonzero(np.abs(np.einsum('fi,fi->f',
                handle-self.center-FACE_NORMALS*self.size/2, FACE_NORMALS)) <= surface_tolerance_m)
            if not len(incident):
                raise ValueError('Hardware centers must lie on the reference surface; no projection')
            if self.handle_patch_node_ids is not None:
                patch = points[self.handle_patch_node_ids[j]]
                incident = np.array([face for face in incident if np.all(np.isclose(
                    patch[:, face % 3],
                    (self.minimum if face < 3 else self.maximum)[face % 3],
                    atol=surface_tolerance_m, rtol=0))], dtype=np.int64)
                if not len(incident):
                    raise ValueError('Public patch must lie on a common cuboid face')
            self.incident_faces.append(incident)
        separation = handles[1]-handles[0]
        unit = separation/np.linalg.norm(separation)
        first = np.cross(unit, np.eye(3)[np.abs(unit).argmin()])
        first /= np.linalg.norm(first)
        self.couple_basis = np.stack([first, np.cross(unit, first)])
        self.lower = np.array([math.log(1500), math.log(400), .28, math.log(.35),
            *([0.]*4), .12, -math.pi, -math.pi, motion[0]])
        self.upper = np.array([math.log(20000), math.log(850), .42, math.log(.85),
            *([1.]*4), .24, math.pi, math.pi, motion[1]])
        self._build_atlas()

    def _point(self, uv, face):
        return (self.center+FACE_NORMALS[face]*self.size/2
                +np.einsum('...i,...ij->...j', uv, FACE_BASES[face])*self.size)

    def _build_atlas(self):
        # Certificate: for x inside a cell of radius r centered at c, each
        # selected k-nearest face node lies within R_k(c)+2r of c. Check EACH
        # such potentially selected node against the actual public hardware
        # points. This is substantially tighter than a bounding-sphere distance
        # to hardware, yet still holds for EVERY center inside the cell.
        cells = np.arange(self.grid**2)
        uv = np.stack([cells//self.grid, cells % self.grid], -1)
        uv = -self.support_uv_limit+(uv+.5)*(2*self.support_uv_limit/self.grid)
        self.feasible_cells = {}
        self.cell_clearance_lower_bound_m = {}
        for bc, count in enumerate([1, 16, 9, 16]):
            for face in range(6):
                face_points = self.reference_points[np.isclose(
                    self.reference_points[:, face % 3],
                    (self.minimum if face < 3 else self.maximum)[face % 3],
                    atol=self.surface_tolerance_m, rtol=0)]
                if len(face_points) < count:
                    raise ValueError('Public mesh has too few nodes for the declared support patch')
                centers = self._point(uv, np.full(len(cells), face))
                distance = np.linalg.norm(centers[:, None]-face_points[None], axis=-1)
                radius_k = np.partition(distance, count-1, axis=1)[:, count-1]
                radius = np.linalg.norm(self.size[FACE_AXES[face]] *
                                       self.support_uv_limit/self.grid)
                potential = distance <= radius_k[:, None]+2*radius+1e-12
                node_clearance = np.linalg.norm(face_points[:, None]-self._clearance_points[None], axis=-1).min(-1)
                lower = np.where(potential, node_clearance[None], np.inf).min(-1)
                feasible = ((lower > self.support_handle_clearance_m) &
                            (radius_k+radius <= .09))
                self.feasible_cells[bc, face] = cells[feasible]
                self.cell_clearance_lower_bound_m[bc, face] = lower
        self.feasible_faces = {}
        for bc in range(4):
            faces = [face for face in range(6) if len(self.feasible_cells[bc, face]) and
                     (bc != 2 or len(self.feasible_cells[bc, (face+3) % 6]))]
            if not faces:
                raise ValueError('No feasible support face for a declared BC; do not silently drop its prior mass')
            self.feasible_faces[bc] = np.array(faces, dtype=np.int64)

    @staticmethod
    def _shapes(state):
        n = len(state.coordinates)
        expected = dict(coordinates=(n, 12), boundary_type=(n,), force_type=(n,),
                        support_face=(n,), support_cells=(n, 2), load_faces=(n, 2))
        if any(np.asarray(getattr(state, k)).shape != shape for k, shape in expected.items()):
            raise ValueError('PriorState arrays have incompatible batch shapes')
        return n

    def sample(self, n, rng=None):
        """Independent joint samples; all inactive nuisance entries are retained."""
        if isinstance(n, bool) or int(n) != n or n < 0:
            raise ValueError('Nonnegative integer sample count required')
        n = int(n)
        rng = np.random.default_rng() if rng is None else rng
        x = rng.uniform(self.lower, self.upper, size=(n, self.dimension))
        # The box sampler above would be LOG-uniform for rho/drag. Replace
        # these two coordinates with the declared physical-uniform marginals.
        x[:, 1] = np.log(rng.uniform(400, 850, n))
        x[:, 3] = np.log(rng.uniform(.35, .85, n))
        bc, force = rng.integers(4, size=(2, n))
        face = np.empty(n, dtype=np.int64)
        cells = np.empty((n, 2), dtype=np.int64)
        for b in range(4):
            ids = np.flatnonzero(bc == b)
            face[ids] = rng.choice(self.feasible_faces[b], size=len(ids))
            for f in self.feasible_faces[b]:
                chosen = ids[face[ids] == f]
                cells[chosen, 0] = rng.choice(self.feasible_cells[b, f], size=len(chosen))
                cells[chosen, 1] = (rng.choice(self.feasible_cells[b, (f+3) % 6], size=len(chosen))
                    if b == 2 else rng.integers(self.grid**2, size=len(chosen)))
        loads = np.stack([rng.choice(faces, size=n) for faces in self.incident_faces], -1)
        return PriorState(x, bc, force, face, cells, loads)

    def log_prob(self, state):
        """Normalized intrinsic joint log density, including categorical mass.

        rho/drag Jacobians are included because the state stores their logs.
        Inactive coordinates/categories remain normalized nuisance factors.
        Invalid physical/categorical support returns -inf rather than clipping.
        """
        n = self._shapes(state)
        if n == 0:
            return np.empty(0, dtype=float)
        x = np.asarray(state.coordinates, dtype=float)
        valid = np.isfinite(x).all(-1) & (x >= self.lower).all(-1) & (x <= self.upper).all(-1)
        # Use half-open periodic angle domains; no double-counted endpoint.
        valid &= (x[:, 9:11] < math.pi).all(-1)
        lp = np.full(n, -np.log(self.upper-self.lower).sum())
        lp += np.log(self.upper[1]-self.lower[1])+x[:, 1]-np.log(450)
        lp += np.log(self.upper[3]-self.lower[3])+x[:, 3]-np.log(.5)
        lp -= 2*np.log(4)
        for name in ['boundary_type', 'force_type', 'support_face', 'support_cells', 'load_faces']:
            values = np.asarray(getattr(state, name))
            valid &= np.isfinite(values).reshape(n, -1).all(-1)
            valid &= (values == np.floor(values)).reshape(n, -1).all(-1)
        bc, family, face = np.asarray(state.boundary_type), np.asarray(state.force_type), np.asarray(state.support_face)
        valid &= (bc >= 0) & (bc < 4) & (family >= 0) & (family < 4)
        for i in range(n):
            if not valid[i]:
                continue
            b, f = int(bc[i]), int(face[i])
            if f not in self.feasible_faces[b]:
                valid[i] = False
                continue
            first = self.feasible_cells[b, f]
            second = self.feasible_cells[b, (f+3) % 6] if b == 2 else np.arange(self.grid**2)
            if state.support_cells[i, 0] not in first or state.support_cells[i, 1] not in second:
                valid[i] = False
                continue
            lp[i] -= np.log(len(self.feasible_faces[b]))+np.log(len(first))+np.log(len(second))
            for j in range(2):
                if state.load_faces[i, j] not in self.incident_faces[j]:
                    valid[i] = False
                lp[i] -= np.log(len(self.incident_faces[j]))
        return np.where(valid, lp, -np.inf)

    def _control(self, control):
        """The deployed interface uses handle0, with a separate known waveform.

        Accept public action-menu keys; reject a different contact/timing rather
        than silently pretending the decoder was trained for that experiment.
        """
        if control is None:
            return None
        family = control.get('force_type_index', control.get('force_type'))
        if isinstance(family, bool) or family not in range(4):
            raise ValueError('Known force_type_index in0..3 required')
        vector = np.asarray(control.get('peak_force_N'), dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError('Known world peak_force_N vector required')
        if control.get('handle_index', 0) != 0:
            raise ValueError('This25D decoder interface uses handle0 for a single applied load')
        if 'center_m' in control and not np.allclose(control['center_m'], self.handle_centers_m[0], atol=1e-8, rtol=0):
            raise ValueError('Known control center differs from public handle0')
        if control.get('start_s', 0.) != 0. or control.get('seconds', 3.) != 3.:
            raise ValueError('Only the existing3s reset waveform with start_s=0 is represented')
        return int(family), vector

    def decode(self, state, known_control: Mapping | None = None):
        """Decode passive parameters or replace old loads by a known reset load.

        Returns physical[N,25], boundary_type[N], force_type[N]. Hardware is
        a conditioning argument of THIS prior from construction, not an edit
        to samples from the old learned joint density. Known controls remove
        all dependence on original amplitude/family/normal/couple nuisances.
        """
        if not np.isfinite(self.log_prob(state)).all():
            raise ValueError('Cannot decode a state outside the declared prior')
        x = np.asarray(state.coordinates)
        bc, family, face = (np.asarray(a, dtype=np.int64) for a in
                            [state.boundary_type, state.force_type, state.support_face])
        cells, load_faces = (np.asarray(a, dtype=np.int64) for a in
                             [state.support_cells, state.load_faces])
        cell_xy = np.stack([cells//self.grid, cells % self.grid], -1)
        uv = -self.support_uv_limit+(cell_xy+x[:, 4:8].reshape(-1, 2, 2))*(2*self.support_uv_limit/self.grid)
        first = self._point(uv[:, 0], face)
        second = np.where((bc == 2)[:, None], self._point(uv[:, 1], (face+3) % 6), first)
        normal = FACE_NORMALS[load_faces[:, 0]]
        couple = np.cos(x[:, 9:10])*self.couple_basis[0]+np.sin(x[:, 9:10])*self.couple_basis[1]
        direction = np.where((family == 0)[:, None], -normal, normal)
        direction = np.where((family == 2)[:, None], SHEAR_DIRECTIONS[load_faces[:, 0]], direction)
        direction = np.where((family == 3)[:, None], couple, direction)
        amplitude = x[:, 8]*np.where(bc == 0, .25, 1.)
        force0 = amplitude[:, None]*direction
        force1 = np.where((family == 3)[:, None], -force0, 0.)
        motion = np.einsum('ni,nij->nj', np.stack([np.cos(x[:, 10]), np.sin(x[:, 10])], -1), FACE_BASES[face])*x[:, 11:12]
        motion = np.where((bc == 3)[:, None], motion, 0.)
        control = self._control(known_control)
        if control is not None:
            family = np.full(len(x), control[0], dtype=np.int64)
            force0 = np.broadcast_to(control[1], (len(x), 3))
            force1 = -force0 if control[0] == 3 else np.zeros_like(force0)
        physical = np.concatenate([x[:, :4], first, second,
            np.broadcast_to(self.handle_centers_m.reshape(1, 6), (len(x), 6)), force0, force1, motion], -1)
        return dict(physical=physical, boundary_type=np.asarray(bc).copy(), force_type=np.asarray(family).copy())

    def decode_torch(self, state, coordinates=None, known_control=None, *, device=None, dtype=None):
        """Differentiable continuous decode; categorical choices stay explicit.

        Optional coordinates tensor lets a future flow/SMC-gradient proposal
        differentiate the12D continuous chart. No25D density is invented.
        """
        import torch
        if not np.isfinite(self.log_prob(state)).all():
            raise ValueError('Categorical/template state outside prior')
        dtype = dtype or (coordinates.dtype if torch.is_tensor(coordinates) else torch.float64)
        device = device or (coordinates.device if torch.is_tensor(coordinates) else 'cpu')
        x = torch.as_tensor(state.coordinates if coordinates is None else coordinates, dtype=dtype, device=device)
        if x.shape != (len(state), 12):
            raise ValueError('Continuous coordinates must be[N,12]')
        lower, upper = (torch.as_tensor(a, dtype=dtype, device=device) for a in [self.lower, self.upper])
        if not bool((torch.isfinite(x).all(-1)&(x >= lower).all(-1)&
                     (x <= upper).all(-1)&(x[:, 9:11] < math.pi).all(-1)).all()):
            raise ValueError('Continuous coordinates outside the declared prior')
        tensor = lambda value: torch.as_tensor(value, device=device, dtype=dtype)
        integer = lambda value: torch.as_tensor(value, device=device, dtype=torch.long)
        bc, family, face = map(integer, [state.boundary_type, state.force_type, state.support_face])
        cells = integer(state.support_cells)
        axes = tensor(FACE_BASES); normals = tensor(FACE_NORMALS)
        cell_xy = torch.stack([cells//self.grid, cells % self.grid], -1)
        uv = -self.support_uv_limit+(cell_xy+x[:, 4:8].reshape(-1, 2, 2))*(2*self.support_uv_limit/self.grid)
        def point(tangent, f):
            return tensor(self.center)+normals[f]*tensor(self.size)/2+torch.einsum('ni,nij->nj', tangent, axes[f])*tensor(self.size)
        first = point(uv[:, 0], face)
        second = torch.where((bc == 2)[:, None], point(uv[:, 1], (face+3) % 6), first)
        normal = normals[integer(state.load_faces[:, 0])]
        couple = torch.cos(x[:, 9:10])*tensor(self.couple_basis[0])+torch.sin(x[:, 9:10])*tensor(self.couple_basis[1])
        direction = torch.where((family == 0)[:, None], -normal, normal)
        direction = torch.where((family == 2)[:, None], tensor(SHEAR_DIRECTIONS)[integer(state.load_faces[:, 0])], direction)
        direction = torch.where((family == 3)[:, None], couple, direction)
        amplitude = x[:, 8]*torch.where(bc == 0, .25, 1.)
        force0 = amplitude[:, None]*direction
        force1 = torch.where((family == 3)[:, None], -force0, 0.)
        motion = torch.einsum('ni,nij->nj', torch.stack([torch.cos(x[:, 10]), torch.sin(x[:, 10])], -1), axes[face])*x[:, 11:12]
        motion = torch.where((bc == 3)[:, None], motion, 0.)
        control = self._control(known_control)
        if control is not None:
            family = torch.full_like(family, control[0])
            force0 = tensor(control[1]).expand(len(x), 3)
            force1 = -force0 if control[0] == 3 else torch.zeros_like(force0)
        physical = torch.cat([x[:, :4], first, second,
            tensor(self.handle_centers_m).reshape(1, 6).expand(len(x), 6), force0, force1, motion], -1)
        return dict(physical=physical, boundary_type=bc, force_type=family)

    def log_prob_torch(self, state, coordinates=None):
        """Differentiable intrinsic log density with identical categorical mass."""
        import torch
        x = torch.as_tensor(state.coordinates) if coordinates is None else coordinates
        if not torch.is_tensor(x) or not x.is_floating_point() or x.shape != (len(state), 12):
            raise ValueError('Continuous coordinates must be[N,12]')
        # Remove template continuous contribution; replace it with candidate's
        # correct log-rho/log-drag terms and validity. Categorical zero mass is
        # preserved, including infeasible cells and nonincident load normals.
        midpoint = np.broadcast_to((self.lower+self.upper)/2, (len(state), 12))
        template = state.with_coordinates(midpoint)
        lp = torch.as_tensor(self.log_prob(template)-midpoint[:, 1]-midpoint[:, 3], dtype=x.dtype, device=x.device)
        lp = lp+x[:, 1]+x[:, 3]
        lower, upper = (torch.as_tensor(a, dtype=x.dtype, device=x.device) for a in [self.lower, self.upper])
        valid = torch.isfinite(x).all(-1)&(x >= lower).all(-1)&(x <= upper).all(-1)&(x[:, 9:11] < math.pi).all(-1)
        return torch.where(valid, lp, torch.full_like(lp, -torch.inf))

    def metadata(self):
        """Portable public conditioning and exact finite-prior specification."""
        spec = dict(prior='designed-known-hardware-conditional-physical-v01',
            qualification='Not exact seeded-generator conditional; not old learned joint flow',
            continuous_measure=list(INTRINSIC_NAMES), lower=self.lower.tolist(), upper=self.upper.tolist(),
            material_marginals=dict(E='loguniform[1500,20000]Pa', rho='uniform[400,850]kg/m3',
                nu='uniform[.28,.42]', damping='uniform[.35,.85]s^-1'),
            reference_points_sha256=hashlib.sha256(self.reference_points.tobytes()).hexdigest(),
            center_m=self.center.tolist(), size_m=self.size.tolist(), handle_centers_m=self.handle_centers_m.tolist(),
            grid=self.grid, support_uv_limit=self.support_uv_limit,
            support_handle_clearance_m=self.support_handle_clearance_m,
            surface_tolerance_m=self.surface_tolerance_m,
            public_hardware=self.hardware_descriptor(),
            incident_load_faces=[a.tolist() for a in self.incident_faces],
            feasible_faces={str(b): f.tolist() for b, f in self.feasible_faces.items()},
            feasible_cells={f'{b}/{f}': a.tolist() for (b, f), a in self.feasible_cells.items()},
            inactive_nuisances='Second support coordinates/cell excepttwo_patches; couple angle excepttwist; moving angle/amplitude exceptmoving_patch; original load variables forknowncontrols',
            motion_amplitude_qualification='Designed uncertainty6..10mm; simulator originallyfixed8mm',
            geometry_qualification=('Nearest1/9/16support nodes certified clear of all public attachment nodes; '
                '25D forward vector still omits this descriptor' if self.handle_patch_node_ids is not None else
                'Nearest1/9/16support nodes certified clear of public centers only; full attachment patch unidentified'),
            force_envelopes='ExistingIsaac3s reference-world families; no follower force or extra timing uncertainty')
        payload = json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()
        return dict(spec, prior_sha256=hashlib.sha256(payload).hexdigest())

    def hardware_descriptor(self):
        """Explicit extra public context, NOT silently inserted into25D inputs.

        The1g mass,3mm inertial floor and initial identity orientation match the
        declared worker conventions. A neural forward must accept this context
        separately to resolve the omitted-attachment ambiguity. If patches are
        unavailable, local offsets/inertia are intentionally not guessed.
        """
        result = dict(centers_m=self.handle_centers_m.tolist(), mass_kg_each=.001,
                      initial_orientation_wxyz=[1., 0., 0., 0.],
                      attachment_geometry_known=self.handle_patch_node_ids is not None,
                      qualification='Extra public context; not encoded in25D physical vector')
        if self.handle_patch_node_ids is not None:
            positions = self.reference_points[self.handle_patch_node_ids]
            offsets = positions-self.handle_centers_m[:, None]
            inertia = .001*np.mean(np.sum(offsets**2, axis=-1)[..., None]-offsets**2, axis=1)
            result.update(node_ids=self.handle_patch_node_ids.tolist(),
                reference_positions_m=positions.tolist(), local_offsets_m=offsets.tolist(),
                diagonal_inertia_kg_m2=np.maximum(inertia, .4*.001*.003**2).tolist(),
                inertia_convention='Worker diagonal approximation, not exact full inertia tensor')
        return result
