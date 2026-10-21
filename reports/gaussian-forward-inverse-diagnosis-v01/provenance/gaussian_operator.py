"""Direct neural trajectories and an invertible physical posterior.

The forward operator accepts geometry and physical coefficients, never an image
embedding. It queries arbitrary times without FEM, a stiffness solve, numerical
time stepping, or per-video optimization. Simulator supervision is a training
choice, not a runtime dependency. The initial implementation uses one reference
object; the interfaces alone do not establish cross-shape generalization.

Physical Gaussian kernels describe support/load fields. They are distinct from
appearance splats and from probability distributions over unknown parameters.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

BOUNDARY_TYPES = ('point_pin', 'fixed_patch', 'two_patches', 'moving_patch')
FORCE_TYPES = ('push_pulse', 'pull_hold', 'shear_oscillation', 'twist_couple')
PHYSICAL_DIM = 25
PHYSICAL_NAMES = (
    'log_youngs_modulus_Pa', 'log_density_kg_m3', 'poissons_ratio',
    'log_linear_damping_s_inverse',
    'support0_x_m', 'support0_y_m', 'support0_z_m',
    'support1_x_m', 'support1_y_m', 'support1_z_m',
    'load0_x_m', 'load0_y_m', 'load0_z_m',
    'load1_x_m', 'load1_y_m', 'load1_z_m',
    'force0_x_N', 'force0_y_N', 'force0_z_N',
    'force1_x_N', 'force1_y_N', 'force1_z_N',
    'moving_support_x_m', 'moving_support_y_m', 'moving_support_z_m',
)


def physical_vector(case: Mapping, spec: Mapping, points=None) -> np.ndarray:
    """Training/evaluation label extraction; never called by inference APIs.

    The four leading material entries contain natural logs where named. Load
    vectors are resultant Newtons at unit temporal envelope. Centers are metric
    reference coordinates. The inactive second support repeats the first and
    inactive force/motion entries are zero. Keeping those entries deterministic
    means their learned density is regularized, not a new independent physics
    quantity. A supplied reference point array determines load handle centers.
    """
    material = case['material']
    supports = np.asarray(spec['support_centers_m'], dtype=float)
    supports = np.stack([supports[0], supports[-1]])
    if points is None:
        # The known reference mesh is generated independently of any motion.
        from boundary_block import block_mesh
        points = block_mesh(cells=case.get('cells', [12, 6, 4]))[0]
    load_centers = np.asarray(points)[np.asarray(spec['load_nodes'], dtype=int)]
    forces = np.zeros((2, 3))
    kind = spec['force_type']
    if kind == 'push_pulse':
        forces[0] = -np.asarray(spec['normal_direction_world'])
    elif kind == 'pull_hold':
        forces[0] = np.asarray(spec['normal_direction_world'])
    elif kind == 'shear_oscillation':
        forces[0] = np.asarray(spec['shear_direction_world'])
    elif kind == 'twist_couple':
        forces[0] = np.asarray(spec['couple_direction_world'])
        forces[1] = -forces[0]
    else:
        raise ValueError(f'Unknown force family: {kind}')
    forces *= spec['peak_force_N']
    motion = spec['motion_amplitude_m'] * np.asarray(spec['motion_direction_world'])
    result = np.concatenate([
        [np.log(material['youngs_modulus_Pa']), np.log(material['density_kg_m3']),
         material['poissons_ratio'], np.log(material['linear_damping_s_inverse'])],
        supports.ravel(), load_centers.ravel(), forces.ravel(), motion,
    ]).astype(np.float32)
    if result.shape != (PHYSICAL_DIM,) or not np.isfinite(result).all():
        raise ValueError('Nonfinite or incorrectly sized physical label')
    return result



def canonicalize_physical(physical: Tensor, boundary_type: Tensor, force_type: Tensor) -> Tensor:
    """Remove category-inactive controls without changing the raw flow density.

    The 25D flow has padded nuisance coordinates so every category has the same
    dimension. The decoder ignores those nuisance values. In particular a twist
    is exactly a zero-resultant force couple, and a fixed support cannot move.
    This projection is not invertible and must NOT be inserted into a flow
    change-of-variables density calculation.
    """
    first_support = physical[..., 4:7]
    second_support = torch.where((boundary_type == 2)[..., None], physical[..., 7:10], first_support)
    first_force = physical[..., 16:19]
    second_force = torch.where((force_type == 3)[..., None], -first_force, torch.zeros_like(first_force))
    motion = torch.where((boundary_type == 3)[..., None], physical[..., 22:25],
                         torch.zeros_like(physical[..., 22:25]))
    return torch.cat([physical[..., :7], second_support, physical[..., 10:19], second_force, motion], -1)


def category_labels(case: Mapping) -> tuple[int, int]:
    """Training/evaluation labels, separate from all observation encoders."""
    return (BOUNDARY_TYPES.index(case['boundary_type']),
            FORCE_TYPES.index(case['force_type']))


class PhysicalParameterTransform(nn.Module):
    """Bijective logistic box transform of a 25D physical coefficient vector.

    Material positivity is supplied by exp(logE), exp(logrho), exp(logdrag).
    Bounded nu remains below incompressibility. Bounds are declared priors, not
    learned confidence intervals. Coordinate bounds include a 10% margin around
    the reference block. Flow densities include this transform's Jacobian.
    """
    def __init__(self, center=(0., 0., .25), size=(.24, .12, .08)):
        super().__init__()
        center, size = np.asarray(center), np.asarray(size)
        spatial_low, spatial_high = center - .55*size, center + .55*size
        low = np.r_[[math.log(300.), math.log(100.), .15, math.log(.05)],
                    np.tile(spatial_low, 4), np.full(6, -.5), np.full(3, -.03)]
        high = np.r_[[math.log(100000.), math.log(1500.), .485, math.log(5.)],
                     np.tile(spatial_high, 4), np.full(6, .5), np.full(3, .03)]
        self.register_buffer('lower', torch.tensor(low, dtype=torch.float32))
        self.register_buffer('upper', torch.tensor(high, dtype=torch.float32))

    def unconstrained_to_physical(self, unconstrained: Tensor) -> Tensor:
        return self.lower + (self.upper-self.lower)*torch.sigmoid(unconstrained)

    def physical_to_unconstrained(self, physical: Tensor, strict: bool = True) -> Tensor:
        valid = self.contains(physical)
        if strict and not bool(valid.all()):
            raise ValueError('Physical coefficients must lie strictly inside declared bounds')
        fraction = (physical-self.lower)/(self.upper-self.lower)
        eps = torch.finfo(fraction.dtype).eps
        return torch.logit(fraction.clamp(eps, 1-eps))

    def contains(self, physical: Tensor) -> Tensor:
        return (torch.isfinite(physical).all(-1) & (physical > self.lower).all(-1)
                & (physical < self.upper).all(-1))

    def log_abs_det_jacobian(self, unconstrained: Tensor) -> Tensor:
        return (torch.log(self.upper-self.lower) + F.logsigmoid(unconstrained)
                + F.logsigmoid(-unconstrained)).sum(-1)

    def normalize(self, physical: Tensor) -> Tensor:
        """Stable coefficient scales for the trajectory decoder (not logits)."""
        return 2*(physical-self.lower)/(self.upper-self.lower)-1

    @staticmethod
    def material_properties(physical: Tensor) -> dict[str, Tensor]:
        young, density, nu, drag = physical[..., 0].exp(), physical[..., 1].exp(), physical[..., 2], physical[..., 3].exp()
        return {'youngs_modulus_Pa': young, 'density_kg_m3': density,
                'poissons_ratio': nu, 'linear_damping_s_inverse': drag,
                'shear_modulus_Pa': young/(2*(1+nu)),
                'bulk_modulus_Pa': young/(3*(1-2*nu))}


def _batch_points(points: Tensor, batch: int) -> Tensor:
    if points.ndim == 2:
        return points.unsqueeze(0).expand(batch, -1, -1)
    if points.ndim != 3 or points.shape[0] != batch or points.shape[-1] != 3:
        raise ValueError('points must be [N,3] or [B,N,3]')
    return points


def _batch_times(times: Tensor, batch: int) -> Tensor:
    if times.ndim == 0:
        times = times.reshape(1)
    if times.ndim == 1:
        return times.unsqueeze(0).expand(batch, -1)
    if times.ndim != 2 or times.shape[0] != batch:
        raise ValueError('times must be [T] or [B,T]')
    return times


def force_envelope(force_type: Tensor, times: Tensor, seconds: float = 3.) -> Tensor:
    """Declared differentiable world-load families, shape [B,T]."""
    t = _batch_times(times, len(force_type))/seconds
    pulse_phase = ((t-.1)/.45).clamp(0, 1)
    pulse = torch.sin(math.pi*pulse_phase).square()
    pull = (torch.sin(math.pi/2*((t-.1)/.2).clamp(0, 1)).square()
            * torch.cos(math.pi/2*((t-.65)/.25).clamp(0, 1)).square())
    shear_phase = ((t-.1)/.8).clamp(0, 1)
    shear = torch.sin(4*math.pi*shear_phase)*torch.sin(math.pi*shear_phase).square()
    return torch.where(force_type[:, None] == 1, pull,
                       torch.where(force_type[:, None] == 2, shear, pulse))


class GaussianKernelFields(nn.Module):
    """Reference-space support/load kernels evaluable at arbitrary 3D points.

    Raw kernels are dimensionless local features. ``traction`` additionally
    requires actual surface area quadrature weights to normalize each load
    kernel, yielding N/m^2 and preserving the supplied resultant Newtons on that
    quadrature. Normalizing over arbitrary query points would be incorrect.
    Finite Gaussian supports approximate the dataset's exact nodal constraints.
    """
    def __init__(self, load_width_m: float = .025):
        super().__init__()
        self.load_width_m = float(load_width_m)
        self.register_buffer('support_widths_m', torch.tensor([.010, .035, .030, .035]))

    def forward(self, points: Tensor, physical: Tensor, boundary_type: Tensor) -> dict[str, Tensor]:
        points = _batch_points(points, len(physical))
        support_centers = physical[:, 4:10].reshape(-1, 2, 3)
        load_centers = physical[:, 10:16].reshape(-1, 2, 3)
        width = self.support_widths_m[boundary_type]
        support = torch.exp(-.5*((points[:, :, None]-support_centers[:, None])
                                 /width[:, None, None, None]).square().sum(-1))
        active = torch.stack([torch.ones_like(boundary_type), (boundary_type == 2).long()], -1)
        support = support*active[:, None]
        load = torch.exp(-.5*((points[:, :, None]-load_centers[:, None])
                              /self.load_width_m).square().sum(-1))
        force = physical[:, 16:22].reshape(-1, 2, 3)
        return {'support': support, 'load': load,
                'force_features_N': torch.einsum('bnk,bkc->bnc', load, force),
                'moving_anchor_features_m': support[..., :1]*physical[:, None, 22:25]}

    def traction(self, points: Tensor, physical: Tensor, boundary_type: Tensor,
                 force_type: Tensor, times: Tensor, area_weights: Tensor,
                 seconds: float = 3.) -> Tensor:
        physical = canonicalize_physical(physical, boundary_type, force_type)
        fields = self(points, physical, boundary_type)
        if area_weights.ndim == 1:
            area_weights = area_weights[None].expand(len(physical), -1)
        if area_weights.shape != fields['load'].shape[:2] or not bool((area_weights > 0).all()):
            raise ValueError('Positive surface area quadrature weights [N] or [B,N] required')
        integral = (fields['load']*area_weights[..., None]).sum(1)
        if bool((integral < 1e-20).any()):
            raise ValueError('Quadrature misses a load kernel; cannot normalize resultant')
        weights = fields['load']/integral[:, None]
        force = physical[:, 16:22].reshape(-1, 2, 3)
        spatial = torch.einsum('bnk,bkc->bnc', weights, force)
        return spatial[:, None]*force_envelope(force_type, times, seconds)[..., None, None]


def _mlp(input_dim: int, output_dim: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers += [nn.Linear(input_dim, hidden), nn.SiLU()]
        input_dim = hidden
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class NeuralTrajectoryOperator(nn.Module):
    """Factorized smooth coordinate operator conditioned on physical inputs.

    Spatial and temporal MLPs form a rank-R displacement field. Time queries
    are parallel; there is no autoregressive state or numerical mechanics loop.
    ``points`` [N,3]/[B,N,3], ``times`` [T]/[B,T], ``physical`` [B,25], and two
    categorical int64 tensors [B] produce positions [B,T,N,3]. Derivatives are
    available through ordinary automatic differentiation. The t^2 factor gives
    the prescribed reference state and zero initial velocity exactly.

    Query points are independent given a fixed geometry context. When querying
    only a subset of points, pass the same ``geometry_reference`` or retain the
    constructor's reference; do not recompute shape context from visible points.
    """
    def __init__(self, hidden: int = 128, rank: int = 24, depth: int = 3,
                 seconds: float = 3., displacement_scale_m: float = .03,
                 reference_points=None):
        super().__init__()
        self.rank, self.seconds, self.displacement_scale_m = rank, float(seconds), float(displacement_scale_m)
        self.transform = PhysicalParameterTransform()
        self.fields = GaussianKernelFields()
        self.register_buffer('reference_center', torch.tensor([0., 0., .25]))
        self.register_buffer('reference_size', torch.tensor([.24, .12, .08]))
        if reference_points is None:
            reference_points = np.array([[x, y, z] for x in [-.12, .12]
                                         for y in [-.06, .06] for z in [.21, .29]])
        reference_points = torch.as_tensor(np.asarray(reference_points), dtype=torch.float32)
        self.register_buffer('geometry_reference', reference_points)
        self.register_buffer('spatial_frequencies', torch.tensor([1., 2., 4.]))
        self.register_buffer('temporal_frequencies', torch.tensor([1., 2., 4., 8.]))
        context_dim = PHYSICAL_DIM+8+12
        # x + sin/cos(x,3freqs) + two support/two load kernels + load/motion vecs.
        self.spatial = _mlp(context_dim+3+18+10, rank*3, hidden, depth)
        self.temporal = _mlp(context_dim+1+8+2, rank, hidden, depth)
        nn.init.normal_(self.spatial[-1].weight, std=.025)
        nn.init.zeros_(self.spatial[-1].bias)
        nn.init.normal_(self.temporal[-1].weight, std=.025)
        nn.init.zeros_(self.temporal[-1].bias)

    def _context(self, physical, boundary_type, force_type, geometry_reference=None):
        geometry = self.geometry_reference if geometry_reference is None else geometry_reference
        geometry = _batch_points(geometry, len(physical))
        normalized = (geometry-self.reference_center)/self.reference_size
        shape = torch.cat([normalized.amin(1), normalized.amax(1), normalized.mean(1),
                           normalized.std(1, correction=0)], -1)
        return torch.cat([self.transform.normalize(physical),
                          F.one_hot(boundary_type, 4).to(physical),
                          F.one_hot(force_type, 4).to(physical), shape], -1)

    def forward(self, points: Tensor, times: Tensor, physical: Tensor,
                boundary_type: Tensor, force_type: Tensor, geometry_reference=None) -> Tensor:
        if physical.ndim != 2 or physical.shape[1] != PHYSICAL_DIM:
            raise ValueError('physical must be [B,25]')
        batch = len(physical)
        physical = canonicalize_physical(physical, boundary_type, force_type)
        points, times = _batch_points(points, batch), _batch_times(times, batch)
        context = self._context(physical, boundary_type, force_type, geometry_reference)
        x = (points-self.reference_center)/self.reference_size
        phase = math.pi*x[..., None]*self.spatial_frequencies
        fields = self.fields(points, physical, boundary_type)
        spatial_input = torch.cat([x, phase.sin().flatten(-2), phase.cos().flatten(-2),
                                   fields['support'], fields['load'],
                                   fields['force_features_N']/.2,
                                   fields['moving_anchor_features_m']/.008,
                                   context[:, None].expand(-1, points.shape[1], -1)], -1)
        spatial = self.spatial(spatial_input).reshape(batch, points.shape[1], self.rank, 3)
        tau = times/self.seconds
        phase_t = math.pi*tau[..., None]*self.temporal_frequencies
        temporal_input = torch.cat([tau[..., None], phase_t.sin(), phase_t.cos(),
                                    force_envelope(force_type, times, self.seconds)[..., None],
                                    torch.sin(math.pi*tau).square()[..., None],
                                    context[:, None].expand(-1, times.shape[1], -1)], -1)
        temporal = self.temporal(temporal_input)
        displacement = torch.einsum('bnrc,btr->btnc', spatial, temporal)/math.sqrt(self.rank)
        return points[:, None] + self.displacement_scale_m*tau[..., None, None].square()*displacement

    def spatial_jacobian(self, points: Tensor, time: Tensor, physical: Tensor,
                         boundary_type: Tensor, force_type: Tensor,
                         create_graph: bool = False) -> tuple[Tensor, Tensor]:
        """F=d chi/d X at ONE query time per batch, plus det(F).

        This derivative evaluates only a local coordinate field, not FEM. The
        caller selects sparse physical collocation points for derivative losses.
        """
        with torch.enable_grad():
            points = _batch_points(points, len(physical)).clone().requires_grad_(True)
            time = torch.as_tensor(time, dtype=points.dtype, device=points.device)
            if time.ndim == 0:
                time = time.expand(len(physical))
            if time.shape != (len(physical),):
                raise ValueError('time must be scalar or [B] for spatial_jacobian')
            positions = self(points, time[:, None], physical, boundary_type, force_type)[:, 0]
            gradients = [torch.autograd.grad(positions[..., axis].sum(), points,
                                             create_graph=create_graph, retain_graph=True)[0]
                         for axis in range(3)]
            jacobian = torch.stack(gradients, -2)
        return jacobian, torch.linalg.det(jacobian)


class NeuralGreenOperator(NeuralTrajectoryOperator):
    """Learned Green-response approximation, exactly linear in load amplitudes.

    The material/support/geometry-conditioned response is neural and nonlinear
    in those inputs; applied force vectors enter ONLY through a final linear
    contraction. A separate learned response multiplies prescribed support
    motion. There is no image input or numerical mechanics solve. Superposition
    is an approximation to the finite-deformation Isaac mechanics and must be
    assessed on held-out force amplitudes/directions.

    Optional ``controls`` is a list of dicts with ``center_m`` [3]/[B,3],
    ``force_N`` [3]/[B,3], ``force_type`` int/[B], and optional ``start_s`` float.
    A control uses the same learned force-family response, not an untrained
    extra head. A start time translates the trained envelope; duration scaling
    is deliberately unsupported. Original forces and a known control add.
    Known reset probes require zeroing the original force entries first.

    Two original handle centers remain in mechanical context even at zero
    load, since the simulator's rigid attached patches add inertia/rigidity.
    New contact hardware is not modeled by merely supplying another control.
    """
    def __init__(self, hidden: int = 128, rank: int = 24, depth: int = 3,
                 seconds: float = 3., displacement_scale_m: float = .03,
                 reference_points=None, force_scale_N: float = .2,
                 motion_scale_m: float = .008):
        super().__init__(hidden, rank, depth, seconds, displacement_scale_m, reference_points)
        self.force_scale_N, self.motion_scale_m = float(force_scale_N), float(motion_scale_m)
        # First16 coefficients exclude EVERY load and motion amplitude.
        # bc/family8 + query-center3 + shape12 + force/moving-support mode2.
        context_dim = 16+8+3+12+2
        self.spatial = _mlp(context_dim+3+18+2+1, rank*3, hidden, depth)
        # Temporal output has three input-force components for every rank.
        self.temporal = _mlp(context_dim+1+8+2, rank*3, hidden, depth)
        nn.init.normal_(self.spatial[-1].weight, std=.025)
        nn.init.zeros_(self.spatial[-1].bias)
        nn.init.normal_(self.temporal[-1].weight, std=.025)
        nn.init.zeros_(self.temporal[-1].bias)

    def _response(self, points, times, physical, boundary_type, force_type,
                  center, amplitude, mode, geometry_reference=None):
        batch, count = points.shape[:2]
        geometry = self.geometry_reference if geometry_reference is None else geometry_reference
        geometry = _batch_points(geometry, batch)
        normalized_geometry = (geometry-self.reference_center)/self.reference_size
        shape = torch.cat([normalized_geometry.amin(1), normalized_geometry.amax(1),
                           normalized_geometry.mean(1), normalized_geometry.std(1, correction=0)], -1)
        center_normalized = (center-self.reference_center)/self.reference_size
        modes = torch.full((batch,), mode, dtype=torch.long, device=points.device)
        context = torch.cat([self.transform.normalize(physical)[:, :16],
                             F.one_hot(boundary_type, 4).to(physical),
                             F.one_hot(force_type, 4).to(physical), center_normalized, shape,
                             F.one_hot(modes, 2).to(physical)], -1)
        x = (points-self.reference_center)/self.reference_size
        spatial_phase = math.pi*x[..., None]*self.spatial_frequencies
        support = self.fields(points, physical, boundary_type)['support']
        load_kernel = torch.exp(-.5*((points-center[:, None])/self.fields.load_width_m).square().sum(-1, keepdim=True))
        spatial_input = torch.cat([x, spatial_phase.sin().flatten(-2), spatial_phase.cos().flatten(-2),
                                   support, load_kernel, context[:, None].expand(-1, count, -1)], -1)
        spatial = self.spatial(spatial_input).reshape(batch, count, self.rank, 3)
        tau = times/self.seconds
        phase = math.pi*tau[..., None]*self.temporal_frequencies
        temporal_input = torch.cat([tau[..., None], phase.sin(), phase.cos(),
                                    force_envelope(force_type, times, self.seconds)[..., None],
                                    torch.sin(math.pi*tau).square()[..., None],
                                    context[:, None].expand(-1, times.shape[1], -1)], -1)
        temporal = self.temporal(temporal_input).reshape(batch, times.shape[1], self.rank, 3)
        scale = self.force_scale_N if mode == 0 else self.motion_scale_m
        if amplitude is None:
            # Direct unit-response matrix avoids repeated forward calls and
            # cancellation against the reference position when forming a basis.
            result = torch.einsum('bnrc,btri->btnci', spatial, temporal)/(scale*math.sqrt(self.rank))
            return self.displacement_scale_m*tau[..., None, None, None].square()*result
        coefficients = (temporal*(amplitude/scale)[:, None, None]).sum(-1)
        result = torch.einsum('bnrc,btr->btnc', spatial, coefficients)/math.sqrt(self.rank)
        return self.displacement_scale_m*tau[..., None, None].square()*result

    def forward(self, points: Tensor, times: Tensor, physical: Tensor,
                boundary_type: Tensor, force_type: Tensor, geometry_reference=None,
                controls=None) -> Tensor:
        if physical.ndim != 2 or physical.shape[1] != PHYSICAL_DIM:
            raise ValueError('physical must be [B,25]')
        batch = len(physical)
        points, times = _batch_points(points, batch), _batch_times(times, batch)
        physical = canonicalize_physical(physical, boundary_type, force_type)
        displacement = points.new_zeros(batch, times.shape[1], points.shape[1], 3)
        # Both handles are evaluated even if an amplitude is zero: stable graph,
        # exactly linear derivatives at zero, and no data-dependent dispatch.
        for handle in range(2):
            displacement = displacement+self._response(
                points, times, physical, boundary_type, force_type,
                physical[:, 10+3*handle:13+3*handle],
                physical[:, 16+3*handle:19+3*handle], 0, geometry_reference)
        displacement = displacement+self._response(
            points, times, physical, boundary_type, torch.zeros_like(force_type),
            physical[:, 4:7], physical[:, 22:25], 1, geometry_reference)
        for control in controls or []:
            if 'seconds' in control and float(control['seconds']) != self.seconds:
                raise ValueError('Untrained control duration rescaling is unsupported')
            center = torch.as_tensor(control['center_m'], dtype=points.dtype, device=points.device)
            amplitude = torch.as_tensor(control['force_N'], dtype=points.dtype, device=points.device)
            if center.ndim == 1:
                center = center[None].expand(batch, -1)
            if amplitude.ndim == 1:
                amplitude = amplitude[None].expand(batch, -1)
            if center.shape != (batch, 3) or amplitude.shape != (batch, 3):
                raise ValueError('Control centers and forces must be [3] or [B,3]')
            family = torch.as_tensor(control['force_type'], dtype=torch.long, device=points.device)
            if family.ndim == 0:
                family = family.expand(batch)
            if family.shape != (batch,):
                raise ValueError('Control force_type must be scalar or [B]')
            start = torch.as_tensor(control.get('start_s', 0.), dtype=times.dtype, device=times.device)
            if start.ndim == 1:
                start = start[:, None]
            shifted = (times-start).clamp_min(0)
            displacement = displacement+self._response(
                points, shifted, physical, boundary_type, family, center, amplitude, 0, geometry_reference)
        return points[:, None]+displacement

    def load_response_basis(self, points: Tensor, times: Tensor, physical: Tensor,
                            boundary_type: Tensor, force_type: Tensor) -> tuple[Tensor, Tensor]:
        """Return offset and exact neural load design [B,T,N,3,6] per Newton.

        The offset retains prescribed support motion; basis columns correspond
        to the RAW first/second force vectors before categorical projection.
        A twist has three zero inactive columns and force0's effect includes
        the opposite force at handle1. This design enables exact Gaussian load
        integration for a separately LINEAR observation model. Gaussian image
        rendering/depth visibility still requires approximation.
        """
        batch = len(physical)
        points, times = _batch_points(points, batch), _batch_times(times, batch)
        physical = canonicalize_physical(physical, boundary_type, force_type)
        offset = points[:, None]+self._response(
            points, times, physical, boundary_type, torch.zeros_like(force_type),
            physical[:, 4:7], physical[:, 22:25], 1)
        first = self._response(points, times, physical, boundary_type, force_type,
                               physical[:, 10:13], None, 0)
        second = self._response(points, times, physical, boundary_type, force_type,
                                physical[:, 13:16], None, 0)
        first = first-torch.where((force_type == 3)[:, None, None, None, None],
                                   second, torch.zeros_like(second))
        return offset, torch.cat([first, torch.zeros_like(first)], -1)


def tetrahedral_deformation(positions: Tensor, rest_points: Tensor,
                            tetrahedra: Tensor) -> tuple[Tensor, Tensor]:
    """Piecewise-affine F and J for predicted mesh; no mechanics solve.

    positions may have arbitrary leading dimensions [...,N,3]. Negative J is a
    detected model failure, not silently repaired. Gaussian covariance can be
    advected by F Sigma F^T only after appropriate binding to these elements.
    """
    reference = rest_points[tetrahedra]
    rest_edges = (reference[:, 1:]-reference[:, :1]).transpose(-1, -2)
    if bool((torch.linalg.det(rest_edges).abs() < 1e-14).any()):
        raise ValueError('Reference contains a degenerate tetrahedron')
    deformed = positions[..., tetrahedra, :]
    edges = (deformed[..., 1:, :]-deformed[..., :1, :]).transpose(-1, -2)
    gradient = edges @ torch.linalg.inv(rest_edges)
    return gradient, torch.linalg.det(gradient)


def deformation_diagnostics(positions: Tensor, rest_points: Tensor, tetrahedra: Tensor) -> dict:
    """Detached finite/J audit suitable for evaluation JSON; not a guarantee."""
    with torch.no_grad():
        finite = bool(torch.isfinite(positions).all())
        if not finite:
            return {'finite': False, 'positive_volume': False,
                    'minimum_volume_ratio': None, 'inverted_tetrahedron_count': None}
        _, determinant = tetrahedral_deformation(positions, rest_points, tetrahedra)
        return {'finite': True, 'positive_volume': bool((determinant > 0).all()),
                'minimum_volume_ratio': float(determinant.min()),
                'inverted_tetrahedron_count': int((determinant <= 0).sum())}


class _AffineCoupling(nn.Module):
    def __init__(self, dimension: int, context_dim: int, mask: Tensor, hidden: int):
        super().__init__()
        self.register_buffer('mask', mask)
        self.conditioner = _mlp(dimension+context_dim, 2*dimension, hidden, 2)
        # Identity initialization gives a stable standard-normal initial density.
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def _affine_parameters(self, value: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        shift, scale = self.conditioner(torch.cat([value*self.mask, context], -1)).chunk(2, -1)
        active = 1-self.mask
        return shift*active, 1.5*torch.tanh(scale/1.5)*active

    def forward(self, value: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        shift, log_scale = self._affine_parameters(value, context)
        return value*self.mask + (1-self.mask)*(value*log_scale.exp()+shift), log_scale.sum(-1)

    def inverse(self, value: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        shift, log_scale = self._affine_parameters(value, context)
        return value*self.mask + (1-self.mask)*(value-shift)*(-log_scale).exp(), -log_scale.sum(-1)


class ConditionalRealNVP(nn.Module):
    """Conditional invertible density; forward maps coefficients to base noise."""
    def __init__(self, context_dim: int, dimension: int = PHYSICAL_DIM,
                 layers: int = 8, hidden: int = 128):
        super().__init__()
        self.dimension, self.context_dim = dimension, context_dim
        self.layers = nn.ModuleList([
            _AffineCoupling(dimension, context_dim,
                            ((torch.arange(dimension)+i) % 2).float(), hidden)
            for i in range(layers)])

    def forward(self, value: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        logdet = value.new_zeros(value.shape[:-1])
        for layer in self.layers:
            value, delta = layer(value, context)
            logdet = logdet+delta
        return value, logdet

    def inverse(self, noise: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        logdet = noise.new_zeros(noise.shape[:-1])
        for layer in reversed(self.layers):
            noise, delta = layer.inverse(noise, context)
            logdet = logdet+delta
        return noise, logdet

    def log_prob(self, value: Tensor, context: Tensor) -> Tensor:
        noise, logdet = self(value, context)
        return -.5*(noise.square()+math.log(2*math.pi)).sum(-1)+logdet


class PhysicalPosterior(nn.Module):
    """Amortized joint posterior over physical coefficients and two categories.

    ``context`` is produced from public observations by an external encoder.
    No file access or private truth enters this class. Continuous flow layers
    additionally condition on the sampled boundary/load types. Density is with
    respect to the declared 25 coefficients (including log-material entries),
    and includes bounded-transform plus categorical probability terms.
    """
    def __init__(self, context_dim: int, hidden: int = 128, layers: int = 8):
        super().__init__()
        self.context_dim = context_dim
        self.transform = PhysicalParameterTransform()
        self.boundary_head = _mlp(context_dim, 4, hidden, 1)
        self.force_head = _mlp(context_dim, 4, hidden, 1)
        self.flow = ConditionalRealNVP(context_dim+8, layers=layers, hidden=hidden)

    def categorical_logits(self, context: Tensor) -> tuple[Tensor, Tensor]:
        return self.boundary_head(context), self.force_head(context)

    @staticmethod
    def _condition(context: Tensor, boundary_type: Tensor, force_type: Tensor) -> Tensor:
        return torch.cat([context, F.one_hot(boundary_type, 4).to(context),
                          F.one_hot(force_type, 4).to(context)], -1)

    def log_prob(self, physical: Tensor, context: Tensor,
                 boundary_type: Tensor, force_type: Tensor,
                 include_categorical: bool = True) -> Tensor:
        if physical.shape[:-1] != context.shape[:-1]:
            raise ValueError('Physical and context leading dimensions must match')
        valid = self.transform.contains(physical)
        xi = self.transform.physical_to_unconstrained(physical, strict=False)
        result = (self.flow.log_prob(xi, self._condition(context, boundary_type, force_type))
                  - self.transform.log_abs_det_jacobian(xi))
        if include_categorical:
            boundary_logits, force_logits = self.categorical_logits(context)
            result = (result + boundary_logits.log_softmax(-1).gather(-1, boundary_type[..., None]).squeeze(-1)
                      + force_logits.log_softmax(-1).gather(-1, force_type[..., None]).squeeze(-1))
        return torch.where(valid, result, torch.full_like(result, -torch.inf))

    def sample(self, context: Tensor, samples: int = 32, boundary_type: Tensor | None = None,
               force_type: Tensor | None = None, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        """Batched hypotheses [B,K,25]; supplied categories are a known-type condition.

        Sampling is differentiable through continuous values, not categorical
        draws. For supervised flow NLL use ``log_prob``; categorical inference
        can be enumerated if a differentiable discrete expectation is needed.
        """
        if context.ndim != 2 or samples < 1:
            raise ValueError('Expected context [B,C] and positive sample count')
        batch = len(context)
        boundary_logits, force_logits = self.categorical_logits(context)
        if boundary_type is None:
            boundary_type = torch.multinomial(boundary_logits.softmax(-1), samples,
                                              replacement=True, generator=generator)
        else:
            boundary_type = boundary_type[:, None].expand(-1, samples)
        if force_type is None:
            force_type = torch.multinomial(force_logits.softmax(-1), samples,
                                           replacement=True, generator=generator)
        else:
            force_type = force_type[:, None].expand(-1, samples)
        expanded_context = context[:, None].expand(-1, samples, -1)
        condition = self._condition(expanded_context, boundary_type, force_type)
        noise = torch.randn(batch, samples, PHYSICAL_DIM, dtype=context.dtype,
                            device=context.device, generator=generator)
        xi, _ = self.flow.inverse(noise, condition)
        physical = self.transform.unconstrained_to_physical(xi)
        return {'physical': physical,
                'effective_physical': canonicalize_physical(physical, boundary_type, force_type),
                'boundary_type': boundary_type, 'force_type': force_type,
                'log_prob': self.log_prob(physical, expanded_context, boundary_type, force_type),
                'unconstrained': xi}
