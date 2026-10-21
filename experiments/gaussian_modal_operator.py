"""Train-only modal direct-time operators for the fixed known reference object.

A randomized SVD compresses *training* displacement trajectories. Metric RBF
interpolation weights are fitted offline, so arbitrary reference-point queries
need no solve. A neural physical/time decoder predicts modal coefficients. No
FEM, stiffness assembly, image-conditioned forward shortcut, or autoregressive
integration enters inference. This fixed-object surrogate is not evidence for
cross-shape generalization. Green superposition remains a physical approximation.
"""
from __future__ import annotations
import math
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gaussian_operator import (PHYSICAL_DIM, PhysicalParameterTransform,
                               canonicalize_physical, force_envelope)


def fit_modal_basis(rest_points, records, rank=32, width_m=.025, ridge=1e-6,
                    case_scale_floor_m=.0004, seed=9123, oversample=12,
                    power_iterations=2):
    """Offline randomized SVD and RBF interpolation, strictly TRAIN labels only.

    Each record must explicitly supply split='train', episode_id, and
    positions_m[T,N,3]. Its displacement matrix is divided by its coordinate RMS
    (with the declared .4mm floor), preventing large rotations from dominating
    every deformation mode. Returned modes have unit coordinate RMS, so modal
    coefficients retain metre units. All train IDs/scales are saved as metadata.
    """
    rest = np.asarray(rest_points, dtype=np.float64)
    if rest.ndim != 2 or rest.shape[1] != 3 or not np.isfinite(rest).all():
        raise ValueError('Finite rest points [N,3] required')
    if rank < 1 or width_m <= 0 or ridge <= 0 or case_scale_floor_m <= 0:
        raise ValueError('Positive rank, RBF width/ridge and case scale floor required')
    matrices, metadata = [], []
    for record in records:
        if record.get('split') != 'train':
            raise ValueError('Modal basis fitting accepts explicitly TRAIN records only')
        positions = np.asarray(record['positions_m'], dtype=np.float64)
        if positions.ndim != 3 or positions.shape[1:] != rest.shape or not np.isfinite(positions).all():
            raise ValueError('Finite training positions [T,N,3] matching rest mesh required')
        displacement = positions-rest
        scale = max(float(np.sqrt(np.mean(displacement**2))), case_scale_floor_m)
        matrices.append(displacement.reshape(len(positions), -1)/scale)
        metadata.append({'episode_id': str(record['episode_id']), 'split': 'train',
                         'frames': len(positions), 'normalization_scale_m': scale})
    if not matrices:
        raise ValueError('At least one declared training trajectory required')
    matrix = np.concatenate(matrices)
    total_energy = float(np.sum(matrix**2))
    if total_energy <= 1e-20:
        raise ValueError('Training trajectories contain no displacement energy')
    rank = min(int(rank), min(matrix.shape))
    components = min(rank+int(oversample), min(matrix.shape))
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(matrix@rng.normal(size=(matrix.shape[1], components)), mode='reduced')
    for _ in range(power_iterations):
        right, _ = np.linalg.qr(matrix.T@q, mode='reduced')
        q, _ = np.linalg.qr(matrix@right, mode='reduced')
    _, singular, vt = np.linalg.svd(q.T@matrix, full_matrices=False)
    # Standardize mode RMS rather than baking unknown physical amplitudes into it.
    modes = (vt[:rank]*math.sqrt(matrix.shape[1])).reshape(rank, len(rest), 3)
    distance = np.sum((rest[:, None]-rest[None])**2, axis=-1)
    kernel = np.exp(-.5*distance/width_m**2)
    rhs = modes.transpose(1, 0, 2).reshape(len(rest), rank*3)
    weights = np.linalg.solve(kernel+ridge*np.eye(len(rest)), rhs).reshape(len(rest), rank, 3)
    interpolation_error = float(np.linalg.norm(kernel@weights.reshape(len(rest), -1)-rhs)/np.linalg.norm(rhs))
    info = {'qualification': 'Single known reference geometry; training-only displacement basis',
            'training_records': metadata, 'seed': seed, 'rank': rank,
            'case_scale_floor_m': case_scale_floor_m, 'width_m': width_m, 'ridge': ridge,
            'randomized_svd_power_iterations': power_iterations,
            'weighted_captured_energy_fraction': float(np.sum(singular[:rank]**2)/total_energy),
            'rbf_node_relative_fit_error': interpolation_error,
            'mode_coordinate_rms': 1.}
    return {'rest_points_m': rest.astype(np.float32), 'modes': modes.astype(np.float32),
            'rbf_weights': weights.astype(np.float32), 'width_m': float(width_m), 'metadata': info}


class RBFModalBasis(nn.Module):
    """Fixed spatial modes; all linear algebra fitting happened offline."""
    def __init__(self, basis: Mapping):
        super().__init__()
        for name in ['rest_points_m', 'modes', 'rbf_weights']:
            value = torch.as_tensor(basis[name], dtype=torch.float32).detach().clone()
            self.register_buffer(name, value)
        self.register_buffer('width_m', torch.tensor(float(basis['width_m'])))
        self.metadata = dict(basis.get('metadata', {}))
        rank, nodes, dimensions = self.modes.shape
        if dimensions != 3 or self.rest_points_m.shape != (nodes, 3) or self.rbf_weights.shape != (nodes, rank, 3):
            raise ValueError('Modal basis shapes disagree')

    @property
    def rank(self):
        return self.modes.shape[0]

    def get_extra_state(self):
        return self.metadata

    def set_extra_state(self, state):
        self.metadata = state

    def forward(self, points: Tensor) -> Tensor:
        """[N,3]->[N,R,3], or batched [B,N,3]->[B,N,R,3].

        Known-node queries use the exact SVD modes. Arbitrary queries interpolate
        them with the fixed metric Gaussian basis; gradients with respect to X
        use the differentiable RBF field, including at known nodes.
        """
        if points.ndim not in (2, 3) or points.shape[-1] != 3:
            raise ValueError('Reference query points must be [N,3] or [B,N,3]')
        if not points.requires_grad and points.shape[-2:] == self.rest_points_m.shape:
            if points.ndim == 2 and torch.equal(points, self.rest_points_m):
                return self.modes.permute(1, 0, 2)
            if points.ndim == 3 and torch.equal(points, self.rest_points_m.expand_as(points)):
                return self.modes.permute(1, 0, 2).expand(len(points), -1, -1, -1)
        distance = (points[..., :, None, :]-self.rest_points_m).square().sum(-1)
        kernel = torch.exp(-.5*distance/self.width_m.square())
        return torch.einsum('...nk,krc->...nrc', kernel, self.rbf_weights)

    def project_displacements(self, displacement: Tensor) -> Tensor:
        """Orthogonal training-basis coefficients [..,N,3]->[..,R], in metres."""
        if displacement.shape[-2:] != self.rest_points_m.shape:
            raise ValueError('Projection requires the known reference nodes')
        return torch.einsum('...nc,rnc->...r', displacement, self.modes)/self.rest_points_m.numel()


def _network(input_dim, output_dim, hidden, depth):
    modules = []
    for _ in range(depth):
        modules += [nn.Linear(input_dim, hidden), nn.SiLU()]; input_dim = hidden
    modules.append(nn.Linear(input_dim, output_dim))
    nn.init.normal_(modules[-1].weight, std=.02); nn.init.zeros_(modules[-1].bias)
    return nn.Sequential(*modules)


def _times(times, batch):
    if times.ndim == 0: times = times[None]
    if times.ndim == 1: return times[None].expand(batch, -1)
    if times.ndim != 2 or times.shape[0] != batch:
        raise ValueError('times must be [T] or [B,T]')
    return times


class NeuralModalGreenOperator(nn.Module):
    """Direct-time learned modal Green map, linear in force/motion amplitudes.

    Same inference/control/load_response_basis interfaces as NeuralGreenOperator.
    The only physical/context inputs are coefficient25, BC/load categories, and
    known rest geometry. Image/video context cannot enter this forward network.
    Save ``checkpoint()`` and restore with ``from_checkpoint`` to retain the
    exact train-only basis and metadata without fitting it a second time.
    """
    def __init__(self, basis, hidden=256, depth=3, seconds=3., displacement_scale_m=.03,
                 force_scale_N=.2, motion_scale_m=.008):
        super().__init__()
        self.basis = RBFModalBasis(basis); self.transform = PhysicalParameterTransform()
        self.hidden, self.depth, self.seconds = hidden, depth, float(seconds)
        self.displacement_scale_m = float(displacement_scale_m)
        self.force_scale_N, self.motion_scale_m = float(force_scale_N), float(motion_scale_m)
        self.register_buffer('time_frequencies', torch.tensor([1., 2., 4., 8.]))
        self.register_buffer('reference_center', torch.tensor([0., 0., .25]))
        self.register_buffer('reference_size', torch.tensor([.24, .12, .08]))
        self.coefficient_network = _network(16+8+3+12+2+11, self.basis.rank*3, hidden, depth)

    def _shape(self, batch):
        x = (self.basis.rest_points_m-self.reference_center)/self.reference_size
        return torch.cat([x.amin(0), x.amax(0), x.mean(0), x.std(0, correction=0)])[None].expand(batch, -1)

    def _coefficient_response(self, times, physical, boundary_type, force_type, center, amplitude, mode):
        batch = len(physical); tau = times/self.seconds
        modes = torch.full((batch,), mode, dtype=torch.long, device=physical.device)
        context = torch.cat([self.transform.normalize(physical)[:, :16],
                             F.one_hot(boundary_type, 4).to(physical), F.one_hot(force_type, 4).to(physical),
                             (center-self.reference_center)/self.reference_size, self._shape(batch),
                             F.one_hot(modes, 2).to(physical)], -1)
        phase = math.pi*tau[..., None]*self.time_frequencies
        features = torch.cat([tau[..., None], phase.sin(), phase.cos(),
                              force_envelope(force_type, times, self.seconds)[..., None],
                              torch.sin(math.pi*tau).square()[..., None],
                              context[:, None].expand(-1, times.shape[1], -1)], -1)
        response = self.coefficient_network(features).reshape(batch, times.shape[1], self.basis.rank, 3)
        scale = self.force_scale_N if mode == 0 else self.motion_scale_m
        response = self.displacement_scale_m*tau[..., None, None].square()*response/scale
        if amplitude is None:
            return response
        return (response*amplitude[:, None, None]).sum(-1)

    def _combine(self, points, coefficients):
        modes = self.basis(points)
        if points.ndim == 2:
            return torch.einsum('nrc,btr->btnc', modes, coefficients)
        if points.shape[0] != coefficients.shape[0]:
            raise ValueError('Batched point and physical coefficient counts differ')
        return torch.einsum('bnrc,btr->btnc', modes, coefficients)

    def forward(self, points, times, physical, boundary_type, force_type,
                geometry_reference=None, controls=None):
        if geometry_reference is not None:
            raise ValueError('This fixed-object modal basis cannot switch reference geometry at inference')
        if physical.ndim != 2 or physical.shape[1] != PHYSICAL_DIM:
            raise ValueError('physical must be [B,25]')
        batch = len(physical); times = _times(times, batch)
        physical = canonicalize_physical(physical, boundary_type, force_type)
        coefficients = physical.new_zeros(batch, times.shape[1], self.basis.rank)
        for i in range(2):
            coefficients = coefficients+self._coefficient_response(times, physical, boundary_type, force_type,
                physical[:, 10+3*i:13+3*i], physical[:, 16+3*i:19+3*i], 0)
        coefficients = coefficients+self._coefficient_response(times, physical, boundary_type, torch.zeros_like(force_type),
            physical[:, 4:7], physical[:, 22:25], 1)
        for control in controls or []:
            if 'seconds' in control and float(control['seconds']) != self.seconds:
                raise ValueError('Untrained control duration rescaling is unsupported')
            center = torch.as_tensor(control['center_m'], dtype=physical.dtype, device=physical.device)
            amplitude = torch.as_tensor(control['force_N'], dtype=physical.dtype, device=physical.device)
            if center.ndim == 1: center = center[None].expand(batch, -1)
            if amplitude.ndim == 1: amplitude = amplitude[None].expand(batch, -1)
            family = torch.as_tensor(control['force_type'], dtype=torch.long, device=physical.device)
            if family.ndim == 0: family = family.expand(batch)
            start = torch.as_tensor(control.get('start_s', 0.), dtype=physical.dtype, device=physical.device)
            if start.ndim == 1: start = start[:, None]
            if center.shape != (batch, 3) or amplitude.shape != (batch, 3) or family.shape != (batch,):
                raise ValueError('Control centers/forces must be [3]/[B,3], force_type scalar/[B]')
            coefficients = coefficients+self._coefficient_response((times-start).clamp_min(0), physical,
                boundary_type, family, center, amplitude, 0)
        displacement = self._combine(points, coefficients)
        reference = points[None, None] if points.ndim == 2 else points[:, None]
        return reference+displacement

    def load_response_basis(self, points, times, physical, boundary_type, force_type):
        times = _times(times, len(physical)); physical = canonicalize_physical(physical, boundary_type, force_type)
        motion = self._coefficient_response(times, physical, boundary_type, torch.zeros_like(force_type),
                                           physical[:, 4:7], physical[:, 22:25], 1)
        reference = points[None, None] if points.ndim == 2 else points[:, None]
        offset = reference+self._combine(points, motion)
        first = self._coefficient_response(times, physical, boundary_type, force_type, physical[:, 10:13], None, 0)
        second = self._coefficient_response(times, physical, boundary_type, force_type, physical[:, 13:16], None, 0)
        first = first-torch.where((force_type == 3)[:, None, None, None], second, torch.zeros_like(second))
        modes = self.basis(points)
        design = (torch.einsum('nrc,btri->btnci', modes, first) if points.ndim == 2
                  else torch.einsum('bnrc,btri->btnci', modes, first))
        return offset, torch.cat([design, torch.zeros_like(design)], -1)

    def checkpoint(self):
        return {'model_type': type(self).__name__, 'config': {
            'hidden': self.hidden, 'depth': self.depth, 'seconds': self.seconds,
            'displacement_scale_m': self.displacement_scale_m,
            'force_scale_N': self.force_scale_N, 'motion_scale_m': self.motion_scale_m},
            'state_dict': self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint):
        state = checkpoint['state_dict']
        basis = {name: state['basis.'+name] for name in ['rest_points_m', 'modes', 'rbf_weights']}
        basis['width_m'] = float(state['basis.width_m']); basis['metadata'] = state.get('basis._extra_state', {})
        model = cls(basis, **checkpoint['config'])
        parameter = state['coefficient_network.0.weight']
        model.to(device=parameter.device, dtype=parameter.dtype)
        model.load_state_dict(state)
        return model


class NeuralModalOperator(NeuralModalGreenOperator):
    """Nonlinear-amplitude modal baseline; marginalize forces by sampling.

    A nonnegative amplitude gate gives exactly rest for zero forces and stationary
    supports. Known RESET controls can replace handle0's forces at its existing
    center; superimposing controls on nonzero original loads is unsupported.
    There is no exact conditional Gaussian force design for this nonlinear map.
    """
    def __init__(self, basis, **kwargs):
        super().__init__(basis, **kwargs)
        self.coefficient_network = _network(25+8+12+11, self.basis.rank, self.hidden, self.depth)

    def forward(self, points, times, physical, boundary_type, force_type,
                geometry_reference=None, controls=None):
        if geometry_reference is not None:
            raise ValueError('Fixed-object modal basis cannot switch reference geometry')
        physical = canonicalize_physical(physical, boundary_type, force_type)
        if controls:
            if len(controls) != 1 or bool((physical[:, 16:22] != 0).any()):
                raise ValueError('Nonlinear controls require a reset with zero original forces and one known control')
            control = controls[0]
            if float(control.get('start_s', 0.)) != 0 or float(control.get('seconds', self.seconds)) != self.seconds:
                raise ValueError('Nonlinear reset control must use the trained unshifted duration')
            center = torch.as_tensor(control['center_m'], dtype=physical.dtype, device=physical.device)
            if not torch.allclose(center, physical[:, 10:13], atol=1e-6):
                raise ValueError('Reset control must use the existing known handle0 center')
            amplitude = torch.as_tensor(control['force_N'], dtype=physical.dtype, device=physical.device)
            if amplitude.ndim == 1: amplitude = amplitude[None].expand(len(physical), -1)
            force_type = torch.as_tensor(control['force_type'], dtype=torch.long, device=physical.device)
            if force_type.ndim == 0: force_type = force_type.expand(len(physical))
            physical = torch.cat([physical[:, :16], amplitude, torch.zeros_like(amplitude), physical[:, 22:]], -1)
            physical = canonicalize_physical(physical, boundary_type, force_type)
        times = _times(times, len(physical)); tau = times/self.seconds
        phase = math.pi*tau[..., None]*self.time_frequencies
        context = torch.cat([self.transform.normalize(physical), F.one_hot(boundary_type, 4).to(physical),
                             F.one_hot(force_type, 4).to(physical), self._shape(len(physical))], -1)
        features = torch.cat([tau[..., None], phase.sin(), phase.cos(),
                              force_envelope(force_type, times, self.seconds)[..., None],
                              torch.sin(math.pi*tau).square()[..., None],
                              context[:, None].expand(-1, times.shape[1], -1)], -1)
        gate = (torch.linalg.vector_norm(physical[:, 16:22], dim=-1)/self.force_scale_N
                + torch.linalg.vector_norm(physical[:, 22:25], dim=-1)/self.motion_scale_m)
        coefficients = self.displacement_scale_m*tau[..., None].square()*gate[:, None, None]*self.coefficient_network(features)
        reference = points[None, None] if points.ndim == 2 else points[:, None]
        return reference+self._combine(points, coefficients)

    def load_response_basis(self, *args, **kwargs):
        raise TypeError('Nonlinear modal forces require posterior sampling; no exact linear load design')
