"""Independently written PIDG-style per-scene neural reconstruction baseline.

This is not the official PIDG implementation or a reproduction of its results.
It uses a smooth coordinate/time MLP, fixed reference Gaussian appearance,
RGB/mask fitting and optional Lagrangian momentum/constitutive residuals. It
does not infer uncertain boundary conditions and cannot predict unseen pokes.
No FEM assembly, equilibrium solve, time integration or hidden state is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
# Torch 2.14 may JIT native Triton kernels during ordinary eager backward.
# Its documented ATen fallback avoids requiring a system compiler here.
if __name__ == '__main__' and not any(shutil.which(name) for name in ('cc', 'gcc', 'clang')):
    os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import torch
from torch import nn


@dataclass(frozen=True)
class FieldConfig:
    width: int = 64
    depth: int = 4
    frequencies: int = 3
    seconds: float = 3.0
    youngs_modulus_pa: float = 5000.0
    poisson_ratio: float = .35
    density_kg_m3: float = 600.0
    learn_youngs_modulus: bool = False

    def __post_init__(self):
        if self.width < 4 or self.depth < 1 or self.frequencies < 0:
            raise ValueError('Invalid network dimensions')
        if self.seconds <= 0 or self.youngs_modulus_pa <= 0 or self.density_kg_m3 <= 0:
            raise ValueError('Positive duration, stiffness and density required')
        if not -1 < self.poisson_ratio < .5:
            raise ValueError('Poisson ratio must be in (-1,.5)')


def symmetric_matrix(components):
    """Order xx, yy, zz, xy, xz, yz; symmetry is exact by construction."""
    xx, yy, zz, xy, xz, yz = components.unbind(-1)
    return torch.stack((xx, xy, xz, xy, yy, yz, xz, yz, zz), -1).reshape(*components.shape[:-1], 3, 3)


def cofactor_matrix(f):
    """Return det(F) F^-T without an inverse, including singular matrices."""
    a, b, c = f.unbind(-1)
    return torch.stack((torch.linalg.cross(b, c), torch.linalg.cross(c, a),
                        torch.linalg.cross(a, b)), -1)


def batch_jacobian(outputs, inputs, create_graph=True):
    """Independent point Jacobian: [N,outputs,inputs]. No cross-point model."""
    if outputs.ndim != 2 or inputs.ndim != 2 or len(outputs) != len(inputs):
        raise ValueError('Expected matching two-dimensional point batches')
    rows = []
    for component in outputs.unbind(-1):
        if not component.requires_grad:
            rows.append(torch.zeros_like(inputs))
            continue
        grad = torch.autograd.grad(component, inputs, torch.ones_like(component),
                                   create_graph=create_graph, retain_graph=True,
                                   allow_unused=True)[0]
        rows.append(torch.zeros_like(inputs) if grad is None else grad)
    return torch.stack(rows, 1)


def convective_acceleration(velocity, spatial_velocity_jacobian):
    """For grad_v[b,i,j]=dv_i/dx_j, return (v dot grad) v, not grad(|v|²/2)."""
    return torch.einsum('bj,bij->bi', velocity, spatial_velocity_jacobian)


def neo_hookean_cauchy(f, youngs_modulus_pa, poisson_ratio, min_j=1e-5):
    """Compressible logarithmic neo-Hookean stress, Pa; positive J is required.

    Clamping only keeps rejected/inverted iterates finite. Callers separately
    penalize J<0.1; clamped states are not valid elastic constitutive solutions.
    """
    j = torch.linalg.det(f)
    safe = j.clamp_min(min_j)
    mu = youngs_modulus_pa / (2 * (1 + poisson_ratio))
    lam = youngs_modulus_pa * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    eye = torch.eye(3, device=f.device, dtype=f.dtype)
    return mu / safe[..., None, None] * (f @ f.transpose(-1, -2) - eye) + \
        lam * safe.log()[..., None, None] / safe[..., None, None] * eye


class PIDGStyleField(nn.Module):
    """Per-scene Lagrangian χ(X,t), velocity and symmetric Cauchy stress.

    Smooth SiLU activations permit second spatial/time derivatives. The exact
    initial conditions χ(X,0)=X and v(X,0)=0 match this controlled dataset's
    declared rest start. E is either a declared constant or a regularized scalar
    latent; it must not be interpreted as calibrated identification by itself.
    """
    def __init__(self, rest_points_m, config=FieldConfig()):
        super().__init__()
        self.config = config
        rest = torch.as_tensor(rest_points_m, dtype=torch.get_default_dtype())
        if rest.ndim != 2 or rest.shape[1] != 3 or not torch.isfinite(rest).all():
            raise ValueError('Expected finite rest points [N,3]')
        span = rest.amax(0) - rest.amin(0)
        if float(span.max()) <= 0:
            raise ValueError('Reference geometry must have nonzero extent')
        self.register_buffer('center', (rest.amax(0) + rest.amin(0)) / 2)
        self.register_buffer('length', span.max())
        self.register_buffer('reference_log_youngs', rest.new_tensor(config.youngs_modulus_pa).log())
        self.log_youngs_modulus = nn.Parameter(self.reference_log_youngs.clone(),
                                             requires_grad=config.learn_youngs_modulus)
        if config.frequencies:
            self.register_buffer('fourier_frequencies', 2. ** torch.arange(config.frequencies, dtype=rest.dtype) * torch.pi)
        else:
            self.register_buffer('fourier_frequencies', rest.new_empty(0))
        feature_size = 4 + 8 * config.frequencies
        layers = []
        for layer in range(config.depth):
            layers.extend((nn.Linear(feature_size if layer == 0 else config.width, config.width), nn.SiLU()))
        self.trunk = nn.Sequential(*layers)
        self.output = nn.Linear(config.width, 12)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @property
    def youngs_modulus_pa(self):
        return self.log_youngs_modulus.exp()

    def forward(self, points, times):
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError('Expected [N,3] material points')
        t = torch.as_tensor(times, device=points.device, dtype=points.dtype)
        if t.ndim == 0:
            t = t.expand(len(points), 1)
        elif t.ndim == 1:
            t = t[:, None]
        if t.shape != (len(points), 1):
            raise ValueError('Times must be scalar or one per material point')
        tau = t / self.config.seconds
        coordinates = torch.cat(((points - self.center) / self.length, tau), -1)
        phase = coordinates[..., None] * self.fourier_frequencies
        features = torch.cat((coordinates, phase.sin().flatten(1), phase.cos().flatten(1)), -1)
        raw = self.output(self.trunk(features))
        displacement = tau.square() * self.length * raw[:, :3]
        velocity = tau * self.length / self.config.seconds * raw[:, 3:6]
        sigma = self.config.youngs_modulus_pa * symmetric_matrix(raw[:, 6:])
        return {'positions_m': points + displacement, 'displacement_m': displacement,
                'velocity_m_s': velocity, 'cauchy_stress_pa': sigma}

    def predict_nodes(self, rest_points, times):
        """Return [times,nodes,3], sharing weights across times, no integration."""
        t = torch.as_tensor(times, device=rest_points.device, dtype=rest_points.dtype).reshape(-1)
        count = len(rest_points)
        points = rest_points[None].expand(len(t), -1, -1).reshape(-1, 3)
        expanded_t = t[:, None].expand(-1, count).reshape(-1, 1)
        return self(points, expanded_t)['positions_m'].reshape(len(t), count, 3)


def physics_residuals(model, points, times):
    """Dimensionless pointwise residuals in the known reference volume.

    Momentum uses reference Div(P), P=J sigma F^-T and material acceleration
    dv/dt. Adding an Eulerian convective term here would double-count advection.
    There are no volumetric loads in the zero-gravity block dataset. Unknown
    surface loading and fixture reactions are not set to zero or supervised.
    """
    x = points.detach().clone().requires_grad_(True)
    t = times.detach().reshape(-1, 1).clone().requires_grad_(True)
    fields = model(x, t)
    q, v, sigma = (fields[k] for k in ('positions_m', 'velocity_m_s', 'cauchy_stress_pa'))
    deformation_gradient = batch_jacobian(q, x)
    material_velocity = batch_jacobian(q, t).squeeze(-1)
    acceleration = batch_jacobian(v, t).squeeze(-1)
    p = sigma @ cofactor_matrix(deformation_gradient)
    divergence_rows = []
    for row in range(3):
        derivative = batch_jacobian(p[:, row], x)
        divergence_rows.append(derivative.diagonal(dim1=-2, dim2=-1).sum(-1))
    divergence = torch.stack(divergence_rows, -1)
    elastic_stress = neo_hookean_cauchy(deformation_gradient, model.youngs_modulus_pa,
                                       model.config.poisson_ratio)
    density = model.config.density_kg_m3
    force_density_scale = model.config.youngs_modulus_pa / model.length + density * model.length / model.config.seconds ** 2
    j = torch.linalg.det(deformation_gradient)
    return {'kinematic': (v - material_velocity) / (model.length / model.config.seconds),
            'momentum': (density * acceleration - divergence) / force_density_scale,
            'constitutive': (sigma - elastic_stress) / model.config.youngs_modulus_pa,
            'inversion': torch.relu(.1 - j), 'determinants': j,
            'displacement_prior': fields['displacement_m'] / model.length,
            'material_prior': model.log_youngs_modulus - model.reference_log_youngs}


def sample_reference_volume(rest, tetrahedra, count, generator=None):
    """Volume-uniform tetrahedral sampling; geometry only, no FEM operation."""
    vertices = rest[tetrahedra]
    volumes = torch.linalg.det((vertices[:, 1:] - vertices[:, :1]).transpose(-1, -2)).abs()
    if not torch.isfinite(volumes).all() or float(volumes.sum()) <= 0:
        raise ValueError('Reference volume must be finite and positive')
    indices = torch.multinomial(volumes, count, replacement=True, generator=generator)
    exponentials = -torch.rand((count, 4), device=rest.device, dtype=rest.dtype,
                               generator=generator).clamp_min(1e-8).log()
    weights = exponentials / exponentials.sum(-1, keepdim=True)
    return (vertices[indices] * weights[..., None]).sum(1)


def observation_loss(prediction, target_rgb, target_mask):
    """Same fixed appearance and supplied masks; no features/truth correspondences."""
    weight = 1 + 4 * target_mask
    rgb = ((prediction['rgb'] - target_rgb).square() * weight[..., None]).sum() / (3 * weight.sum())
    silhouette = (prediction['alpha'] - target_mask).square().mean()
    return rgb + .1 * silhouette, {'rgb_mse_weighted': rgb, 'silhouette_mse': silhouette}


def fit_episode(episode, output, *, steps=500, device='cuda:3', size=(64, 48),
                seed=1701, physics_weight=.001, kinematic_weight=.001,
                constitutive_weight=.001, collocation_points=64,
                prior_weight=.0001, learning_rate=.002,
                config=None, query_cameras=None, log_every=20,
                roi=False, roi_margin=30, appearance_calibration=None):
    """Fit only public observations; fixed final checkpoint, no test selection.

    Query camera calibration is allowed; it is read only after model fitting.
    No recipe, summary, trajectory, visibility label or withheld image is opened.
    """
    try:
        from gaussian_observation import IsaacGaussianRenderer, load_rgb_inputs, project_reference_crops
    except ImportError:
        from experiments.gaussian_observation import IsaacGaussianRenderer, load_rgb_inputs, project_reference_crops
    if steps < 1 or collocation_points < 1:
        raise ValueError('Positive steps and collocation count required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), Path(__file__).with_name('gaussian_observation.py')]
    source_bytes = {p.name: p.read_bytes() for p in source_paths}
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name,data in source_bytes.items()}
    (output/'provenance').mkdir()
    for name,data in source_bytes.items():
        (output/'provenance'/name).write_bytes(data)
    started = time.monotonic()
    torch.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        torch.empty(0, device=device) # allocator counters require initialized CUDA
        torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(seed)
    inputs = load_rgb_inputs(episode, size, roi=roi, margin=roi_margin)
    reference, cameras = inputs['reference'], inputs['cameras']
    if config is None:
        config = FieldConfig(seconds=float(np.max(cameras['times_s'])))
    rest = torch.as_tensor(reference['rest_points_m'], device=device, dtype=torch.float32)
    tetrahedra = torch.as_tensor(reference['tetrahedra'], device=device, dtype=torch.long)
    model = PIDGStyleField(reference['rest_points_m'], config).to(device)
    if isinstance(appearance_calibration, (str, Path)):
        appearance_path = Path(appearance_calibration)
        appearance_data = json.loads(appearance_path.read_text())
        appearance_sha256 = hashlib.sha256(appearance_path.read_bytes()).hexdigest()
    else:
        appearance_data = appearance_calibration or {}
        appearance_sha256 = None
    renderer = IsaacGaussianRenderer(reference, size=size, device=device,
                                     calibration_size=inputs.get('calibration_size',(640,480)),
                                     color_matrix=appearance_data.get('color_matrix'),
                                     color_bias=appearance_data.get('color_bias'))
    target_rgb = torch.as_tensor(inputs['rgb'], device=device)
    target_mask = torch.as_tensor(inputs['mask'], device=device)
    poses = torch.as_tensor(cameras['camera_to_world_usd_row'], device=device, dtype=rest.dtype)
    intrinsics = torch.as_tensor(cameras['K'], device=device, dtype=rest.dtype)
    times = torch.as_tensor(cameras['times_s'], device=device, dtype=rest.dtype)
    background_pixels = inputs['rgb'][0][inputs['mask'][0] < .01]
    if len(background_pixels) == 0:
        raise ValueError('Observed initial frame has no background pixels')
    background = np.median(background_pixels, axis=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    use_physics = physics_weight > 0 or kinematic_weight > 0 or constitutive_weight > 0
    history = []
    for step in range(1, steps + 1):
        # A single frame bounds the dense differentiable reference splatter memory.
        index = int(rng.integers(len(times)))
        positions = model.predict_nodes(rest, times[index:index+1])
        calibration = intrinsics[index:index+1] if intrinsics.ndim == 3 else intrinsics
        prediction = renderer(positions, poses[index:index+1], calibration, background)
        loss, components = observation_loss(prediction, target_rgb[index:index+1], target_mask[index:index+1])
        if use_physics:
            points = sample_reference_volume(rest, tetrahedra, collocation_points)
            query_t = torch.rand(collocation_points, device=device, dtype=rest.dtype) * config.seconds
            residuals = physics_residuals(model, points, query_t)
            for name, weight in [('momentum', physics_weight), ('kinematic', kinematic_weight),
                                 ('constitutive', constitutive_weight)]:
                components[name] = residuals[name].square().mean()
                loss = loss + weight * components[name]
            components['inversion'] = residuals['inversion'].square().mean()
            components['displacement_prior'] = residuals['displacement_prior'].square().mean()
            loss = loss + .1 * components['inversion'] + prior_weight * components['displacement_prior']
            loss = loss + prior_weight * residuals['material_prior'].square().mean()
        else:
            components['displacement_prior'] = ((positions - rest) / model.length).square().mean()
            loss = loss + prior_weight * components['displacement_prior']
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Non-finite training loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step == steps or step % log_every == 0:
            row = {'step': step, 'observed_frame_id': int(cameras['frame_ids'][index]),
                   'loss': float(loss.detach()), 'gradient_norm': float(gradient_norm),
                   **{k: float(v.detach()) for k, v in components.items()}}
            history.append(row)
            print(json.dumps(row), flush=True)
            (output/'training.json').write_text(json.dumps(history, indent=2)+'\n')
    # Camera queries are not measurement values and do not affect training.
    if query_cameras:
        with np.load(query_cameras) as raw:
            queries = {k: raw[k] for k in raw.files}
        if roi:
            crops = project_reference_crops(reference, queries['camera_to_world_usd_row'],
                                             queries['K'], size, roi_margin)
            queries = {**queries, 'K': crops['K'], 'crop_boxes_xyxy': crops['crop_boxes_xyxy']}
    else:
        queries = cameras
    model.eval()
    with torch.no_grad():
        q = model.predict_nodes(rest, queries['times_s']).cpu().numpy()
    prediction_data = {'positions_m':q, 'times_s':queries['times_s'],
                       'frame_ids':queries['frame_ids'], 'K':queries['K'],
                       'camera_to_world_usd_row':queries['camera_to_world_usd_row']}
    if 'crop_boxes_xyxy' in queries:
        prediction_data['crop_boxes_xyxy'] = queries['crop_boxes_xyxy']
    np.savez_compressed(output/'predictions.npz', **prediction_data)
    torch.save({'model_state_dict': model.state_dict(), 'field_config': asdict(config),
                'rest_points_m': reference['rest_points_m']}, output/'model.pt')
    episode = Path(episode)
    files = [episode/'reference.npz', *sorted((episode/'observations').rglob('*'))]
    file_hashes = {str(p.relative_to(episode)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in files if p.is_file() and (p.suffix == '.npz' and p.parent.name != 'depth' or p.parent.name in ('rgb', 'masks'))}
    result = {'baseline': 'PIDG-style independent per-scene coordinate MLP; not official PIDG',
              'official_reproduction': False, 'episode_id': episode.name,
              'input_channels': inputs['input_channels'], 'hidden_state_used': False,
              'appearance_fitted': False, 'network_config': asdict(config),
              'frozen_appearance_calibration': appearance_data,
              'appearance_calibration_sha256': appearance_sha256,
              'roi': bool(roi), 'roi_margin_native_pixels': roi_margin if roi else None,
              'estimated_scalar_youngs_modulus_pa': float(model.youngs_modulus_pa.detach()),
              'stiffness_identification_qualification': 'Scalar latent/default, not calibrated identification; unknown loads and boundary responses remain confounded',
              'steps': steps, 'seed': seed, 'resolution': list(size),
              'torch_disable_native_jit': os.environ.get('TORCH_DISABLE_NATIVE_JIT'),
              'physics_weight': physics_weight, 'kinematic_weight': kinematic_weight,
              'constitutive_weight': constitutive_weight, 'prior_weight': prior_weight,
              'collocation_points': collocation_points, 'learning_rate': learning_rate,
              'checkpoint_selection': 'fixed final step; no withheld image or test score read',
              'observed_frame_ids': cameras['frame_ids'].tolist(),
              'query_frame_ids': np.asarray(queries['frame_ids']).tolist(),
              'seconds_elapsed': time.monotonic()-started,
              'peak_gpu_memory_allocated_bytes': torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else None,
              'input_sha256': file_hashes,
              'source_sha256': source_hashes,
              'limitations': ['one scene fitted at a time', 'deterministic prediction',
                             'no unknown boundary-type posterior', 'no control-conditioned new-poke prediction',
                             'fixed Gaussian/Isaac shader discrepancy',
                             'constitutive prior differs from the simulator constitutive implementation']}
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode'); parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:3'); parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--size', type=int, nargs=2, default=(64,48), metavar=('WIDTH','HEIGHT'))
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--physics-weight', type=float, default=.001)
    parser.add_argument('--kinematic-weight', type=float, default=.001)
    parser.add_argument('--constitutive-weight', type=float, default=.001)
    parser.add_argument('--prior-weight', type=float, default=.0001)
    parser.add_argument('--collocation-points', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=.002)
    parser.add_argument('--query-cameras')
    parser.add_argument('--roi', action='store_true', help='Known-rest-geometry camera crops, identical to shared perception model')
    parser.add_argument('--roi-margin', type=int, default=30)
    parser.add_argument('--appearance-calibration', help='Frozen global train-only appearance calibration JSON')
    parser.add_argument('--learn-youngs-modulus', action='store_true')
    args = parser.parse_args()
    # Duration is known from the public observation schedule, not a truth recipe.
    with np.load(Path(args.episode)/'observations/cameras.npz') as raw:
        seconds = float(np.max(raw['times_s']))
    config = FieldConfig(seconds=seconds, learn_youngs_modulus=args.learn_youngs_modulus)
    result = fit_episode(args.episode, args.output, steps=args.steps, device=args.device,
                         size=tuple(args.size), seed=args.seed, physics_weight=args.physics_weight,
                         kinematic_weight=args.kinematic_weight,
                         constitutive_weight=args.constitutive_weight, prior_weight=args.prior_weight,
                         collocation_points=args.collocation_points, learning_rate=args.learning_rate,
                         query_cameras=args.query_cameras, config=config, roi=args.roi,
                         roi_margin=args.roi_margin, appearance_calibration=args.appearance_calibration)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
