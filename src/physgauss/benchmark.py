"""Synthetic ground truth is separated from the observation-only fitting API."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import Tensor
from .camera import orbit
from .physics import Mesh, Trajectory, cube_mesh, simulate
from .render import GaussianRenderer
from .stochastic import calibration_law, spatial_basis


@dataclass
class Config:
    seed: int = 1729
    diffusion_seed: int = 918
    frames: int = 21
    duration: float = 1.0
    image_size: int = 24
    substeps: int = 3
    observed_frames: tuple = (0, 3, 6, 14, 17, 20)
    noise_rank: int = 3
    observation_sigma: float = .015
    truth_stochastic: bool = True
    truth_process_scale: float = 1.0
    truth_external_pulse: bool = False
    adam_steps: int = 100
    lbfgs_steps: int = 20
    posterior_samples: int = 64
    diffusion_steps: int = 350
    finite_difference_chunk: int = 12
    threads: int = 2

    def validate(self):
        obs = list(self.observed_frames)
        if self.frames < 3 or self.duration <= 0 or self.image_size < 8:
            raise ValueError("Need >=3 frames, positive duration, and image_size>=8.")
        if obs != sorted(set(obs)) or not obs or min(obs) < 0 or max(obs) >= self.frames:
            raise ValueError("Observed frames must be unique, sorted, and in range.")
        if obs[0] != 0 or obs[-1] != self.frames - 1 or len(obs) >= self.frames:
            raise ValueError("This offline smoothing demo needs observed endpoints and withheld frames.")
        if self.substeps < 1 or self.observation_sigma <= 0 or self.noise_rank < 1:
            raise ValueError("Invalid step count, observation noise, or noise rank.")
        if not math.isfinite(self.truth_process_scale) or self.truth_process_scale < 0:
            raise ValueError("Truth process scale must be finite and nonnegative.")
        if self.posterior_samples < 4 or self.finite_difference_chunk < 1 or self.threads < 1:
            raise ValueError("Need >=4 posterior samples, positive chunk size and thread count.")
        if min(self.adam_steps, self.lbfgs_steps, self.diffusion_steps) < 0:
            raise ValueError("Optimization step counts cannot be negative.")
        return self

    @classmethod
    def quick(cls):
        return cls(frames=9, duration=.4, image_size=12, substeps=2,
                   observed_frames=(0, 2, 6, 8), noise_rank=2,
                   adam_steps=16, lbfgs_steps=6, posterior_samples=8,
                   diffusion_steps=70, finite_difference_chunk=8)


@dataclass
class Observations:
    indices: Tensor
    rgb: Tensor
    sigma: float


class ParameterPrior:
    """Eight standardized coordinates; known density fixes a scale ambiguity."""
    names = ("log_young", "log_drag", "offset_x", "offset_y", "offset_z",
             "velocity_x", "velocity_y", "velocity_z")

    def __init__(self, dtype=torch.float64):
        self.mean = torch.tensor([math.log(16), math.log(.6), .08, -.01, .02, 0, 0, 0], dtype=dtype)
        self.std = torch.tensor([.35, .5, .06, .04, .06, .15, .10, .15], dtype=dtype)

    def decode(self, z):
        raw = self.mean + self.std * z[..., :8]
        return raw[..., 0].exp(), raw[..., 1].exp(), raw[..., 2:5], raw[..., 5:8]


class SceneModel:
    def __init__(self, config: Config, diffusion=None, process_scale=1.0):
        if not math.isfinite(process_scale) or process_scale < 0:
            raise ValueError("Process scale must be finite and nonnegative.")
        self.config = config.validate()
        self.mesh = cube_mesh()
        self.times = torch.linspace(0, config.duration, config.frames, dtype=torch.float64)
        self.eye, self.rotation = orbit(self.times)
        self.renderer = GaussianRenderer(self.mesh, config.image_size)
        self.prior = ParameterPrior()
        self.diffusion = diffusion
        self.process_scale = process_scale
        self.basis = spatial_basis(self.mesh, config.noise_rank)
        self.nsteps = (config.frames - 1) * config.substeps
        self.dimension = 8 + (self.nsteps * config.noise_rank if diffusion is not None else 0)

    def trajectory(self, z):
        if z.ndim != 2 or z.shape[-1] != self.dimension:
            raise ValueError(f"Expected [batch,{self.dimension}] standardized coordinates.")
        young, drag, offset, velocity = self.prior.decode(z)
        white = z[:, 8:].reshape(-1, self.nsteps, self.config.noise_rank) if self.diffusion else None
        if white is not None:
            white = self.process_scale * white
        return simulate(self.mesh, self.times, young, drag, offset, velocity,
                        self.config.substeps, white=white, diffusion=self.diffusion,
                        noise_basis=self.basis)

    def predict(self, z, indices=None):
        trajectory = self.trajectory(z)
        if indices is None:
            return self.renderer(trajectory.x, self.eye, self.rotation)
        return self.renderer(trajectory.x[:, indices], self.eye[indices], self.rotation[indices])


@dataclass
class Benchmark:
    config: Config
    observations: Observations
    truth: Trajectory
    truth_rgb: Tensor
    truth_novel_rgb: Tensor
    truth_parameters: dict


def make_benchmark(config: Config) -> Benchmark:
    config.validate()
    scene = SceneModel(config)
    gen = torch.Generator().manual_seed(config.seed)
    young = scene.times.new_tensor([19.5])
    drag = scene.times.new_tensor([.55])
    offset = scene.times.new_tensor([[.11, -.035, .06]])
    velocity = scene.times.new_tensor([[-.12, .04, .10]])
    white = None
    if config.truth_stochastic:
        white = config.truth_process_scale * torch.randn(
            1, scene.nsteps, config.noise_rank, generator=gen, dtype=torch.float64)
    external = None
    if config.truth_external_pulse:
        def external(t, x, v):
            # Deterministic external acceleration absent from all inference models.
            strength = math.exp(-.5 * ((t / config.duration - .5) / .08) ** 2)
            height = (~scene.mesh.pinned).to(x.dtype)[None, :, None]
            return strength * height * x.new_tensor([.45, 0., -.30])
    with torch.no_grad():
        truth = simulate(scene.mesh, scene.times, young, drag, offset, velocity, config.substeps,
                         white=white, diffusion=calibration_law if white is not None else None,
                         noise_basis=scene.basis, external_acceleration=external)
        rgb = scene.renderer(truth.x, scene.eye, scene.rotation)[0]
        novel = scene.renderer(truth.x, *orbit(scene.times, offset_degrees=100))[0]
    indices = torch.tensor(config.observed_frames, dtype=torch.long)
    obs_gen = torch.Generator().manual_seed(config.seed + 100_000)
    noise = torch.randn(rgb[indices].shape, generator=obs_gen, dtype=torch.float64)
    # Keep floating RGB unclipped: inference's Gaussian observation likelihood is exact.
    observed = rgb[indices] + config.observation_sigma * noise
    return Benchmark(config, Observations(indices, observed, config.observation_sigma),
                     truth, rgb, novel,
                     {"young_pa": float(young[0]), "drag_per_second": float(drag[0]),
                      "process_scale": config.truth_process_scale if config.truth_stochastic else 0.,
                      "unmodeled_external_pulse": config.truth_external_pulse,
                      "offset_m": offset[0].tolist(), "velocity_m_per_second": velocity[0].tolist()})
