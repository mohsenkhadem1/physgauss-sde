"""Low-rank, material-attached external nodal load prior, in newtons.

Five temporal inducing points and three world-frame directions give fifteen
standard-normal coordinates. The Matérn-3/2 inducing projection is a finite
prior; the conditional GP residual is deliberately not included.
"""
from dataclasses import dataclass
import math
import torch
from .benchmark import SceneModel
from .physics import simulate, edge_impulses


def matern32(t, s, length):
    r = math.sqrt(3)*(t[:, None]-s[None]).abs()/length
    return (1+r)*torch.exp(-r)


@dataclass
class ForceConfig:
    knots: int = 5
    temporal_length_fraction: float = .15
    spatial_length_m: float = .6
    total_force_std_n: float = .05

    def validate(self):
        if self.knots < 2 or any(not math.isfinite(v) or v <= 0 for v in
            [self.temporal_length_fraction, self.spatial_length_m, self.total_force_std_n]):
            raise ValueError("Need >=2 knots and positive finite force-prior scales.")
        return self


class ForcePrior:
    def __init__(self, mesh, duration, config=ForceConfig()):
        self.config = config.validate()
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Force duration must be positive and finite.")
        self.duration = duration
        self.knots = torch.linspace(0, duration, config.knots, dtype=mesh.rest.dtype)
        kernel = self.kernel(self.knots, self.knots)
        self.cholesky = torch.linalg.cholesky(kernel + 1e-8*torch.eye(config.knots, dtype=kernel.dtype))
        free = ~mesh.pinned
        center = mesh.rest[free].mean(0)
        radial = torch.exp(-.5*((mesh.rest-center)/config.spatial_length_m).square().sum(1))
        weights = mesh.mass*radial*free.to(mesh.mass.dtype)
        self.spatial_weights = weights/weights.sum()
        self.dimension = 3*config.knots

    def kernel(self, t, s):
        return matern32(t, s, self.duration*self.config.temporal_length_fraction)

    def features(self, times):
        if times.ndim != 1 or not bool(torch.isfinite(times).all()) or bool(((times < 0)|(times > self.duration)).any()):
            raise ValueError("The finite force prior supports only the declared observation-time interval.")
        # Phi(t) = k(t,Z) L^-T; coefficients are independent N(0,1).
        cross = self.kernel(times, self.knots)
        return torch.linalg.solve_triangular(self.cholesky, cross.T, upper=False).T

    def total_force(self, coordinates, times):
        coefficients = coordinates.reshape(-1, self.config.knots, 3)
        return self.config.total_force_std_n*torch.einsum("tk,bkc->btc", self.features(times), coefficients)

    def nodal_force(self, coordinates, times):
        return self.total_force(coordinates, times)[:, :, None, :]*self.spatial_weights[None, None, :, None]


def evaluate_force_posterior(fields, times):
    """Reconstruct a coherent force distribution from portable fields.json.

    Moments use every retained coefficient state; the saved subset gives
    non-Gaussian function draws. These are conditional external-load outputs,
    not white-noise instantaneous forces or force-presence probabilities.
    """
    times = torch.as_tensor(times, dtype=torch.float64)
    duration = fields["duration_s"]
    if times.ndim != 1 or not bool(torch.isfinite(times).all()) or bool(((times < 0)|(times > duration)).any()):
        raise ValueError("Query times must stay inside the recorded interval.")
    weights = times.new_tensor(fields["spatial_nodal_weights"])
    if fields["whitened_force_coefficient_mean"] is None:
        mean = times.new_zeros(len(times), 3)
        covariance = times.new_zeros(3*len(times), 3*len(times))
        draws = mean[None]
    else:
        config = ForceConfig(**fields["force_prior"])
        knots = times.new_tensor(fields["inducing_times_s"])
        length = duration*config.temporal_length_fraction
        kernel = matern32(knots, knots, length)
        chol = torch.linalg.cholesky(kernel+1e-8*torch.eye(len(knots), dtype=times.dtype))
        phi = torch.linalg.solve_triangular(chol, matern32(times, knots, length).T, upper=False).T
        mapping = torch.kron((config.total_force_std_n*phi).contiguous(), torch.eye(3, dtype=times.dtype))
        coefficient_mean = times.new_tensor(fields["whitened_force_coefficient_mean"])
        coefficient_covariance = times.new_tensor(fields["whitened_force_coefficient_covariance"])
        mean = (mapping@coefficient_mean).reshape(-1, 3)
        covariance = mapping@coefficient_covariance@mapping.T
        coefficients = times.new_tensor(fields["whitened_force_coefficient_draws"])
        draws = (coefficients@mapping.T).reshape(len(coefficients), len(times), 3)
    return {"mean_total_N": mean, "joint_total_covariance_N2": covariance,
        "total_draws_N": draws, "mean_nodal_N": mean[:, None, :]*weights[None, :, None],
        "nodal_draws_N": draws[:, :, None, :]*weights[None, None, :, None]}


class ForceSceneModel(SceneModel):
    """Same mechanics/renderer, augmented by a proper external load prior.

    Coordinates: 8 standardized scene parameters, optional force coordinates,
    optional standardized Brownian increments. No truth flags enter dynamics.
    """
    def __init__(self, config, diffusion=None, force=True, force_config=ForceConfig()):
        super().__init__(config, diffusion)
        self.force_prior = ForcePrior(self.mesh, config.duration, force_config)
        self.has_force = bool(force)
        self.force_dimensions = self.force_prior.dimension if force else 0
        self.noise_start = 8+self.force_dimensions
        self.dimension = self.noise_start + (self.nsteps*config.noise_rank if diffusion is not None else 0)
        step_times, step_sizes = [], []
        time = float(self.times[0])
        for interval in range(config.frames-1):
            dt = float(self.times[interval+1]-self.times[interval])/config.substeps
            for _ in range(config.substeps):
                step_times.append(time); step_sizes.append(dt); time += dt
        self.step_times = self.times.new_tensor(step_times)
        self.step_sizes = self.times.new_tensor(step_sizes)
        self.step_features = self.force_prior.features(self.step_times)

    def external_forces(self, coordinates, times=None):
        times = self.step_times if times is None else times
        if not self.has_force:
            return coordinates.new_zeros(len(coordinates), len(times), len(self.mesh.rest), 3)
        return self.force_prior.nodal_force(coordinates[:, 8:self.noise_start], times)

    def _simulate(self, z, record):
        if z.ndim != 2 or z.shape[1] != self.dimension:
            raise ValueError(f"Expected [batch,{self.dimension}] coordinates.")
        young, drag, offset, velocity = self.prior.decode(z)
        white = z[:, self.noise_start:].reshape(-1, self.nsteps, self.config.noise_rank) if self.diffusion else None
        if self.has_force:
            coefficients = z[:, 8:self.noise_start].reshape(-1, self.force_prior.config.knots, 3)
            totals = self.force_prior.config.total_force_std_n*torch.einsum("tk,bkc->btc", self.step_features, coefficients)
            forces = totals[:, :, None, :]*self.force_prior.spatial_weights[None, None, :, None]
        else:
            forces = z.new_zeros(len(z), self.nsteps, len(self.mesh.rest), 3)
        impulses = []
        step = 0
        def external(t, x, v):
            nonlocal step
            if record:
                impulse = (edge_impulses(x, v, self.mesh, white[:, step]*self.step_sizes[step].sqrt(),
                           self.diffusion, self.basis) if white is not None else torch.zeros_like(x))
                impulses.append(impulse)
            acceleration = forces[:, step]/self.mesh.mass[None, :, None]
            step += 1
            return acceleration
        trajectory = simulate(self.mesh, self.times, young, drag, offset, velocity,
            self.config.substeps, white=white, diffusion=self.diffusion, noise_basis=self.basis,
            external_acceleration=external)
        if record:
            return trajectory, forces, torch.stack(impulses, 1)
        return trajectory

    def trajectory(self, z):
        return self._simulate(z, record=False)

    def force_record(self, z):
        return self._simulate(z, record=True)
