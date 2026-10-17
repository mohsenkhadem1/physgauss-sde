"""Second-order staggered dynamics with exact Gaussian force elimination.

Stored velocities are drift/half-step velocities. Position is their exact
integral. Centered drag introduces a parameter-dependent triangular Jacobian;
omitting that determinant would define the wrong trajectory density.
"""
import torch
from .marginal_forces import TrajectoryForceModel, GaussianLoadElimination
from .physics import Trajectory, material_response


class StaggeredForceModel(TrajectoryForceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        h = self.config.duration/self.nsteps
        self.step_sizes = self.times.new_full((self.nsteps,), h)
        self.step_times = torch.arange(self.nsteps, dtype=self.times.dtype)*h
        self.step_features = self.force_prior.features(self.step_times)
        self.kick_sizes = self.step_sizes.clone(); self.kick_sizes[0] *= .5
        for frame, stop in enumerate(self.frame_steps.tolist()):
            if stop:
                self.position_map[frame, :, 8:8+stop*self.free_dof] = torch.cat([
                    torch.eye(self.free_dof, dtype=self.times.dtype)*h*self.marginal_config.velocity_coordinate_scale
                    for _ in range(stop)], 1)
        scalar = (self.kick_sizes[:, None, None]*self.force_prior.config.total_force_std_n*self.step_features[:, None]
                  *(self.force_prior.spatial_weights[self.free]/self.mesh.mass[self.free])[None, :, None])
        mapping = torch.einsum("snk,cd->snckd", scalar, torch.eye(3, dtype=self.times.dtype))
        sd = self.marginal_config.velocity_diffusion*self.kick_sizes.sqrt()
        self.elimination = GaussianLoadElimination(mapping.reshape(-1, self.force_dimension),
                                                   sd[:, None].expand(-1, self.free_dof).reshape(-1))

    def residual(self, state):
        full = self.step_trajectory(state)
        young, drag, _, _ = self.prior.decode(state)
        force, _, _ = material_response(full.x, self.mesh, young[:, None])
        acceleration = force[:, :-1]/self.mesh.mass[None, None, :, None]+state.new_tensor([0., -.6, 0.])
        c = drag[:, None]*self.kick_sizes[None]/2
        residual = ((1+c[:, :, None, None])*full.v[:, 1:]
                    -(1-c[:, :, None, None])*full.v[:, :-1]
                    -self.kick_sizes[None, :, None, None]*acceleration)
        return residual[:, :, self.free].flatten(1)

    def log_dynamic_jacobian(self, state):
        drag = self.prior.decode(state)[1]
        return self.free_dof*torch.log1p(drag[:, None]*self.kick_sizes[None]/2).sum(1)

    def objective(self, state, observations, coefficients=None):
        return super().objective(state, observations, coefficients)-self.log_dynamic_jacobian(state)

    def information(self, mean, observations):
        marginal, explicit, cross = super().information(mean, observations)
        # Exact curvature of -log|dr/du|; likelihood/residual curvature is
        # otherwise Gauss-Newton, as disclosed for all solver controls.
        curvature = self.normalizer_curvature(mean)
        marginal[1, 1] += curvature; explicit[1, 1] += curvature
        return marginal, explicit, cross

    def normalizer_curvature(self, mean):
        drag = self.prior.decode(mean[None])[1][0]
        q = drag*self.kick_sizes/2
        return -self.free_dof*self.prior.std[1].square()*(q/(1+q).square()).sum()

    def rollout(self, theta, coefficients, white=None, external_acceleration=None):
        young, drag, offset, initial_velocity = self.prior.decode(theta)
        x = self.mesh.rest[None]+self.height[None, :, None]*offset[:, None]
        previous = self.height[None, :, None]*initial_velocity[:, None]
        force_external = self.force_prior.nodal_force(coefficients, self.step_times)
        if white is None:
            white = theta.new_zeros(len(theta), self.nsteps, self.free_dof)
        if white.shape != (len(theta), self.nsteps, self.free_dof):
            raise ValueError("Incorrect staggered impulse shape.")
        velocities = []
        for step, kick in enumerate(self.kick_sizes):
            internal, _, _ = material_response(x, self.mesh, young)
            acceleration = ((internal+force_external[:, step])/self.mesh.mass[None, :, None]
                            +theta.new_tensor([0., -.6, 0.]))
            if external_acceleration is not None:
                acceleration = acceleration+external_acceleration(float(self.step_times[step]), x, previous)
            noise = torch.zeros_like(previous)
            noise[:, self.free] = (self.marginal_config.velocity_diffusion*kick.sqrt()
                                  *white[:, step].reshape(len(theta), -1, 3))
            c = drag[:, None, None]*kick/2
            u = ((1-c)*previous+kick*acceleration+noise)/(1+c)
            u = torch.where(self.mesh.pinned[None, :, None], 0., u)
            x = torch.where(self.mesh.pinned[None, :, None], self.mesh.rest[None], x+self.step_sizes[step]*u)
            velocities.append(u); previous = u
        state = self.encode_velocities(theta, torch.stack(velocities, 1))
        self.residual(state)
        return state

    def trajectory(self, state):
        full = self.step_trajectory(state)
        # Second-order finite-difference estimates at integer times. The raw
        # stored variables remain staggered; these are not exact SDE velocities.
        interior = .5*(full.v[:, 1:-1]+full.v[:, 2:])
        last = (1.5*full.v[:, -1]-.5*full.v[:, -2])[:, None]
        velocity = torch.cat((full.v[:, :1], interior, last), 1)
        return Trajectory(full.x[:, self.frame_steps], velocity[:, self.frame_steps])
