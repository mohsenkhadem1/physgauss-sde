"""Exact elimination of Gaussian loads in a nonlinear trajectory model.

The mechanics and renderer are unchanged. The NEW model adds independent,
full-rank velocity diffusion on free nodes, so the trajectory has a proper
density. This is not an exact reparameterization of the singular v0.4 model.
All momentum residuals below are divided by nodal mass (velocity units).
"""
from dataclasses import dataclass
import math
import torch
from .forces import ForceConfig, ForceSceneModel
from .physics import Trajectory, material_response


@dataclass
class MarginalConfig:
    # d v = ... dt + velocity_diffusion * dW; units (m/s) / sqrt(s).
    velocity_diffusion: float = .02
    velocity_coordinate_scale: float = .1

    def validate(self):
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.velocity_diffusion, self.velocity_coordinate_scale)):
            raise ValueError("Trajectory diffusion and coordinate scales must be positive and finite.")
        return self


class GaussianLoadElimination:
    """r = G a + eta; a~N(0,I), eta~N(0,diag(sd**2)).

    Woodbury/determinant-lemma operations only factor a force-size matrix.
    The SVD supplies a symmetric inverse square root for Gauss-Newton residuals.
    No dense trajectory covariance or inverse is formed during inference.
    """
    def __init__(self, mapping, sd):
        if mapping.ndim != 2 or sd.shape != mapping.shape[:1]:
            raise ValueError("Expected mapping [residual,force] and sd [residual].")
        if not bool(torch.isfinite(mapping).all() and torch.isfinite(sd).all()) or bool((sd <= 0).any()):
            raise ValueError("Need finite load mapping and positive residual standard deviations.")
        self.mapping, self.sd = mapping, sd
        self.white_mapping = mapping/sd[:, None]
        self.precision = torch.eye(mapping.shape[1], dtype=mapping.dtype)+self.white_mapping.T@self.white_mapping
        self.cholesky = torch.linalg.cholesky(self.precision)
        self.conditional_covariance = torch.cholesky_inverse(self.cholesky)
        self.u, singular, _ = torch.linalg.svd(self.white_mapping, full_matrices=False)
        self.whitening_adjustment = torch.rsqrt(1+singular.square())-1
        self.logdet = 2*sd.log().sum()+2*self.cholesky.diagonal().log().sum()

    def conditional_mean(self, residual):
        rhs = (residual/self.sd)@self.white_mapping
        return torch.cholesky_solve(rhs.T, self.cholesky).T

    def whiten(self, residual):
        white = residual/self.sd
        return white+((white@self.u)*self.whitening_adjustment)@self.u.T

    def marginal_nll(self, residual):
        return .5*(self.whiten(residual).square().sum(-1)+self.logdet
                   +self.mapping.shape[0]*math.log(2*math.pi))

    def joint_nll(self, residual, coefficients):
        white = residual/self.sd-coefficients@self.white_mapping.T
        return (.5*(white.square().sum(-1)+coefficients.square().sum(-1))
                +self.sd.log().sum()+.5*(self.mapping.shape[0]+self.mapping.shape[1])*math.log(2*math.pi))

    def sample_conditional(self, residual, generator):
        mean = self.conditional_mean(residual)
        eps = torch.randn(mean.shape, dtype=mean.dtype, generator=generator)
        return mean+torch.linalg.solve_triangular(self.cholesky.T, eps.T, upper=True).T


class TrajectoryForceModel(ForceSceneModel):
    """All solver-step velocities plus eight scene coordinates are unknown.

    Positions are exact cumulative semi-implicit Euler updates. The initial
    state, clamped supports, material law, cameras and force basis match v0.4.
    Full-rank free-node velocity diffusion is fixed, declared model discrepancy;
    it is not the previous balanced, neural, low-rank internal diffusion.
    """
    def __init__(self, config, force_config=ForceConfig(), marginal_config=MarginalConfig()):
        super().__init__(config, force=True, force_config=force_config)
        self.marginal_config = marginal_config.validate()
        self.free = ~self.mesh.pinned
        self.free_dof = 3*int(self.free.sum())
        self.state_dimension = 8+self.nsteps*self.free_dof
        self.dimension = self.state_dimension
        self.force_dimension = self.force_prior.dimension
        self.height = (self.mesh.rest[:, 1]-self.mesh.rest[:, 1].min())
        self.height = self.height/self.height.max()
        self.frame_steps = torch.arange(config.frames)*config.substeps
        temporal = force_config.total_force_std_n*self.step_features
        # G maps whitened total-force coefficients to free-node delta-v.
        scalar = (self.step_sizes[:, None, None]*temporal[:, None, :]
                  *(self.force_prior.spatial_weights[self.free]/self.mesh.mass[self.free])[None, :, None])
        mapping = torch.einsum("snk,cd->snckd", scalar, torch.eye(3, dtype=self.times.dtype))
        mapping = mapping.reshape(self.nsteps*self.free_dof, self.force_dimension)
        sd = (self.marginal_config.velocity_diffusion*self.step_sizes.sqrt())
        sd = sd[:, None].expand(-1, self.free_dof).reshape(-1)
        self.elimination = GaussianLoadElimination(mapping, sd)
        # Position is linear in these trajectory coordinates. Cache its map.
        position_map = self.times.new_zeros(config.frames, self.free_dof, self.state_dimension)
        for frame, stop in enumerate(self.frame_steps.tolist()):
            for node in range(int(self.free.sum())):
                for axis in range(3):
                    row = 3*node+axis
                    position_map[frame, row, 2+axis] = self.prior.std[2+axis]*self.height[self.free][node]
            if stop:
                position_map[frame, :, 8:8+stop*self.free_dof] = torch.cat([
                    torch.eye(self.free_dof, dtype=self.times.dtype)*dt*self.marginal_config.velocity_coordinate_scale
                    for dt in self.step_sizes[:stop]], 1)
        self.position_map = position_map

    def step_trajectory(self, state):
        if state.ndim != 2 or state.shape[1] != self.state_dimension:
            raise ValueError(f"Expected [batch,{self.state_dimension}] trajectory coordinates.")
        _, _, offset, initial_velocity = self.prior.decode(state)
        batch = len(state)
        x0 = self.mesh.rest[None]+self.height[None, :, None]*offset[:, None]
        v0 = self.height[None, :, None]*initial_velocity[:, None]
        free_v = state[:, 8:].reshape(batch, self.nsteps, -1, 3)*self.marginal_config.velocity_coordinate_scale
        next_v = state.new_zeros(batch, self.nsteps, len(self.mesh.rest), 3)
        next_v[:, :, self.free] = free_v
        x = torch.cat((x0[:, None], x0[:, None]+torch.cumsum(next_v*self.step_sizes[None, :, None, None], 1)), 1)
        v = torch.cat((v0[:, None], next_v), 1)
        return Trajectory(x, v)

    def trajectory(self, state):
        full = self.step_trajectory(state)
        return Trajectory(full.x[:, self.frame_steps], full.v[:, self.frame_steps])

    def external_forces(self, state, times=None):
        """Conditional mean only; use elimination for conditional uncertainty."""
        coefficients = self.elimination.conditional_mean(self.residual(state))
        return self.force_prior.nodal_force(coefficients, self.step_times if times is None else times)

    def force_record(self, state):
        raise NotImplementedError("Trajectory inference has conditional force distributions; no single latent force/innovation record. Use elimination and paired conditional draws.")

    def residual(self, state):
        full = self.step_trajectory(state)
        young, drag, _, _ = self.prior.decode(state)
        # Validate ALL solver states, including the last one, as in simulate().
        force, _, _ = material_response(full.x, self.mesh, young[:, None])
        acceleration = (force[:, :-1]/self.mesh.mass[None, None, :, None]
                        +state.new_tensor([0., -.6, 0.])-drag[:, None, None, None]*full.v[:, :-1])
        residual = full.v[:, 1:]-full.v[:, :-1]-self.step_sizes[None, :, None, None]*acceleration
        return residual[:, :, self.free].flatten(1)

    def encode_velocities(self, theta, next_velocities):
        return torch.cat((theta, next_velocities[:, :, self.free].flatten(1)
                          /self.marginal_config.velocity_coordinate_scale), 1)

    def rollout(self, theta, coefficients, white=None):
        """Causal NEW-model simulator, also used to initialize/validate paths."""
        young, drag, offset, initial_velocity = self.prior.decode(theta)
        x = self.mesh.rest[None]+self.height[None, :, None]*offset[:, None]
        v = self.height[None, :, None]*initial_velocity[:, None]
        forces = self.force_prior.nodal_force(coefficients, self.step_times)
        if white is None:
            white = theta.new_zeros(len(theta), self.nsteps, self.free_dof)
        if white.shape != (len(theta), self.nsteps, self.free_dof):
            raise ValueError("Incorrect full-rank innovation shape.")
        velocities = []
        for step, dt in enumerate(self.step_sizes):
            force, _, _ = material_response(x, self.mesh, young)
            acceleration = (force/self.mesh.mass[None, :, None]+theta.new_tensor([0., -.6, 0.])
                            -drag[:, None, None]*v+forces[:, step]/self.mesh.mass[None, :, None])
            noise = torch.zeros_like(v)
            noise[:, self.free] = (self.marginal_config.velocity_diffusion*dt.sqrt()
                                   *white[:, step].reshape(len(theta), -1, 3))
            v = torch.where(self.mesh.pinned[None, :, None], 0., v+dt*acceleration+noise)
            x = torch.where(self.mesh.pinned[None, :, None], self.mesh.rest[None], x+dt*v)
            velocities.append(v)
        state = self.encode_velocities(theta, torch.stack(velocities, 1))
        self.residual(state)  # Includes final deformation validation.
        return state

    def image_residual(self, state, observations):
        return ((self.predict(state, observations.indices)-observations.rgb[None])
                /observations.sigma).flatten(1)

    def objective(self, state, observations, coefficients=None):
        r = self.residual(state)
        dynamics = (self.elimination.marginal_nll(r) if coefficients is None
                    else self.elimination.joint_nll(r, coefficients))
        # Density is in scaled velocity coordinates: the Jacobian is constant.
        jacobian = (self.state_dimension-8)*math.log(self.marginal_config.velocity_coordinate_scale)
        return (.5*self.image_residual(state, observations).square().sum(1)
                +.5*state[:, :8].square().sum(1)+dynamics-jacobian)

    def jacobians(self, mean, observations):
        """Exact AD on the current depth-order branch; no image finite differences.

        Images depend on only 12 free-node coordinates per frame. Differentiate
        there, then chain the cached position map; this avoids a huge image AD
        batch over hundreds of trajectory coordinates.
        """
        with torch.no_grad():
            self.residual(mean[None])
            frames = self.trajectory(mean[None]).x
        image_columns = []
        for frame in observations.indices.tolist():
            x = frames[:, [frame]].clone()
            def render_free(q):
                nodes = x.clone()
                nodes[0, 0, self.free] = q.reshape(-1, 3)
                return self.renderer(nodes, self.eye[[frame]], self.rotation[[frame]]).flatten()
            jac = torch.autograd.functional.jacobian(render_free, x[0, 0, self.free].flatten(),
                                                    vectorize=True, strategy="forward-mode")
            image_columns.append(jac@self.position_map[frame]/observations.sigma)
        images = torch.cat(image_columns, 0).detach()
        dynamics = torch.autograd.functional.jacobian(lambda q: self.residual(q[None])[0], mean,
                                                     vectorize=True, strategy="forward-mode").detach()
        return images, dynamics

    def information(self, mean, observations):
        images, dynamics = self.jacobians(mean, observations)
        prior = mean.new_zeros(self.state_dimension, self.state_dimension)
        prior[:8, :8] = torch.eye(8, dtype=mean.dtype)
        white = dynamics/self.elimination.sd[:, None]
        cross = -white.T@self.elimination.white_mapping
        explicit = prior+images.T@images+white.T@white
        marginal_jac = self.elimination.whiten(dynamics.T).T
        marginal = prior+images.T@images+marginal_jac.T@marginal_jac
        return marginal, explicit, cross
