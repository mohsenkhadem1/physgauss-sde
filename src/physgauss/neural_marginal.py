"""Causal state-dependent internal diffusion with exact Gaussian load removal.

The network is independently calibrated and frozen. At each kick its input
uses x_j and the PRECEDING stored velocity, never u_j or the force coefficients.
Q(z) therefore varies along an inferred path while the causal density Jacobian
and exact conditional Gaussian elimination remain valid. A declared free-node
floor is retained; balanced edge noise alone is singular on this toy mesh.
"""
from dataclasses import dataclass
import math
import torch
from .accurate_pipeline import AccurateViewModel, dynamics_jacobian
from .physics import edge_features, material_response
from .stochastic import spatial_basis


@dataclass
class InternalNoiseConfig:
    mode: str = "neural"
    rank: int = 3
    spatial_length_m: float = .45
    constant_amplitude: float = .0025

    def validate(self):
        if self.mode not in ("floor", "constant", "neural"):
            raise ValueError("Unknown internal diffusion model.")
        if self.rank < 1 or min(self.spatial_length_m, self.constant_amplitude) <= 0:
            raise ValueError("Need positive spatial/noise settings.")
        if not all(math.isfinite(v) for v in (self.spatial_length_m, self.constant_amplitude)):
            raise ValueError("Noise settings must be finite.")
        return self


class BlockLoadElimination:
    """Batched block-diagonal Q plus global low-rank G G^T; no Q freezing."""
    def __init__(self, mapping, covariance):
        # mapping [steps,dof,force]; covariance [batch,steps,dof,dof].
        self.mapping = mapping
        self.covariance = covariance
        self.noise_cholesky = torch.linalg.cholesky(covariance)
        self.white_mapping = torch.linalg.solve_triangular(self.noise_cholesky, mapping[None], upper=False)
        self.precision = (torch.eye(mapping.shape[-1], dtype=mapping.dtype)
                          +torch.einsum("bsdk,bsdl->bkl", self.white_mapping, self.white_mapping))
        self.cholesky = torch.linalg.cholesky(self.precision)
        self.conditional_covariance = torch.cholesky_inverse(self.cholesky)
        self.logdet_q = 2*self.noise_cholesky.diagonal(dim1=-2, dim2=-1).log().sum((-1, -2))
        self.logdet = self.logdet_q+2*self.cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    def whiten_q(self, residual):
        shape = self.covariance.shape
        return torch.linalg.solve_triangular(self.noise_cholesky,
            residual.reshape(shape[0], shape[1], shape[2], 1), upper=False).squeeze(-1)

    def conditional_mean(self, residual):
        white = self.whiten_q(residual)
        rhs = torch.einsum("bsdk,bsd->bk", self.white_mapping, white)
        return torch.cholesky_solve(rhs[..., None], self.cholesky).squeeze(-1)

    def marginal_nll(self, residual):
        white = self.whiten_q(residual)
        rhs = torch.einsum("bsdk,bsd->bk", self.white_mapping, white)
        mean = torch.cholesky_solve(rhs[..., None], self.cholesky).squeeze(-1)
        profiled = white-torch.einsum("bsdk,bk->bsd", self.white_mapping, mean)
        square = profiled.square().sum((-1, -2))+mean.square().sum(-1)
        n = self.mapping.shape[0]*self.mapping.shape[1]
        return .5*(square+self.logdet+n*math.log(2*math.pi))

    def joint_nll(self, residual, coefficients):
        white = self.whiten_q(residual)-torch.einsum("bsdk,bk->bsd", self.white_mapping, coefficients)
        n = self.mapping.shape[0]*self.mapping.shape[1]+self.mapping.shape[2]
        return .5*(white.square().sum((-1, -2))+coefficients.square().sum(-1)+self.logdet_q+n*math.log(2*math.pi))

    def mean_information(self, jacobian):
        if self.covariance.shape[0] != 1:
            raise ValueError("Information is evaluated at one trajectory.")
        steps, dof = self.mapping.shape[:2]
        white = torch.linalg.solve_triangular(self.noise_cholesky[0],
                    jacobian.reshape(steps, dof, -1), upper=False).flatten(0, 1)
        mapping = self.white_mapping[0].flatten(0, 1)
        u, singular, _ = torch.linalg.svd(mapping, full_matrices=False)
        # Symmetric C^{-1/2} in Q-whitened coordinates, used only as a metric.
        white = white+u@((torch.rsqrt(1+singular.square())-1)[:, None]*(u.T@white))
        return white.T@white


def covariance_information(elimination, local_derivatives, state_map):
    """Exact .5 tr(C^-1 C_,p C^-1 C_,q) without dense C derivatives.

    dQ_j/dlocal [steps,local,dof,dof], dlocal_j/dz [steps,local,state].
    Expand (I - W A^-1 W^T) on both sides of block-diagonal dQ.
    The two local terms plus a global force-size feature product are exact.
    """
    chol = elimination.noise_cholesky[0]
    d = torch.linalg.solve_triangular(chol[:, None], local_derivatives, upper=False)
    d = torch.linalg.solve_triangular(chol[:, None], d.transpose(-1, -2), upper=False).transpose(-1, -2)
    w = elimination.white_mapping[0]
    steps, dof, force = w.shape
    v = torch.linalg.solve_triangular(elimination.cholesky[0],
                    w.flatten(0, 1).T, upper=False).T.reshape(steps, dof, force)
    influence = v@v.transpose(-1, -2)
    local = (.5*torch.einsum("spij,sqij->spq", d, d)
             -torch.einsum("spij,sqij->spq", influence[:, None]@d, d))
    flattened_map = state_map.flatten(0, 1)
    metric = flattened_map.T@(local@state_map).flatten(0, 1)
    # V_j^T dQ_white V_j is a force-by-force matrix per local coordinate.
    features = v[:, None].transpose(-1, -2)@d@v[:, None]
    global_features = torch.einsum("spab,spz->abz", features, state_map).flatten(0, 1)
    metric = metric+.5*global_features.T@global_features
    return .5*(metric+metric.T)


class NeuralMarginalModel(AccurateViewModel):
    def __init__(self, config, network, noise_config=InternalNoiseConfig(), **kwargs):
        super().__init__(config, **kwargs)
        self.internal_config = noise_config.validate()
        self.network = network
        if network is not None:
            network.eval()
            for parameter in network.parameters():
                parameter.requires_grad_(False)
        if self.internal_config.mode == "neural" and network is None:
            raise ValueError("Neural diffusion requires frozen calibrated weights.")
        self.internal_basis = spatial_basis(self.mesh, self.internal_config.rank,
                                           self.internal_config.spatial_length_m)
        incidence = self.times.new_zeros(len(self.mesh.rest), len(self.mesh.edges))
        for e, (i, j) in enumerate(self.mesh.edges.tolist()):
            incidence[i, e] = 1.; incidence[j, e] = -1.
        self.edge_incidence = incidence
        self.load_mapping = self.elimination.mapping.reshape(self.nsteps, self.free_dof, self.force_dimension)

    def internal_factor(self, x, previous):
        """Velocity diffusion [batch,steps,node,xyz,rank], before clamping."""
        features, direction = edge_features(x, previous, self.mesh)
        if self.internal_config.mode == "floor":
            return x.new_zeros(*x.shape[:-1], 3, self.internal_config.rank)
        amplitude = (self.network(features) if self.internal_config.mode == "neural"
                     else torch.full_like(features[..., 0], self.internal_config.constant_amplitude))
        impulse = torch.einsum("ne,bsec,bse,er->bsncr", self.edge_incidence,
                               direction, amplitude, self.internal_basis)
        return impulse/self.mesh.mass[None, None, :, None, None]

    def covariance_at_nodes(self, x, previous):
        factor = self.internal_factor(x, previous)[:, :, self.free].flatten(2, 3)
        floor = self.marginal_config.velocity_diffusion**2*torch.eye(self.free_dof, dtype=x.dtype)
        return self.kick_sizes[None, :, None, None]*(factor@factor.transpose(-1, -2)+floor)

    def noise_covariance(self, state):
        full = self.step_trajectory(state)
        return self.covariance_at_nodes(full.x[:, :-1], full.v[:, :-1])

    def elimination_at(self, state):
        return BlockLoadElimination(self.load_mapping, self.noise_covariance(state))

    def dynamics_objective(self, state, coefficients=None):
        """Normalized path/parameter density, including force-volume terms."""
        residual = self.residual(state)
        elimination = self.elimination_at(state)
        dynamics = (elimination.marginal_nll(residual) if coefficients is None
                    else elimination.joint_nll(residual, coefficients))
        return (dynamics+.5*state[:, :8].square().sum(-1)
                -self.log_dynamic_jacobian(state)
                -(self.state_dimension-8)*math.log(self.marginal_config.velocity_coordinate_scale))

    def objective(self, state, observations, coefficients=None):
        return self.dynamics_objective(state, coefficients)+.5*self.image_residual(state, observations).square().sum(-1)

    def local_covariance_derivatives(self, mean):
        with torch.no_grad():
            full = self.step_trajectory(mean[None])
            x, previous = full.x[:, :-1].detach(), full.v[:, :-1].detach()
        def covariance(shift):
            a, b = shift.split(self.free_dof)
            nodes, velocity = x.clone(), previous.clone()
            nodes[:, :, self.free] = nodes[:, :, self.free]+a.reshape(1, 1, -1, 3)
            velocity[:, :, self.free] = velocity[:, :, self.free]+b.reshape(1, 1, -1, 3)
            return self.covariance_at_nodes(nodes, velocity)[0]
        derivative = torch.autograd.functional.jacobian(covariance, mean.new_zeros(2*self.free_dof),
                                                        vectorize=True, strategy="forward-mode").detach()
        return derivative.permute(0, 3, 1, 2)

    def local_state_map(self, mean):
        if not hasattr(self, "noise_state_map"):
            # dynamics_jacobian creates solver_position_map without using noise.
            if not hasattr(self, "solver_position_map"):
                dynamics_jacobian(self, mean)
            velocity_map = torch.zeros_like(self.solver_position_map)
            for node, height in enumerate(self.height[self.free]):
                velocity_map[0, 3*node:3*node+3, 5:8] = torch.diag(self.prior.std[5:8]*height)
            rows = torch.arange((self.nsteps-1)*self.free_dof)
            velocity_map[1:].reshape(-1, self.state_dimension)[rows, 8+rows] = self.marginal_config.velocity_coordinate_scale
            self.noise_state_map = torch.cat((self.solver_position_map, velocity_map), 1)
        return self.noise_state_map

    def fisher_precision(self, mean, observations):
        # Reuse the verified fixed-Q image/prior information; replace its mean
        # dynamics metric and ADD the covariance derivative information.
        baseline, _, _ = super().information(mean, observations)
        if self.internal_config.mode == "floor":
            return baseline, {"covariance_information_trace": 0.}
        jacobian = dynamics_jacobian(self, mean)
        with torch.no_grad():
            elimination = self.elimination_at(mean[None])
            base_j = self.elimination.whiten(jacobian.T).T
            mean_metric = elimination.mean_information(jacobian)
        derivatives = self.local_covariance_derivatives(mean)
        with torch.no_grad():
            cov_metric = covariance_information(elimination, derivatives, self.local_state_map(mean))
            precision = baseline-base_j.T@base_j+mean_metric+cov_metric
            precision = .5*(precision+precision.T)
        return precision, {"covariance_information_trace": float(cov_metric.trace())}

    def covariance_rollout(self, theta, coefficients, white):
        """Causal normalized simulator for density/innovation checks.

        A full-rank Cholesky draw is equivalent in distribution to independent
        balanced internal and floor increments. Supports are clamped explicitly.
        """
        young, drag, offset, v0 = self.prior.decode(theta)
        x = self.mesh.rest[None]+self.height[None, :, None]*offset[:, None]
        previous = self.height[None, :, None]*v0[:, None]
        external = self.force_prior.nodal_force(coefficients, self.step_times)
        stored = []
        for step, kick in enumerate(self.kick_sizes):
            force = material_response(x, self.mesh, young)[0]
            acceleration = (force+external[:, step])/self.mesh.mass[None, :, None]+x.new_tensor([0., -.6, 0.])
            factor = self.internal_factor(x[:, None], previous[:, None])[:, 0, self.free].flatten(1, 2)
            cov = kick*(factor@factor.transpose(-1, -2)+self.marginal_config.velocity_diffusion**2*torch.eye(self.free_dof, dtype=x.dtype))
            noise = torch.zeros_like(previous)
            noise[:, self.free] = (torch.linalg.cholesky(cov)@white[:, step, :, None]).squeeze(-1).reshape(len(theta), -1, 3)
            c = drag[:, None, None]*kick/2
            u = ((1-c)*previous+kick*acceleration+noise)/(1+c)
            u = torch.where(self.mesh.pinned[None, :, None], 0., u)
            x = x+self.step_sizes[step]*u
            stored.append(u); previous = u
        return self.encode_velocities(theta, torch.stack(stored, 1))

    def internal_rollout(self, theta, coefficients, increments, external_acceleration=None):
        """Production staggered kicks driven by specified internal Brownian increments.

        Used for coupled integrator audits with zero floor realization. Input
        increment variances are kick_sizes, including the first half kick.
        """
        if increments.shape != (len(theta), self.nsteps, self.internal_config.rank):
            raise ValueError("Incorrect internal Brownian increment shape.")
        young, drag, offset, initial_v = self.prior.decode(theta)
        x = self.mesh.rest[None]+self.height[None, :, None]*offset[:, None]
        previous = self.height[None, :, None]*initial_v[:, None]
        external = self.force_prior.nodal_force(coefficients, self.step_times)
        stored = []
        for step, kick in enumerate(self.kick_sizes):
            internal = material_response(x, self.mesh, young)[0]
            acceleration = (internal+external[:, step])/self.mesh.mass[None, :, None]+x.new_tensor([0., -.6, 0.])
            if external_acceleration is not None:
                acceleration = acceleration+external_acceleration(float(self.step_times[step]), x, previous)
            factor = self.internal_factor(x[:, None], previous[:, None])[:, 0]
            noise = torch.einsum("bncr,br->bnc", factor, increments[:, step])
            c = drag[:, None, None]*kick/2
            u = ((1-c)*previous+kick*acceleration+noise)/(1+c)
            u = torch.where(self.mesh.pinned[None, :, None], 0., u)
            x = x+self.step_sizes[step]*u
            stored.append(u); previous = u
        state = self.encode_velocities(theta, torch.stack(stored, 1))
        self.residual(state)
        return state
