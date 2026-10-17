"""Known-camera observations and efficient exact first derivatives of dynamics."""
import torch
from .accurate_dynamics import StaggeredForceModel
from .camera import orbit
from .marginal_forces import TrajectoryForceModel
from .physics import material_response


def dynamics_jacobian(model, mean):
    """Assemble causal blocks using AD of 12 local spatial coordinates.

    Exploits fixed Neo-Hookean E scaling, known mass and linear drag. Tested
    against independent full-path AD for both implemented integrators.
    """
    with torch.no_grad():
        full = model.step_trajectory(mean[None])
        young, drag, _, _ = model.prior.decode(mean[None])
        nodes = full.x[:, :-1].detach()
        internal = material_response(nodes, model.mesh, young[:, None])[0]
    def accelerations(shift):
        shifted = nodes.clone()
        shifted[:, :, model.free] = shifted[:, :, model.free]+shift.reshape(1, 1, -1, 3)
        f = material_response(shifted, model.mesh, young[:, None])[0]
        return (f/model.mesh.mass[None, None, :, None])[:, :, model.free].reshape(model.nsteps, model.free_dof)
    local = torch.autograd.functional.jacobian(accelerations, mean.new_zeros(model.free_dof),
                                              vectorize=True, strategy="forward-mode").detach()
    if not hasattr(model, "solver_position_map"):
        mapping = mean.new_zeros(model.nsteps, model.free_dof, model.state_dimension)
        offset = mean.new_zeros(model.free_dof, 3)
        for node, height in enumerate(model.height[model.free]):
            offset[3*node:3*node+3] = torch.diag(model.prior.std[2:5]*height)
        mapping[:, :, 2:5] = offset[None]
        temporal = torch.tril(torch.ones(model.nsteps, model.nsteps, dtype=mean.dtype), diagonal=-1)*model.step_sizes[None]
        mapping[:, :, 8:] = torch.kron(temporal.contiguous(), torch.eye(model.free_dof, dtype=mean.dtype)).reshape(model.nsteps, model.free_dof, -1)*model.marginal_config.velocity_coordinate_scale
        model.solver_position_map = mapping
    staggered = isinstance(model, StaggeredForceModel)
    kick = model.kick_sizes if staggered else model.step_sizes
    gamma = drag[0]
    plus = 1+gamma*kick/2 if staggered else torch.ones_like(kick)
    minus = 1-gamma*kick/2 if staggered else 1-gamma*kick
    jac = -kick[:, None, None]*(local@model.solver_position_map)
    jac = jac.reshape(-1, model.state_dimension)
    rows = torch.arange(model.nsteps*model.free_dof)
    scale = model.marginal_config.velocity_coordinate_scale
    jac[rows, 8+rows] += plus.repeat_interleave(model.free_dof)*scale
    previous = rows[model.free_dof:]
    jac[previous, 8+previous-model.free_dof] -= minus[1:].repeat_interleave(model.free_dof)*scale
    initial_map = mean.new_zeros(model.free_dof, 3)
    for node, height in enumerate(model.height[model.free]):
        initial_map[3*node:3*node+3] = torch.diag(model.prior.std[5:8]*height)
    jac[:model.free_dof, 5:8] -= minus[0]*initial_map
    acceleration = (internal/model.mesh.mass[None, None, :, None])[:, :, model.free][0].reshape(model.nsteps, -1)
    jac[:, 0] += (-kick[:, None]*acceleration*model.prior.std[0]).flatten()
    u = full.v[0, 1:, model.free].reshape(model.nsteps, -1)
    prev = full.v[0, :-1, model.free].reshape(model.nsteps, -1)
    drag_derivative = (gamma*kick[:, None]*(u+prev)/2 if staggered else gamma*kick[:, None]*prev)
    jac[:, 1] += drag_derivative.flatten()*model.prior.std[1]
    return jac.detach()


class KnownViews:
    def set_views(self, specs):
        self.view_specs = list(specs)
        self.view_frames = torch.tensor([frame for frame, _ in specs], dtype=torch.long)
        cameras = {offset: orbit(self.times, offset_degrees=offset) for _, offset in specs}
        self.view_eye = torch.stack([cameras[offset][0][frame] for frame, offset in specs])
        self.view_rotation = torch.stack([cameras[offset][1][frame] for frame, offset in specs])
        return self

    def predict(self, state, indices=None):
        if indices is None:
            return super().predict(state)
        if not torch.equal(indices, self.view_frames):
            raise ValueError("Observation slots do not match the declared cameras.")
        return self.renderer(self.trajectory(state).x[:, indices], self.view_eye, self.view_rotation)

    def information(self, mean, observations):
        with torch.no_grad():
            self.residual(mean[None])
            frames = self.trajectory(mean[None]).x
        image_h = mean.new_zeros(self.state_dimension, self.state_dimension)
        for slot, (frame, _) in enumerate(self.view_specs):
            x = frames[:, [frame]].clone()
            def render(q):
                nodes = x.clone(); nodes[0, 0, self.free] = q.reshape(-1, 3)
                return self.renderer(nodes, self.view_eye[[slot]], self.view_rotation[[slot]]).flatten()
            j = torch.autograd.functional.jacobian(render, x[0, 0, self.free].flatten(),
                                                  vectorize=True, strategy="forward-mode")/observations.sigma
            p = self.position_map[frame]
            image_h += p.T@(j.T@j)@p
        dynamics = dynamics_jacobian(self, mean)
        prior = mean.new_zeros(self.state_dimension, self.state_dimension)
        prior[:8, :8] = torch.eye(8, dtype=mean.dtype)
        white = dynamics/self.elimination.sd[:, None]
        cross = -white.T@self.elimination.white_mapping
        explicit = prior+image_h+white.T@white
        marginal_j = self.elimination.whiten(dynamics.T).T
        marginal = prior+image_h+marginal_j.T@marginal_j
        if isinstance(self, StaggeredForceModel):
            curvature = self.normalizer_curvature(mean)
            marginal[1, 1] += curvature; explicit[1, 1] += curvature
        return marginal, explicit, cross


class AccurateViewModel(KnownViews, StaggeredForceModel):
    pass


class LegacyViewModel(KnownViews, TrajectoryForceModel):
    pass
