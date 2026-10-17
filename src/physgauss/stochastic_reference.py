"""Independent fine-step Ito reference with coupled Brownian refinement.

Uses the same FEM constitutive law, but does not call either inference
integrator or physics.simulate. Central edge impulses are assembled directly.
The symplectic Euler-Maruyama reference has first-order deterministic drift
accuracy, so its production grid is very fine; stochastic accuracy is audited
by coupled refinement rather than inferred from deterministic second order.
"""
import math
import torch
from .physics import cube_mesh, material_response
from .stochastic import spatial_basis


def staggered_increments(finest, steps):
    """Couple midpoint velocity kicks to one finest Brownian path.

    u_j represents velocity near (j+1/2)h: first noise interval [0,h/2],
    then [(j-1/2)h,(j+1/2)h]. The last half interval is unused by x_N.
    """
    if finest.shape[0] % steps or (finest.shape[0]//steps) % 2:
        raise ValueError("Need an even nested Brownian stride for staggered kicks.")
    stride = finest.shape[0]//steps
    cumulative = torch.cat((torch.zeros_like(finest[:1]), finest.cumsum(0)), 0)
    end = (torch.arange(steps)+1)*stride-stride//2
    start = torch.cat((end.new_zeros(1), end[:-1]))
    return cumulative[end]-cumulative[start]


@torch.no_grad()
def reference_path(times, brownian, substeps, pulse=False, stochastic=True):
    mesh = cube_mesh(dtype=times.dtype)
    steps = (len(times)-1)*substeps
    if brownian.shape[0] % steps or brownian.shape[1] != 3:
        raise ValueError("Brownian reference must have a nested rank-three grid.")
    stride = brownian.shape[0]//steps
    increments = brownian.reshape(steps, stride, 3).sum(1)
    dt = float(times[-1]-times[0])/steps
    basis = spatial_basis(mesh, 3, .45)
    free = ~mesh.pinned
    x = mesh.rest.clone(); x[free] += x.new_tensor([.11, -.035, .06])
    v = torch.zeros_like(x); v[free] = x.new_tensor([-.12, .04, .10])
    young = x.new_tensor(19.5)
    positions, velocities = [x.clone()], [v.clone()]
    minimum_det, maximum_net, maximum_torque, maximum_momentum = float("inf"), 0., 0., 0.
    edges = mesh.edges
    for step in range(steps):
        internal, _, det = material_response(x, mesh, young)
        minimum_det = min(minimum_det, float(det.min()))
        strength = math.exp(-.5*((step*dt-.5)/.08)**2) if pulse else 0.
        external_acceleration = strength*free[:, None]*x.new_tensor([.45, 0., -.30])
        drift = internal/mesh.mass[:, None]+x.new_tensor([0., -.6, 0.])-.55*v+external_acceleration
        impulse = torch.zeros_like(x)
        if stochastic:
            edge = x[edges[:, 1]]-x[edges[:, 0]]
            length = edge.norm(dim=-1)
            direction = edge/length[:, None]
            strain = length/mesh.edge_lengths-1
            speed = ((v[edges[:, 1]]-v[edges[:, 0]])*direction).sum(-1)
            # Direct independent implementation of the declared analytic truth
            # law, avoiding calls to the fitted network or its feature helper.
            amplitude = .0017*(1+2*(strain.square()+.001**2).sqrt()+.18*(speed.square()+.001**2).sqrt())
            edge_impulse = (amplitude*(basis@increments[step]))[:, None]*direction
            impulse.index_add_(0, edges[:, 0], edge_impulse)
            impulse.index_add_(0, edges[:, 1], -edge_impulse)
            maximum_net = max(maximum_net, float(impulse.sum(0).abs().max()))
            maximum_torque = max(maximum_torque, float(torch.linalg.cross(x, impulse).sum(0).abs().max()))
        next_v = v+dt*drift+impulse/mesh.mass[:, None]
        next_v[mesh.pinned] = 0.
        maximum_momentum = max(maximum_momentum, float((mesh.mass[free, None]*(next_v[free]-v[free])
            -dt*mesh.mass[free, None]*drift[free]-impulse[free]).abs().max()))
        x = x+dt*next_v; x[mesh.pinned] = mesh.rest[mesh.pinned]
        v = next_v
        if (step+1) % substeps == 0:
            positions.append(x.clone()); velocities.append(v.clone())
    minimum_det = min(minimum_det, float(material_response(x, mesh, young)[2].min()))
    audit = {"substeps": substeps, "dt_s": dt, "minimum_det_F": minimum_det,
        "maximum_internal_net_impulse_Ns": maximum_net,
        "maximum_internal_impulse_torque_Nms": maximum_torque,
        "maximum_free_momentum_residual_Ns": maximum_momentum,
        "support_position_error_m": float((x[mesh.pinned]-mesh.rest[mesh.pinned]).abs().max()),
        "brownian_quadratic_variation_per_mode_s": increments.square().sum(0).tolist()}
    return torch.stack(positions), torch.stack(velocities), audit


@torch.no_grad()
def coupled_reference(times, seed, pulse=False, fine_substeps=768, tolerance_m=.00025):
    if fine_substeps < 4 or fine_substeps % 4:
        raise ValueError("Fine grid must be divisible by four.")
    steps = (len(times)-1)*fine_substeps
    dt = float(times[-1]-times[0])/steps
    generator = torch.Generator().manual_seed(seed+700000)
    brownian = math.sqrt(dt)*torch.randn(steps, 3, dtype=times.dtype, generator=generator)
    grids = [fine_substeps//4, fine_substeps//2, fine_substeps]
    paths, audit = [], []
    for grid in grids:
        x, v, record = reference_path(times, brownian, grid, pulse=pulse)
        paths.append((x, v)); audit.append(record)
    free = ~cube_mesh(dtype=times.dtype).pinned
    changes = [float((a[0][:, free]-b[0][:, free]).square().mean().sqrt()) for a, b in zip(paths[:-1], paths[1:])]
    if changes[-1] > tolerance_m:
        raise RuntimeError(f"Stochastic coupled refinement exceeds declared tolerance: {changes[-1]} m")
    if any(r["maximum_internal_net_impulse_Ns"] > 1e-12 or r["maximum_internal_impulse_torque_Nms"] > 1e-12
           or r["maximum_free_momentum_residual_Ns"] > 1e-12 or r["support_position_error_m"] != 0 for r in audit):
        raise RuntimeError("Stochastic reference balance/constraint audit failed.")
    return {"positions": paths[-1][0], "velocities": paths[-1][1], "brownian_finest_increments": brownian,
            "audit": {"integrator": "independent symplectic Euler-Maruyama, analytic state-dependent edge diffusion",
                "seed": seed, "pulse": pulse, "grids": audit, "successive_coordinate_rmse_changes_m": changes,
                "last_change_tolerance_m": tolerance_m,
                "qualification": "coupled finite-grid consistency check; not a rigorous absolute SDE error bound, mesh convergence, or experimental validation"}}
