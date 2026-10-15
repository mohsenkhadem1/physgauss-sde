"""Differentiable tetrahedral Neo-Hookean mechanics.

All tensors use SI units. Nodal masses are fixed from reference volume and
density. The demo has a clamped base, gravity, and external linear drag.
This is a small FEM discretization, not an MPM implementation.
"""
from dataclasses import dataclass
from itertools import combinations, permutations
import math

import torch
from torch import Tensor


class InvalidDeformation(RuntimeError):
    """An inverted/collapsed element must not be hidden by clamping log(det F)."""


@dataclass
class Mesh:
    rest: Tensor
    cells: Tensor
    inv_rest: Tensor
    volumes: Tensor
    mass: Tensor
    pinned: Tensor
    edges: Tensor
    edge_lengths: Tensor


def cube_mesh(dtype=torch.float64, density=1.0, side=0.65) -> Mesh:
    if density <= 0 or side <= 0:
        raise ValueError("Density and side length must be positive.")
    rest = torch.tensor(
        [[i * side - side / 2, j * side + 0.20, k * side - side / 2]
         for i in range(2) for j in range(2) for k in range(2)], dtype=dtype,
    )
    # Six tetrahedra around the cube's body diagonal, with positive orientation.
    cells = []
    for order in permutations(range(3)):
        xyz = [0, 0, 0]
        path = [0]
        for axis in order:
            xyz[axis] = 1
            path.append(xyz[0] * 4 + xyz[1] * 2 + xyz[2])
        nodes = rest[path]
        if torch.linalg.det((nodes[1:] - nodes[:1]).T) < 0:
            path[1], path[2] = path[2], path[1]
        cells.append(path)
    cells = torch.tensor(cells, dtype=torch.long)
    dm = (rest[cells[:, 1:]] - rest[cells[:, :1]]).transpose(-1, -2)
    volumes = torch.linalg.det(dm) / 6
    mass = torch.zeros(len(rest), dtype=dtype)
    mass.index_add_(0, cells.flatten(), (density * volumes[:, None].expand(-1, 4) / 4).flatten())
    edges = torch.tensor(sorted({tuple(sorted(e)) for c in cells.tolist()
                                 for e in combinations(c, 2)}), dtype=torch.long)
    lengths = torch.linalg.vector_norm(rest[edges[:, 1]] - rest[edges[:, 0]], dim=-1)
    return Mesh(rest, cells, torch.linalg.inv(dm), volumes, mass,
                rest[:, 1] == rest[:, 1].min(), edges, lengths)


def deformation(x: Tensor, mesh: Mesh) -> Tensor:
    ds = (x[..., mesh.cells[:, 1:], :] - x[..., mesh.cells[:, :1], :]).transpose(-1, -2)
    return ds @ mesh.inv_rest


def determinant_and_cofactor(f: Tensor):
    a, b, c = f.unbind(-1)
    cof = torch.stack((torch.linalg.cross(b, c), torch.linalg.cross(c, a),
                       torch.linalg.cross(a, b)), dim=-1)
    return (a * cof[..., 0]).sum(-1), cof


def material_response(x: Tensor, mesh: Mesh, young: Tensor, poisson=0.30):
    if not -1.0 < poisson < 0.5:
        raise ValueError("Poisson ratio must lie in (-1, 0.5).")
    f = deformation(x, mesh)
    j, cof = determinant_and_cofactor(f)
    if not bool(torch.isfinite(j).all()) or bool((j <= 1e-5).any()):
        raise InvalidDeformation("Nonpositive/near-zero det(F); reduce step size or tighten the prior.")
    young = torch.as_tensor(young, dtype=x.dtype, device=x.device)
    if bool((young <= 0).any()):
        raise ValueError("Young's modulus must be positive.")
    mu = young[..., None] / (2 * (1 + poisson))
    lam = young[..., None] * poisson / ((1 + poisson) * (1 - 2 * poisson))
    log_j = j.log()
    energy_density = (0.5 * mu * (f.square().sum((-2, -1)) - 3)
                      - mu * log_j + 0.5 * lam * log_j.square())
    p = mu[..., None, None] * f + ((lam * log_j - mu) / j)[..., None, None] * cof
    h = -mesh.volumes[..., None, None] * (p @ mesh.inv_rest.transpose(-1, -2))
    forces = torch.zeros_like(x)
    forces = forces.index_add(-2, mesh.cells[:, 0], -h.sum(-1))
    for local in range(3):
        forces = forces.index_add(-2, mesh.cells[:, local + 1], h[..., local])
    return forces, (energy_density * mesh.volumes).sum(-1), j


@dataclass
class Trajectory:
    x: Tensor
    v: Tensor


def simulate(mesh: Mesh, times: Tensor, young: Tensor, drag: Tensor,
             offset: Tensor, initial_velocity: Tensor, substeps=3,
             gravity=-0.6, white: Tensor | None = None, diffusion=None,
             noise_basis: Tensor | None = None, external_acceleration=None) -> Trajectory:
    """Semi-implicit Euler; stochastic impulses use the pre-step state (Ito).

    Shapes: parameters [B], offset/velocity [B,3], white [B,steps,rank].
    White increments are standardized: this routine multiplies by sqrt(dt).
    Both sides of each edge receive opposite central impulses. Clamped nodes
    exchange momentum with the support; the object is not an isolated system.
    """
    if substeps < 1 or times.ndim != 1 or len(times) < 2:
        raise ValueError("Need at least two times and a positive substep count.")
    if not bool(((times[1:] - times[:-1]) > 0).all()):
        raise ValueError("Times must be strictly increasing.")
    batch = offset.shape[0]
    n_steps = (len(times) - 1) * substeps
    if white is not None:
        if diffusion is None or noise_basis is None:
            raise ValueError("Noise needs a diffusion field and spatial basis.")
        if white.shape != (batch, n_steps, noise_basis.shape[1]):
            raise ValueError("Incorrect number of stochastic increments.")
    height = (mesh.rest[:, 1] - mesh.rest[:, 1].min())
    height = height / height.max()
    x = mesh.rest[None] + height[None, :, None] * offset[:, None]
    v = height[None, :, None] * initial_velocity[:, None]
    xs, vs = [x], [v]
    g = x.new_tensor([0.0, gravity, 0.0])
    step = 0
    current_time = float(times[0])
    for interval in range(len(times) - 1):
        dt = float(times[interval + 1] - times[interval]) / substeps
        for _ in range(substeps):
            force, _, _ = material_response(x, mesh, young)
            acceleration = force / mesh.mass[None, :, None] + g - drag[:, None, None] * v
            if external_acceleration is not None:
                acceleration = acceleration + external_acceleration(current_time, x, v)
            impulse = torch.zeros_like(x)
            if white is not None:
                impulse = edge_impulses(x, v, mesh, white[:, step] * math.sqrt(dt),
                                        diffusion, noise_basis)
            v_new = v + dt * acceleration + impulse / mesh.mass[None, :, None]
            v = torch.where(mesh.pinned[None, :, None], torch.zeros_like(v_new), v_new)
            x_new = x + dt * v
            x = torch.where(mesh.pinned[None, :, None], mesh.rest[None], x_new)
            step += 1
            current_time += dt
        xs.append(x)
        vs.append(v)
    # Validate the final state as well as every force-evaluation state.
    material_response(x, mesh, young)
    return Trajectory(torch.stack(xs, dim=1), torch.stack(vs, dim=1))


def edge_features(x: Tensor, v: Tensor, mesh: Mesh):
    edge = x[..., mesh.edges[:, 1], :] - x[..., mesh.edges[:, 0], :]
    length = torch.linalg.vector_norm(edge, dim=-1)
    direction = edge / length.clamp_min(1e-10)[..., None]
    rel_v = v[..., mesh.edges[:, 1], :] - v[..., mesh.edges[:, 0], :]
    radial_speed = (rel_v * direction).sum(-1)
    return torch.stack((length / mesh.edge_lengths - 1, radial_speed), dim=-1), direction


def edge_impulses(x: Tensor, v: Tensor, mesh: Mesh, dw: Tensor,
                  diffusion, basis: Tensor):
    features, direction = edge_features(x, v, mesh)
    amplitudes = diffusion(features)
    scalar = amplitudes * (dw @ basis.T)
    edge_impulse = scalar[..., None] * direction
    impulse = torch.zeros_like(x)
    impulse = impulse.index_add(-2, mesh.edges[:, 0], edge_impulse)
    return impulse.index_add(-2, mesh.edges[:, 1], -edge_impulse)
