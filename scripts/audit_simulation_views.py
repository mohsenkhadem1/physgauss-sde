"""Visual evidence and independent simulation/view-coverage audit.

Uses exact published input images and fitted trajectories. No images are
invented/upscaled by a generative model. Native RGB is nearest-neighbor enlarged.
"""
import argparse
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from physgauss.benchmark import Config, Observations, make_benchmark
from physgauss.camera import orbit, project
from physgauss.forces import ForceConfig
from physgauss.marginal_forces import MarginalConfig, TrajectoryForceModel
from physgauss.marginal_inference import TrajectoryFitConfig, fit_trajectory
from physgauss.physics import (cube_mesh, material_response, simulate, edge_impulses,
                               deformation, InvalidDeformation)
from physgauss.stochastic import calibration_law, spatial_basis
from physgauss.study import write_json


CONDITIONS = {"no_load": (False, False), "external_pulse": (True, False),
              "stochastic": (False, True), "pulse_and_stochastic": (True, True)}
ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def load_case(condition):
    folder = ROOT/"examples/marginal-force-study/cases"/f"{condition}__4101"
    inputs = json.loads((folder/"observations.json").read_text())
    point = json.loads((folder/"marginalized/point_estimate.json").read_text())
    config = Config(**inputs["config"])
    model = TrajectoryForceModel(config, ForceConfig(**point["force_prior"]), MarginalConfig(**point["trajectory_model"]))
    benchmark = make_benchmark(config)
    actual = torch.tensor(inputs["rgb"], dtype=torch.float64)
    digest = hashlib.sha256(actual.numpy().tobytes()).hexdigest()
    if digest != inputs["observations_sha256_float64_tensor_bytes"]:
        raise ValueError("Published input hash mismatch.")
    torch.testing.assert_close(actual, benchmark.observations.rgb, rtol=0, atol=0)
    return inputs, point, model, benchmark


@torch.no_grad()
def galleries(output):
    inputs, point, model, benchmark = load_case("pulse_and_stochastic")
    state = torch.tensor(point["state_coordinates"], dtype=torch.float64)[None]
    trajectory = model.trajectory(state)
    rgb = model.renderer(trajectory.x, model.eye, model.rotation)[0]
    selected = [0, 3, 6, 10, 14, 17, 20]
    fig, axes = plt.subplots(4, len(selected), figsize=(14, 8), constrained_layout=True)
    for col, frame in enumerate(selected):
        observed = frame in model.config.observed_frames
        axes[0, col].set_title(f"t={float(model.times[frame]):.2f}s\n{'OBSERVED' if observed else 'MISSING'}", fontsize=11)
        for row in range(4):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        if observed:
            index = list(model.config.observed_frames).index(frame)
            axes[0, col].imshow(np.clip(np.asarray(inputs["rgb"])[index], 0, 1), interpolation="nearest")
        else:
            axes[0, col].set_facecolor("#e6e9ed")
            axes[0, col].text(.5, .5, "No input", ha="center", va="center", transform=axes[0, col].transAxes)
        axes[1, col].imshow(benchmark.truth_rgb[frame].clamp(0, 1), interpolation="nearest")
        axes[2, col].imshow(rgb[frame].clamp(0, 1), interpolation="nearest")
        error = (rgb[frame]-benchmark.truth_rgb[frame]).abs().mean(-1)
        axes[3, col].imshow(error, vmin=0, vmax=.05, cmap="magma", interpolation="nearest")
        axes[3, col].set_xlabel(f"MAE={float(error.mean()):.4f}", fontsize=10)
    for row, label in enumerate(("Actual noisy input", "Simulation truth", "Marginalized MAP", "Absolute RGB error\nscale: 0–0.05")):
        axes[row, 0].set_ylabel(label, fontsize=11)
    fig.suptitle("Actual 24×24 data · mixed pulse + stochastic sequence · nearest-neighbor display enlargement", fontsize=13)
    fig.savefig(output/"input_reconstruction.png", dpi=145); plt.close(fig)
    # Same missing timestamp viewed from known and unseen cameras. High-res
    # truth is not used: every tile below is rendered at the original 24 pixels.
    frame = 10
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5), constrained_layout=True)
    offsets = [0, 100, 200, 300]
    for col, offset in enumerate(offsets):
        eye, rotation = orbit(model.times, offset_degrees=offset)
        truth = model.renderer(benchmark.truth.x[:, [frame]], eye[[frame]], rotation[[frame]])[0, 0]
        predicted = model.renderer(trajectory.x[:, [frame]], eye[[frame]], rotation[[frame]])[0, 0]
        axes[0, col].set_title(f"Camera offset {offset}°\n{'recording path' if offset == 0 else 'unseen view'}")
        axes[0, col].imshow(truth.clamp(0, 1), interpolation="nearest")
        axes[1, col].imshow(predicted.clamp(0, 1), interpolation="nearest")
        axes[2, col].imshow((truth-predicted).abs().mean(-1), cmap="magma", vmin=0, vmax=.05, interpolation="nearest")
        for row in range(3):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    for row, label in enumerate(("Truth", "Reconstruction", "Absolute RGB error\nscale: 0–0.05")):
        axes[row, 0].set_ylabel(label)
    fig.suptitle("Missing t=0.50 s · same physical state from four directions · evaluation only", fontsize=13)
    fig.savefig(output/"unseen_views.png", dpi=145); plt.close(fig)
    # A wireframe makes the actual volumetric object and camera geometry clear.
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax = fig.add_subplot(121, projection="3d")
    for nodes, color, label in ((benchmark.truth.x[0, frame], "#202a34", "truth"),
                                (trajectory.x[0, frame], "#15808b", "reconstruction")):
        for i, (a, b) in enumerate(model.mesh.edges.tolist()):
            edge = nodes[[a, b]].numpy()
            ax.plot(edge[:, 0], edge[:, 2], edge[:, 1], color=color, alpha=.65, label=label if i == 0 else None)
    pins = model.mesh.rest[model.mesh.pinned].numpy()
    ax.scatter(pins[:, 0], pins[:, 2], pins[:, 1], color="#b66423", s=35, label="clamped nodes")
    ax.set_xlabel("world x [m]"); ax.set_ylabel("world z [m]"); ax.set_zlabel("height y [m]")
    ax.set_title("Physical mesh at missing t=0.50 s · true scale")
    ax.set_box_aspect((1, 1, 1)); ax.legend(fontsize=8)
    ax = fig.add_subplot(122)
    eye = model.eye.numpy()
    ax.plot(eye[:, 0], eye[:, 2], color="#9ca7b0", label="recording camera arc")
    indices = list(model.config.observed_frames)
    ax.scatter(eye[indices, 0], eye[indices, 2], color="#167f8c", label="six observed camera positions")
    for index in indices:
        ax.text(eye[index, 0], eye[index, 2], f" {float(model.times[index]):.2f}s", fontsize=8)
        ax.plot([eye[index, 0], 0], [eye[index, 2], 0], color="#167f8c", alpha=.2)
    for offset in (100, 200):
        extra, _ = orbit(model.times, offset_degrees=offset)
        ax.plot(extra[:, 0], extra[:, 2], "--", alpha=.5, label=f"evaluation path +{offset}°")
    ax.scatter([0], [0], marker="s", s=80, color="#343d46", label="object")
    ax.set_aspect("equal"); ax.set_xlabel("world x [m]"); ax.set_ylabel("world z [m]")
    ax.set_title("Camera coverage from above · one elevation")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=.2)
    fig.savefig(output/"mesh_camera_coverage.png", dpi=150); plt.close(fig)
    return {"native_image_size": 24, "display": "nearest-neighbor enlargement; display RGB clipped only for viewing, inference uses unclipped floating RGB",
            "sequence": "pulse_and_stochastic__4101", "camera_orbit_degrees": 100,
            "horizontal_fov_degrees": math.degrees(2*math.atan(.5/1.65)),
            "observed_times_s": model.times[benchmark.observations.indices].tolist(),
            "nominal_frame_duration_s": .05,
            "source_inputs_verified": True}


def truth_parameters(dtype=torch.float64):
    return (torch.tensor([19.5], dtype=dtype), torch.tensor([.55], dtype=dtype),
            torch.tensor([[.11, -.035, .06]], dtype=dtype), torch.tensor([[-.12, .04, .10]], dtype=dtype))


def extra_acceleration(t, x, mesh, pulse):
    if not pulse:
        return torch.zeros_like(x)
    strength = math.exp(-.5*((t-.5)/.08)**2)
    return strength*(~mesh.pinned).to(x.dtype)[None, :, None]*x.new_tensor([.45, 0., -.30])


@torch.no_grad()
def rk4_reference(mesh, times, pulse, subdivisions):
    """Independent fourth-order first-order-system integration; no simulate()."""
    young, drag, offset, velocity = truth_parameters()
    height = (mesh.rest[:, 1]-mesh.rest[:, 1].min())/.65
    x = mesh.rest[None]+height[None, :, None]*offset[:, None]
    v = height[None, :, None]*velocity[:, None]
    xs, vs = [x.clone()], [v.clone()]
    def derivative(t, nodes, speeds):
        force, _, _ = material_response(nodes, mesh, young)
        acceleration = force/mesh.mass[None, :, None]+nodes.new_tensor([0., -.6, 0.])-drag[:, None, None]*speeds
        acceleration += extra_acceleration(t, nodes, mesh, pulse)
        return (torch.where(mesh.pinned[None, :, None], 0., speeds),
                torch.where(mesh.pinned[None, :, None], 0., acceleration))
    for interval in range(len(times)-1):
        dt = float(times[interval+1]-times[interval])/subdivisions
        for step in range(subdivisions):
            t = float(times[interval])+step*dt
            k1x, k1v = derivative(t, x, v)
            k2x, k2v = derivative(t+dt/2, x+dt*k1x/2, v+dt*k1v/2)
            k3x, k3v = derivative(t+dt/2, x+dt*k2x/2, v+dt*k2v/2)
            k4x, k4v = derivative(t+dt, x+dt*k3x, v+dt*k3v)
            x = x+dt/6*(k1x+2*k2x+2*k3x+k4x)
            v = v+dt/6*(k1v+2*k2v+2*k3v+k4v)
        xs.append(x.clone()); vs.append(v.clone())
    return torch.stack(xs, 1), torch.stack(vs, 1)


@torch.no_grad()
def logged_simulation(condition, subdivisions=3, white=None):
    pulse, stochastic = CONDITIONS[condition]
    _, _, model, benchmark = load_case(condition)
    mesh, times = model.mesh, model.times
    young, drag, offset, velocity = truth_parameters()
    if stochastic and white is None:
        white = torch.randn(1, (len(times)-1)*subdivisions, 3, dtype=torch.float64,
                            generator=torch.Generator().manual_seed(4101))
    if not stochastic:
        white = None
    states_x, states_v, internal, forcing, impulses, dts = [], [], [], [], [], []
    step = 0
    def external(t, x, v):
        nonlocal step
        interval = step//subdivisions
        dt = float(times[interval+1]-times[interval])/subdivisions
        force, _, _ = material_response(x, mesh, young)
        acceleration = extra_acceleration(t, x, mesh, pulse)
        impulse = (edge_impulses(x, v, mesh, white[:, step]*math.sqrt(dt), calibration_law, model.basis)
                   if stochastic else torch.zeros_like(x))
        states_x.append(x.clone()); states_v.append(v.clone()); internal.append(force)
        forcing.append(acceleration*mesh.mass[None, :, None]); impulses.append(impulse); dts.append(dt)
        step += 1
        return acceleration
    trajectory = simulate(mesh, times, young, drag, offset, velocity, subdivisions,
        white=white, diffusion=calibration_law if stochastic else None, noise_basis=model.basis,
        external_acceleration=external)
    states_x.append(trajectory.x[:, -1]); states_v.append(trajectory.v[:, -1])
    x, v = torch.stack(states_x, 1), torch.stack(states_v, 1)
    force, external_force, impulse = torch.stack(internal, 1), torch.stack(forcing, 1), torch.stack(impulses, 1)
    dt = times.new_tensor(dts)
    actual = mesh.mass[None, None, :, None]*(v[:, 1:]-v[:, :-1])
    known = mesh.mass[None, None, :, None]*(x.new_tensor([0., -.6, 0.])-drag[:, None, None, None]*v[:, :-1])
    applied = dt[None, :, None, None]*(force+known+external_force)+impulse
    reaction = actual-applied
    angular_change = torch.linalg.cross(x[:, 1:], mesh.mass[None, None, :, None]*v[:, 1:]).sum(2)
    angular_change -= torch.linalg.cross(x[:, :-1], mesh.mass[None, None, :, None]*v[:, :-1]).sum(2)
    angular_applied = torch.linalg.cross(x[:, :-1], applied+reaction).sum(2)
    _, elastic, j = material_response(x, mesh, young[:, None])
    energy = .5*(mesh.mass[None, None, :, None]*v.square()).sum((2, 3))+elastic+.6*(mesh.mass[None, None]*x[..., 1]).sum(2)
    # Trapezoidal continuous external work/dissipation: defect should decrease
    # under deterministic time refinement, not be zero for semi-implicit Euler.
    drag_power = -.55*(mesh.mass[None, None, :, None]*v.square()).sum((2, 3))
    force_end = torch.stack([extra_acceleration(float(t), x[:, k], mesh, pulse)*mesh.mass[None, :, None]
                             for k, t in enumerate(torch.cat((dt.new_zeros(1), dt.cumsum(0))))], 1)
    work_power = drag_power+(force_end*v).sum((2, 3))
    work = ((work_power[:, :-1]+work_power[:, 1:])*.5*dt[None]).sum()
    record = {"condition": condition, "substeps": subdivisions, "dt_max_s": float(dt.max()),
        "minimum_det_F_all_steps": float(j.min()),
        "max_free_node_momentum_residual_Ns": float(reaction[:, :, ~mesh.pinned].abs().max()),
        "max_internal_force_net_N": float(force.sum(2).abs().max()),
        "max_internal_impulse_net_Ns": float(impulse.sum(2).abs().max()),
        "max_internal_impulse_torque_Nms": float(torch.linalg.cross(x[:, :-1], impulse).sum(2).abs().max()),
        "max_total_angular_balance_residual_kg_m2_per_s": float((angular_change-angular_applied).abs().max()),
        "max_support_position_error_m": float((x[:, :, mesh.pinned]-mesh.rest[mesh.pinned]).abs().max()),
        "max_support_velocity_m_per_s": float(v[:, :, mesh.pinned].abs().max()),
        "max_support_reaction_impulse_Ns": float(reaction[:, :, mesh.pinned].abs().max()),
        "initial_energy_J": float(energy[0, 0]), "final_energy_J": float(energy[0, -1]),
        "deterministic_work_energy_defect_J": float(energy[0, -1]-energy[0, 0]-work) if not stochastic else None}
    if subdivisions == 3 and (white is None or torch.equal(white, torch.randn(1, 60, 3, dtype=torch.float64,
                                                  generator=torch.Generator().manual_seed(4101)))):
        torch.testing.assert_close(trajectory.x, benchmark.truth.x, rtol=1e-12, atol=1e-14)
    return trajectory, record


def physics_audit(output):
    began = time.perf_counter()
    mesh = cube_mesh()
    gen = torch.Generator().manual_seed(57017)
    young = mesh.rest.new_tensor([19.5])
    force_errors, energy_errors = [], []
    # Independent scalar strain energy, using direct determinant and autograd.
    for _ in range(12):
        x = (mesh.rest[None]+.025*torch.randn(1, 8, 3, dtype=torch.float64, generator=gen)).requires_grad_()
        ds = (x[:, mesh.cells[:, 1:]]-x[:, mesh.cells[:, :1]]).transpose(-1, -2)
        f = ds@mesh.inv_rest
        mu, lam = 19.5/(2*1.3), 19.5*.3/(1.3*.4)
        logj = torch.linalg.det(f).log()
        independent = (mesh.volumes*(mu/2*(f.square().sum((-2, -1))-3)-mu*logj+lam/2*logj.square())).sum()
        derivative, = torch.autograd.grad(independent, x)
        analytic_force, energy, _ = material_response(x.detach(), mesh, young)
        force_errors.append(float((analytic_force+derivative).abs().max()))
        energy_errors.append(abs(float(energy)-float(independent.detach())))
    # Linearized rest-state modes quantify the integrator step relative to a
    # small-amplitude stability limit; this is not a nonlinear stability proof.
    free = ~mesh.pinned
    def rest_energy(q):
        x = mesh.rest.clone(); x[free] = q.reshape(-1, 3)
        return material_response(x[None], mesh, young)[1].sum()
    stiffness = torch.autograd.functional.hessian(rest_energy, mesh.rest[free].flatten())
    invroot = mesh.mass[free].repeat_interleave(3).rsqrt()
    eigenvalues = torch.linalg.eigvalsh(invroot[:, None]*stiffness*invroot[None])
    omega = float(eigenvalues.max().sqrt())
    dt_limit = (-.55+math.sqrt(.55**2+4*omega**2))/omega**2
    records = []
    for condition in CONDITIONS:
        _, record = logged_simulation(condition)
        records.append(record)
    refinement = []
    for condition in ("no_load", "external_pulse"):
        _, _, model, _ = load_case(condition)
        reference48, _ = rk4_reference(mesh, model.times, CONDITIONS[condition][0], 48)
        reference96, velocity96 = rk4_reference(mesh, model.times, CONDITIONS[condition][0], 96)
        rk_error = float((reference48[:, :, free]-reference96[:, :, free]).square().mean().sqrt())
        # A corrected high-accuracy deterministic reference is published for
        # future inference tests. It is NOT used in the view-count ablation.
        with torch.no_grad():
            benchmark = load_case(condition)[3]
            clean = model.renderer(reference96, model.eye, model.rotation)[0]
            indices = benchmark.observations.indices
            noise = benchmark.observations.rgb-benchmark.truth_rgb[indices]
        reference_folder = output/"high_accuracy_reference"
        reference_folder.mkdir(exist_ok=True)
        write_json(reference_folder/f"{condition}.json", {
            "condition": condition, "integrator": "independent classical RK4", "substeps_per_frame": 96,
            "RK4_48_vs_96_coordinate_rmse_m": rk_error, "times_s": model.times.tolist(),
            "positions_m": reference96[0].tolist(), "velocities_m_per_s": velocity96[0].tolist(),
            "observed_indices": indices.tolist(), "observed_rgb": (clean[indices]+noise).tolist(),
            "observation_sigma": benchmark.observations.sigma,
            "observation_noise": "same six-image noise realization as historical benchmark; underlying clean images regenerated with RK4",
            "camera_eye_m": model.eye.tolist(), "camera_rotation_world_to_camera": model.rotation.tolist(),
            "material": {"young_pa": 19.5, "poisson_ratio": .3, "density_kg_per_m3": 1.},
            "drag_per_second": .55, "gravity_m_per_s2": [0., -.6, 0.],
            "initial_offset_m": [.11, -.035, .06], "initial_velocity_m_per_second": [-.12, .04, .10],
            "rest_nodes_m": mesh.rest.tolist(), "tetrahedra": mesh.cells.tolist(),
            "nodal_mass_kg": mesh.mass.tolist(), "pinned_nodes": mesh.pinned.tolist(),
            "qualification": "deterministic time-converged reference for the SAME six-tetrahedron material model; not mesh-converged or experimentally validated"})
        for subdivisions in (3, 6, 12, 24, 48):
            trajectory, record = logged_simulation(condition, subdivisions)
            error = trajectory.x[:, :, free]-reference96[:, :, free]
            refinement.append({"condition": condition, "substeps": subdivisions,
                "position_rmse_m_vs_RK4_96": float(error.square().mean().sqrt()),
                "max_node_error_m_vs_RK4_96": float(torch.linalg.vector_norm(error, dim=-1).max()),
                "RK4_48_vs_96_position_rmse_m": rk_error,
                "work_energy_defect_J": record["deterministic_work_energy_defect_J"]})
            print("Refinement", condition, subdivisions, refinement[-1]["position_rmse_m_vs_RK4_96"], flush=True)
    # Coupled Brownian refinements share exactly the original coarse increments.
    stochastic = []
    coarse = torch.randn(1, 60, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(4101))
    u = torch.randn(1, 60, 16, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(514101))
    bridge = u-u.mean(2, keepdim=True)+coarse[:, :, None]/4
    for condition in ("stochastic", "pulse_and_stochastic"):
        paths = {}
        for factor in (1, 2, 4, 8, 16):
            white = bridge.reshape(1, 60*factor, 16//factor, 3).sum(2)/math.sqrt(16//factor)
            trajectory, _ = logged_simulation(condition, 3*factor, white)
            paths[factor] = trajectory.x
        torch.testing.assert_close(paths[1], load_case(condition)[3].truth.x, rtol=1e-12, atol=1e-14)
        for factor in (1, 2, 4, 8):
            error = paths[factor][:, :, free]-paths[16][:, :, free]
            stochastic.append({"condition": condition, "substeps": 3*factor,
                "coupled_position_rmse_m_vs_48": float(error.square().mean().sqrt()),
                "qualification": "one coupled realization, finest Euler path is reference, not exact stochastic solution"})
    # The simulation's stochastic amplitudes must scale with sqrt(dt), not dt.
    x = mesh.rest[None].clone(); x[:, free, 0] += .05
    basis = spatial_basis(mesh, 3)
    unit = torch.eye(3, dtype=torch.float64)
    h = edge_impulses(x.expand(3, -1, -1), torch.zeros(3, 8, 3, dtype=torch.float64), mesh, unit, calibration_law, basis).flatten(1).T
    innovations = torch.randn(8192, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(57117))
    variance = []
    for dt in (.0025, .005, .01, .02):
        values = edge_impulses(x.expand(len(innovations), -1, -1), torch.zeros(len(innovations), 8, 3, dtype=torch.float64),
                              mesh, math.sqrt(dt)*innovations, calibration_law, basis).flatten(1)
        empirical = torch.cov(values.T)
        expected = dt*h@h.T
        variance.append({"dt_s": dt, "relative_covariance_Frobenius_error": float(torch.linalg.norm(empirical-expected)/torch.linalg.norm(expected)),
                         "empirical_trace_Ns2": float(empirical.trace()), "analytic_trace_Ns2": float(expected.trace())})
    result = {"force_negative_independent_energy_gradient_max_error_N": max(force_errors),
        "independent_energy_max_error_J": max(energy_errors), "rest_linearized_omega_max_rad_per_s": omega,
        "rest_linearized_damped_semiimplicit_dt_limit_s": dt_limit, "benchmark_dt_s": 1/60,
        "benchmark_dt_over_linearized_limit": (1/60)/dt_limit,
        "physical_constants": {"young_Pa": 19.5, "density_kg_per_m3": 1., "gravity_m_per_s2": .6,
            "poisson_ratio": .3, "drag_per_s": .55, "side_m": .65, "total_mass_kg": float(mesh.mass.sum())},
        "balances": records, "deterministic_refinement": refinement, "coupled_stochastic_refinement": stochastic,
        "noise_covariance_checks": variance, "seconds": time.perf_counter()-began,
        "scope": "numerical implementation audit of a toy prescribed material/geometry, not experimental validation or mesh convergence"}
    write_json(output/"physics_audit.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for condition in ("no_load", "external_pulse"):
        rows = [r for r in refinement if r["condition"] == condition]
        axes[0].loglog([.05/r["substeps"] for r in rows], [1000*r["position_rmse_m_vs_RK4_96"] for r in rows], "o-", label=condition)
        axes[1].loglog([.05/r["substeps"] for r in rows], [abs(r["work_energy_defect_J"]) for r in rows], "o-", label=condition)
    axes[0].set_ylabel("position RMSE vs RK4 reference [mm]")
    axes[1].set_ylabel("absolute work–energy defect [J]")
    for ax in axes:
        ax.set_xlabel("simulation time step [s]"); ax.grid(alpha=.25); ax.legend()
    fig.savefig(output/"time_refinement.png", dpi=150); plt.close(fig)
    return result


class ViewAuditModel(TrajectoryForceModel):
    """Same trajectory density; observations can share times across cameras.

    Camera poses are known. Per-frame position information is accumulated in
    small blocks; this is algebraically equivalent to the full RGB Jacobian.
    """
    def set_views(self, specs):
        self.view_specs = list(specs)
        self.view_frames = torch.tensor([f for f, _ in specs], dtype=torch.long)
        cameras = {offset: orbit(self.times, offset_degrees=offset) for _, offset in specs}
        self.view_eye = torch.stack([cameras[offset][0][frame] for frame, offset in specs])
        self.view_rotation = torch.stack([cameras[offset][1][frame] for frame, offset in specs])
        return self

    def predict(self, state, indices=None):
        if indices is None:
            return super().predict(state)
        if not torch.equal(indices, self.view_frames):
            raise ValueError("View audit observation indices do not match declared camera slots.")
        trajectory = self.trajectory(state)
        return self.renderer(trajectory.x[:, indices], self.view_eye, self.view_rotation)

    def pose_information(self, mean, observations):
        with torch.no_grad():
            self.residual(mean[None])
            frames = self.trajectory(mean[None]).x
        blocks = {}
        h = mean.new_zeros(self.state_dimension, self.state_dimension)
        for slot, (frame, _) in enumerate(self.view_specs):
            x = frames[:, [frame]].clone()
            def render_free(q):
                nodes = x.clone(); nodes[0, 0, self.free] = q.reshape(-1, 3)
                return self.renderer(nodes, self.view_eye[[slot]], self.view_rotation[[slot]]).flatten()
            j = torch.autograd.functional.jacobian(render_free, x[0, 0, self.free].flatten(),
                                                   vectorize=True, strategy="forward-mode")/observations.sigma
            pose_h = j.T@j
            blocks[frame] = blocks.get(frame, torch.zeros_like(pose_h))+pose_h
            mapping = self.position_map[frame]
            h += mapping.T@pose_h@mapping
        ranks, minima, maxima = [], [], []
        for frame, pose_h in blocks.items():
            if frame == 0:
                # The known initial shape has only three displacement freedoms.
                initial_map = self.position_map[0, :, 2:5]/self.prior.std[None, 2:5]
                pose_h = initial_map.T@pose_h@initial_map
            values = torch.linalg.eigvalsh(pose_h).clamp_min(0)
            threshold = float(values.max())*1e-12
            ranks.append(int((values > threshold).sum()))
            minima.append(float(values[0].sqrt())); maxima.append(float(values[-1].sqrt()))
        return h.detach(), {"image_only_local_rank": sum(ranks),
            "maximum_rank_from_observed_pose_coordinates": 3+(len(blocks)-1)*self.free_dof,
            "weakest_pose_singular_value_per_m": min(minima),
            "strongest_pose_singular_value_per_m": max(maxima),
            "qualification": "current depth-order branch at fixed scene; known appearance/geometry; rank is not global identifiability"}

    def information(self, mean, observations):
        image_h, _ = self.pose_information(mean, observations)
        dynamics = torch.autograd.functional.jacobian(lambda q: self.residual(q[None])[0], mean,
                                                     vectorize=True, strategy="forward-mode").detach()
        prior = mean.new_zeros(self.state_dimension, self.state_dimension)
        prior[:8, :8] = torch.eye(8, dtype=mean.dtype)
        white = dynamics/self.elimination.sd[:, None]
        cross = -white.T@self.elimination.white_mapping
        explicit = prior+image_h+white.T@white
        marginal_j = self.elimination.whiten(dynamics.T).T
        marginal = prior+image_h+marginal_j.T@marginal_j
        return marginal, explicit, cross


VIEW_SCHEMES = {
    "six_single": (list((0, 3, 6, 14, 17, 20)), (0,)),
    "twelve_single": (list((0, 2, 3, 5, 6, 8, 10, 12, 14, 17, 18, 20)), (0,)),
    "six_dual": (list((0, 3, 6, 14, 17, 20)), (0, 100)),
    "twentyone_single": (list(range(21)), (0,)),
    "twentyone_dual": (list(range(21)), (0, 100))}


def build_view_data(condition):
    inputs, _, base, benchmark = load_case(condition)
    renderings, observed = {}, {}
    with torch.no_grad():
        for offset in (0, 100):
            true = base.renderer(benchmark.truth.x, *orbit(base.times, offset_degrees=offset))[0]
            renderings[offset] = true
            seed = 304101 if offset == 0 else 404101
            noise = torch.randn(true.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
            observed[offset] = true+base.config.observation_sigma*noise
        # Every scheme retains EXACTLY the original six noisy primary images.
        observed[0][benchmark.observations.indices] = torch.tensor(inputs["rgb"], dtype=torch.float64)
    return inputs, base, benchmark, renderings, observed


def fixed_state_information(model, observations, state):
    information, _, _ = model.information(state, observations)
    chol = torch.linalg.cholesky(information)
    _, rank = model.pose_information(state, observations)
    selection = torch.eye(model.state_dimension, dtype=state.dtype)[:, :8]
    scene_covariance = torch.cholesky_solve(selection, chol)[:8]
    dynamics = torch.autograd.functional.jacobian(lambda q: model.residual(q[None])[0], state,
                                                 vectorize=True, strategy="forward-mode").detach()
    dmu = torch.cholesky_solve(model.elimination.white_mapping.T@(dynamics/model.elimination.sd[:, None]),
                              model.elimination.cholesky)
    force_covariance = model.elimination.conditional_covariance+dmu@torch.cholesky_solve(dmu.T, chol)
    return {**rank, "scene_standardized_covariance": scene_covariance.tolist(),
        "log_young_std": float(scene_covariance[0, 0].sqrt()*model.prior.std[0]),
        "log_drag_std": float(scene_covariance[1, 1].sqrt()*model.prior.std[1]),
        "force_coefficient_covariance_trace": float(force_covariance.trace()),
        "qualification": "same reference state for all schemes; local linear Gaussian information, not fitted/calibrated posterior"}


def view_audit(output):
    began = time.perf_counter()
    script_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records, data_records = [], []
    for condition in ("external_pulse", "pulse_and_stochastic"):
        inputs, base, benchmark, truth_rgb, observed = build_view_data(condition)
        point = load_case(condition)[1]
        reference_state = torch.tensor(point["state_coordinates"], dtype=torch.float64)
        initial = torch.tensor(inputs["initial_state_coordinates"], dtype=torch.float64)
        sample_data = {"condition": condition, "config": asdict(base.config),
            "primary_rgb_all_times": observed[0].tolist(), "secondary_rgb_all_times": observed[100].tolist(),
            "camera_offsets_degrees": [0, 100], "original_six_indices": list(base.config.observed_frames),
            "additional_primary_noise_seed": 304101, "secondary_noise_seed": 404101,
            "original_input_hash": inputs["observations_sha256_float64_tensor_bytes"],
            "additional_noise_note": "new independent noise; original primary six overwritten with exact published inputs"}
        write_json(output/f"view_inputs_{condition}.json", sample_data)
        data_records.append({k: sample_data[k] for k in ("condition", "original_input_hash", "additional_primary_noise_seed", "secondary_noise_seed")})
        local_reference = {}
        for scheme, (frames, offsets) in VIEW_SCHEMES.items():
            if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_digest:
                raise RuntimeError("Audit script changed during the view experiment.")
            print("View ablation", condition, scheme, flush=True)
            specs = [(frame, offset) for offset in offsets for frame in frames]
            model = ViewAuditModel(base.config, base.force_prior.config, base.marginal_config).set_views(specs)
            images = torch.stack([observed[offset][frame] for frame, offset in specs])
            observations = Observations(model.view_frames, images, base.config.observation_sigma)
            information = fixed_state_information(model, observations, reference_state)
            local_reference[scheme] = information
            destination = output/"fits"/f"{condition}__{scheme}"
            destination.mkdir(parents=True, exist_ok=True)
            fitted = fit_trajectory(model, observations, initial, "marginalized", TrajectoryFitConfig(iterations=20),
                                    progress=lambda message: print(message, flush=True))
            torch.save(fitted, destination/"fit.pt")
            with torch.no_grad():
                trajectory = model.trajectory(fitted["mean"][None])
                error = trajectory.x[0, :, model.free]-benchmark.truth.x[0, :, model.free]
                gap = torch.zeros(base.config.frames, dtype=torch.bool); gap[7:14] = True
                unobserved = torch.ones(base.config.frames, dtype=torch.bool); unobserved[frames] = False
                novel_camera = orbit(base.times, offset_degrees=200)
                novel_truth = base.renderer(benchmark.truth.x, *novel_camera)[0]
                novel_fit = base.renderer(trajectory.x, *novel_camera)[0]
                novel_mse = (novel_fit-novel_truth).square().mean()
                a = fitted["conditional_force_mean"]
                total = base.force_prior.total_force(a[None], base.step_times)[0]
                true_total = torch.zeros_like(total)
                if CONDITIONS[condition][0]:
                    strength = torch.exp(-.5*((base.step_times-.5)/.08).square())
                    true_total = strength[:, None]*base.mesh.mass[base.free].sum()*total.new_tensor([.45, 0., -.30])
                young, drag, _, _ = model.prior.decode(fitted["mean"][None])
            record = {"condition": condition, "scheme": scheme, "observed_timestamps": frames,
                "camera_offsets_degrees": list(offsets), "input_image_count": len(specs),
                "maximum_observation_gap_s": max(np.diff(frames))*.05,
                "observations_sha256": hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                "shared_initial_state_sha256": hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
                "fixed_reference_information": information, "fit": fitted["record"],
                "all_frame_position_rmse_m": float(error.square().mean().sqrt()),
                "original_gap_position_rmse_m": float(error[gap].square().mean().sqrt()),
                "still_unobserved_position_rmse_m": float(error[unobserved].square().mean().sqrt()) if bool(unobserved.any()) else None,
                "unseen_200_degree_view_psnr_db": float(-10*novel_mse.clamp_min(1e-15).log10()),
                "conditional_MAP_force_rmse_N": float((total-true_total).square().mean().sqrt()),
                "young_pa": float(young[0]), "drag_per_second": float(drag[0]),
                "point_state_coordinates": fitted["mean"].tolist(),
                "qualification": "same exact truth and shared images; extra observations allowed by scheme; original-gap error is NOT held-out when those times become observed; no nonlinear uncertainty calibration claim"}
            write_json(destination/"summary.json", record)
            records.append(record)
            print(f"  original-gap pose RMSE={record['original_gap_position_rmse_m']*1000:.3f} mm; "
                  f"stationarity={fitted['record']['stationarity_screen_pass']}", flush=True)
        # Extra observations must not increase local linear posterior covariance
        # when evaluated at the same state with the same model and noise.
        reference_diagonal = np.diag(np.asarray(local_reference["six_single"]["scene_standardized_covariance"]))
        for scheme in local_reference:
            diagonal = np.diag(np.asarray(local_reference[scheme]["scene_standardized_covariance"]))
            if np.max(diagonal-reference_diagonal) > 1e-8:
                raise RuntimeError("Adding matched data increased fixed-state scene covariance; audit the information calculation.")
    result = {"cases": records, "inputs": data_records, "seconds": time.perf_counter()-began,
        "protocol": {"schemes": {k: {"frames": v[0], "camera_offsets_degrees": list(v[1])} for k, v in VIEW_SCHEMES.items()},
            "equal_image_budget_pair": ["twelve_single", "six_dual"], "optimizer": asdict(TrajectoryFitConfig(iterations=20)),
            "shared_initialization": "original six-image observation-only forward fit; no added/withheld truth used in initialization",
            "force_prior": asdict(ForceConfig()), "trajectory_model": asdict(MarginalConfig())},
        "scope": "two synthetic sequences of one known geometry; only observation number/timing/camera varies; local information and finite optimization, not a generalization benchmark"}
    write_json(output/"view_audit.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = list(VIEW_SCHEMES)
    x = np.arange(len(labels))
    for condition, shift, color in (("external_pulse", -.18, "#167f8c"), ("pulse_and_stochastic", .18, "#aa6230")):
        rows = [r for r in records if r["condition"] == condition]
        axes[0].bar(x+shift, [1000*r["original_gap_position_rmse_m"] for r in rows], width=.34, color=color, label=condition)
        axes[1].plot(x, [r["fixed_reference_information"]["force_coefficient_covariance_trace"] for r in rows], "o-", color=color, label=condition)
    axes[0].set_ylabel("original-gap MAP position RMSE [mm]")
    axes[0].set_title("Finite nonlinear fits · see stationarity flags")
    axes[1].set_ylabel("force coefficient covariance trace")
    axes[1].set_title("Local information at the SAME reference state")
    for ax in axes:
        ax.set_xticks(x, ["6 frames\n1 camera", "12 frames\n1 camera", "6 frames\n2 cameras", "21 frames\n1 camera", "21 frames\n2 cameras"])
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
    fig.savefig(output/"view_ablation.png", dpi=150); plt.close(fig)
    return result


@torch.no_grad()
def camera_audit(output):
    _, _, model, benchmark = load_case("pulse_and_stochastic")
    identity = torch.eye(3, dtype=torch.float64)
    orthogonality = float((model.rotation@model.rotation.transpose(-1, -2)-identity).abs().max())
    determinant_error = float((torch.linalg.det(model.rotation)-1).abs().max())
    target = model.times.new_tensor([0., .51, 0.])[None, None, None].expand(1, len(model.times), 1, 3)
    projection, _, _ = project(target, model.eye, model.rotation, model.config.image_size)
    principal_error = float((projection-(model.config.image_size-1)/2).abs().max())
    centers = torch.einsum("pk,bftkd->bftpd", model.renderer.bary,
                           benchmark.truth.x[:, :, model.mesh.cells]).flatten(2, 3)
    means, depth, jac = project(centers, model.eye, model.rotation, model.config.image_size)
    f = deformation(benchmark.truth.x, model.mesh)
    covariance = f@model.renderer.sigma0@f.transpose(-1, -2)
    covariance = covariance.unsqueeze(-3).expand(1, 21, 6, 5, 3, 3).flatten(2, 3)
    camera_covariance = model.rotation[None, :, None]@covariance@model.rotation[None, :, None].transpose(-1, -2)
    projected = jac@camera_covariance@jac.transpose(-1, -2)+.3*torch.eye(2, dtype=torch.float64)
    radius = 3*projected.diagonal(dim1=-2, dim2=-1).sqrt()
    outside_centers = ((means < -.5)|(means > 23.5)).any(-1)
    clipped_three_sigma = ((means-radius < -.5)|(means+radius > 23.5)).any(-1)
    foreground = ((benchmark.truth_rgb-.035).square().sum(-1).sqrt() > .1)
    record = {"rotation_orthogonality_max_error": orthogonality, "rotation_determinant_max_error": determinant_error,
        "look_at_target_principal_point_max_error_pixels": principal_error,
        "minimum_gaussian_depth_m": float(depth.min()), "centers_outside_image_fraction": float(outside_centers.double().mean()),
        "three_sigma_gaussian_contours_cross_image_boundary_fraction": float(clipped_three_sigma.double().mean()),
        "foreground_fraction_at_RGB_norm_threshold_0_1": float(foreground.double().mean()),
        "camera_azimuth_range_degrees": [-25., 75.], "horizontal_field_of_view_degrees": math.degrees(2*math.atan(.5/1.65)),
        "one_camera_per_time": True, "rest_geometry_and_appearance_known": True,
        "qualification": "center/contour bounds and local projections do not establish surface visibility or geometric identifiability"}
    write_json(output/"camera_audit.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--images-only", action="store_true")
    modes.add_argument("--physics-only", action="store_true")
    modes.add_argument("--views-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    from physgauss.cli import source_fingerprint
    from physgauss.likelihood_audit import _environment
    write_json(args.output/"manifest.json", {"source_fingerprint_sha256": source_fingerprint(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": _environment(), "mode": "images" if args.images_only else "physics" if args.physics_only else "views" if args.views_only else "full",
        "historical_summary_sha256_normalized_lf": hashlib.sha256((ROOT/"examples/marginal-force-study/summary.json").read_bytes().replace(b"\r\n", b"\n")).hexdigest()})
    if not args.physics_only and not args.views_only:
        visual = galleries(args.output)
        write_json(args.output/"visual_setup.json", visual)
        print(json.dumps(visual, indent=2), flush=True)
        camera_audit(args.output)
    if not args.images_only and not args.views_only:
        physics_audit(args.output)
    if not args.images_only and not args.physics_only:
        view_audit(args.output)


if __name__ == "__main__":
    main()
