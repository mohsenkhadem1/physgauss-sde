"""Reproducible independent-reference comparison of integrators and visibility.

Inference receives RGB, cameras and fixed priors only. RK4 states are used to
generate observations and score finished fits, never to initialize or fit them.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from .accurate_pipeline import AccurateViewModel, LegacyViewModel
from .benchmark import Config, Observations
from .camera import orbit
from .continuous_render import ContinuousGaussianRenderer, VolumeRenderConfig
from .forces import ForceConfig
from .likelihood_audit import _environment
from .marginal_forces import MarginalConfig
from .marginal_inference import (TrajectoryFitConfig, fit_trajectory,
    trajectory_gaussian_draws, recover_force_moments)
from .marginal_study import audit_gaussian
from .physics import InvalidDeformation
from .render import GaussianRenderer
from .study import normalized, write_json


SCHEMES = {
    "six_single": ([0, 3, 6, 14, 17, 20], [0]),
    "twelve_single": ([0, 2, 3, 5, 6, 8, 10, 12, 14, 17, 18, 20], [0]),
    "six_dual": ([0, 3, 6, 14, 17, 20], [0, 100]),
    "twentyone_single": (list(range(21)), [0]),
    "twentyone_dual": (list(range(21)), [0, 100]),
}
VARIANTS = {
    "corrected": (True, True),
    "legacy": (False, False),
    "integrator_only": (True, False),
    "renderer_only": (False, True),
}


def default_plan(quick=False):
    return {
        "conditions": ["external_pulse"] if quick else ["no_load", "external_pulse"],
        "methods": ["marginalized"] if quick else ["marginalized", "joint", "alternating"],
        "variants": ["corrected"] if quick else list(VARIANTS),
        "schemes": ["six_single"] if quick else list(SCHEMES),
        "substeps": 3 if quick else 6, "legacy_substeps": 3,
        "render_bins": 32 if quick else 128, "reference_bins": 128 if quick else 512,
        "optimizer": asdict(TrajectoryFitConfig(iterations=2 if quick else 20)),
        "force_prior": asdict(ForceConfig()), "trajectory_model": asdict(MarginalConfig()),
        "evaluation_samples": 8 if quick else 128, "audit_samples": 8 if quick else 128,
        "sampling_seed": 106600, "threads": 2, "smoke_only": quick,
    }


def digest_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def validate_plan(plan):
    for key, allowed in (("conditions", ("no_load", "external_pulse")),
                         ("methods", ("marginalized", "joint", "alternating")),
                         ("variants", VARIANTS), ("schemes", SCHEMES)):
        values = plan[key]
        if not values or len(values) != len(set(values)) or not set(values) <= set(allowed):
            raise ValueError("Invalid/repeated "+key)
    if "corrected" not in plan["variants"] or "six_single" not in plan["schemes"]:
        raise ValueError("Need the primary corrected six-view comparison.")
    if min(plan[k] for k in ("substeps", "legacy_substeps", "threads")) < 1:
        raise ValueError("Positive integrator and thread settings required.")
    if min(plan[k] for k in ("evaluation_samples", "audit_samples")) < 4:
        raise ValueError("Need at least four independent draws.")
    if plan["substeps"] % plan["legacy_substeps"]:
        raise ValueError("Shared initialization needs nested solver grids.")
    TrajectoryFitConfig(**plan["optimizer"]).validate()
    ForceConfig(**plan["force_prior"]).validate()
    MarginalConfig(**plan["trajectory_model"]).validate()
    for bins in (plan["render_bins"], plan["reference_bins"]):
        VolumeRenderConfig(bins=bins).validate()


def construct(plan, variant, scheme):
    staggered, continuous = VARIANTS[variant]
    config = Config(substeps=plan["substeps"] if staggered else plan["legacy_substeps"],
                    truth_stochastic=False, threads=plan["threads"])
    cls = AccurateViewModel if staggered else LegacyViewModel
    model = cls(config, ForceConfig(**plan["force_prior"]), MarginalConfig(**plan["trajectory_model"]))
    if continuous:
        model.renderer = ContinuousGaussianRenderer(model.mesh, config.image_size,
                                                   VolumeRenderConfig(bins=plan["render_bins"]))
    frames, offsets = SCHEMES[scheme]
    return model.set_views([(frame, offset) for offset in offsets for frame in frames])


@torch.no_grad()
def inputs(plan, reference_file, model):
    """Re-render independent RK4 motion; preserve every original noise pixel."""
    raw = Path(reference_file).read_bytes()
    reference = json.loads(raw)
    x = model.times.new_tensor(reference["positions_m"])[None]
    old_indices = torch.tensor(SCHEMES["six_single"][0])
    old_rgb = model.times.new_tensor(reference["observed_rgb"])
    hard = GaussianRenderer(model.mesh, model.config.image_size)
    original_noise = old_rgb-hard(x[:, old_indices], model.eye[old_indices], model.rotation[old_indices])[0]
    renderer = ContinuousGaussianRenderer(model.mesh, model.config.image_size,
                                         VolumeRenderConfig(bins=plan["reference_bins"]))
    clean, noisy = {}, {}
    for offset, seed in ((0, 304101), (100, 404101), (200, 504101)):
        eye, rotation = orbit(model.times, offset_degrees=offset)
        # Chunking keeps reference integration memory bounded at 1024 bins.
        clean[offset] = torch.cat([renderer(x[:, a:a+3], eye[a:a+3], rotation[a:a+3])[0]
                                   for a in range(0, model.config.frames, 3)])
        noise = model.config.observation_sigma*torch.randn(clean[offset].shape, dtype=x.dtype,
                                                         generator=torch.Generator().manual_seed(seed))
        if offset == 0:
            noise[old_indices] = original_noise
        noisy[offset] = clean[offset]+noise
    return {"reference": reference, "x": x[0], "clean": clean, "noisy": noisy,
            "reference_sha256_normalized_lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
            "original_noise_sha256": digest_tensor(original_noise)}


@torch.no_grad()
def initial_state(plan):
    # Prior mean, zero GP load, deterministic corrected rollout; no truth, no
    # withheld image, and no image optimization used to select this start.
    model = construct(plan, "corrected", "six_single")
    return model.rollout(torch.zeros(1, 8, dtype=torch.float64),
                         torch.zeros(1, model.force_dimension, dtype=torch.float64))[0]


@torch.no_grad()
def nested_initial(model, common, plan):
    source = construct(plan, "corrected", "six_single")
    positions = source.step_trajectory(common[None]).x
    stride = plan["substeps"]//model.config.substeps
    nodes = positions[:, ::stride]
    velocity = (nodes[:, 1:]-nodes[:, :-1])/model.step_sizes[None, :, None, None]
    return model.encode_velocities(common[None, :8], velocity)[0]


def observations(model, data):
    rgb = torch.stack([data["noisy"][offset][frame] for frame, offset in model.view_specs])
    return Observations(model.view_frames, rgb, model.config.observation_sigma)


@torch.no_grad()
def point_and_interval_scores(model, fitted, data):
    mean = fitted["mean"]
    x = model.trajectory(mean[None]).x[0]
    error = x[:, model.free]-data["x"][:, model.free]
    mapping = model.position_map.flatten(0, 1)
    transformed = torch.linalg.solve_triangular(fitted["precision_cholesky"], mapping.T, upper=False)
    sd = transformed.square().sum(0).sqrt().reshape(model.config.frames, -1, 3)
    band = 1.6448536269514722*sd
    z = error/sd.clamp_min(1e-30)
    phi = torch.exp(-.5*z.square())/math.sqrt(2*math.pi)
    cdf = .5*(1+torch.erf(z/math.sqrt(2)))
    crps = sd*(z*(2*cdf-1)+2*phi-1/math.sqrt(math.pi))
    masks = {"all": torch.ones(model.config.frames, dtype=torch.bool),
             "original_gap": (model.times > .3+1e-8)&(model.times < .7-1e-8),
             "withheld": torch.ones(model.config.frames, dtype=torch.bool)}
    masks["withheld"][model.view_frames.unique()] = False
    scores = {}
    for label, mask in masks.items():
        if bool(mask.any()):
            scores[label] = {"frames": torch.where(mask)[0].tolist(),
                "position_rmse_m": float(error[mask].square().mean().sqrt()),
                "component_90_coverage": float((error[mask].abs() <= band[mask]).double().mean()),
                "component_90_mean_width_m": float((2*band[mask]).mean()),
                "component_gaussian_crps_m": float(crps[mask].mean())}
        else:
            scores[label] = None
    images = {}
    for offset in (0, 200):
        eye, rotation = orbit(model.times, offset_degrees=offset)
        rgb = torch.cat([model.renderer(x[None, a:a+3], eye[a:a+3], rotation[a:a+3])[0]
                         for a in range(0, model.config.frames, 3)])
        mse = (rgb-data["clean"][offset]).square().mean()
        scores[f"camera_{offset}_clean_psnr_db"] = float(-10*torch.log10(mse))
        images[offset] = rgb
    young, drag, _, _ = model.prior.decode(mean[None])
    scores.update({"young_pa": float(young[0]), "drag_per_s": float(drag[0]),
                   "interval_definition": "exact position pushforward of LOCAL GN Gaussian; conditional known scene; not calibrated"})
    return scores, {"positions_m": x.tolist(), "free_position_sd_m": sd.tolist()}, images


@torch.no_grad()
def force_distribution(model, fitted, data, count, seed):
    draws = trajectory_gaussian_draws(fitted, count, seed)
    valid, invalid = [], []
    for index, state in enumerate(draws):
        try:
            model.residual(state[None]); valid.append(index)
        except (InvalidDeformation, ValueError):
            invalid.append(index)
    if len(valid) < 4:
        raise RuntimeError("Too few valid local Gaussian trajectories; no silent replacement.")
    residual = model.residual(draws[valid])
    moments = recover_force_moments(model.elimination, residual)
    coefficients = model.elimination.sample_conditional(residual, torch.Generator().manual_seed(seed+1))
    prior = model.force_prior
    total = prior.total_force(coefficients, model.times)
    average = prior.total_force(moments["mean"][None], model.times)[0]
    pulse = torch.exp(-.5*((model.times-.5)/.08).square()) if data["condition"] == "external_pulse" else torch.zeros_like(model.times)
    truth = pulse[:, None]*(model.mesh.mass[model.free].sum()*model.times.new_tensor([.45, 0., -.30]))
    lo, hi = torch.quantile(total, model.times.new_tensor([.05, .95]), dim=0)
    fields = {"duration_s": model.config.duration, "times_s": model.times.tolist(),
        "inducing_times_s": prior.knots.tolist(), "force_prior": asdict(prior.config),
        "spatial_nodal_weights": prior.spatial_weights.tolist(),
        "external_total_mean_N": average.tolist(), "external_total_q05_N": lo.tolist(), "external_total_q95_N": hi.tolist(),
        "truth_external_total_N_evaluation_only": truth.tolist(),
        "whitened_force_coefficient_mean": moments["mean"].tolist(),
        "whitened_force_coefficient_covariance": moments["covariance"].tolist(),
        "whitened_force_coefficient_draws": coefficients.tolist(),
        "conditional_coefficient_covariance": moments["conditional_covariance"].tolist(),
        "between_trajectory_coefficient_covariance": moments["between_covariance"].tolist(),
        "inference": {"status": "local_Gaussian_approximation_not_calibrated",
            "stationarity_screen_pass": fitted["record"]["stationarity_screen_pass"],
            "invalid_trajectory_draws": len(invalid), "requested_draws": count,
            "qualification": "fixed force profile, full-rank independent nodal impulse discrepancy; no MCMC calibration"}}
    score = {"rmse_N": float((average-truth).square().mean().sqrt()),
        "component_90_coverage": float(((truth >= lo)&(truth <= hi)).double().mean()),
        "component_90_mean_width_N": float((hi-lo).mean()),
        "invalid_draw_indices": invalid, "valid_draws": len(valid)}
    return score, fields


@torch.no_grad()
def numerical_audit(plan, reference_root):
    model = construct(plan, "corrected", "six_single")
    results = {"integration": [], "rendering": []}
    for condition in plan["conditions"]:
        ref = json.loads((reference_root/f"{condition}.json").read_text())
        target = model.times.new_tensor(ref["positions_m"])
        raw = model.times.new_tensor([[math.log(19.5), math.log(.55), .11, -.035, .06, -.12, .04, .1]])
        theta = (raw-model.prior.mean)/model.prior.std
        for substeps in sorted(set((3, plan["substeps"], 2*plan["substeps"]))):
            local_plan = dict(plan, substeps=substeps, legacy_substeps=1)
            m = construct(local_plan, "corrected", "six_single")
            def external(t, x, previous):
                strength = math.exp(-.5*((t-.5)/.08)**2) if condition == "external_pulse" else 0.
                return strength*m.free[None, :, None]*m.times.new_tensor([.45, 0., -.30])
            state = m.rollout(theta, theta.new_zeros(1, 15), external_acceleration=external)
            difference = m.trajectory(state).x[0, :, m.free]-target[:, m.free]
            results["integration"].append({"condition": condition, "substeps": substeps,
                "position_rmse_m": float(difference.square().mean().sqrt()),
                "maximum_node_error_m": float(difference.norm(dim=-1).max())})
        indices = torch.tensor(SCHEMES["six_single"][0])
        rgb = {}
        for bins in sorted(set((plan["render_bins"], plan["reference_bins"], 2*plan["reference_bins"]))):
            r = ContinuousGaussianRenderer(model.mesh, 24, VolumeRenderConfig(bins=bins))
            rgb[bins] = torch.cat([r(target[None, indices[a:a+2]], model.eye[indices[a:a+2]],
                                     model.rotation[indices[a:a+2]])[0] for a in range(0, len(indices), 2)])
        fine = rgb[2*plan["reference_bins"]]
        for bins, image in rgb.items():
            error = image-fine
            results["rendering"].append({"condition": condition, "bins": bins,
                "reference_bins": 2*plan["reference_bins"], "rgb_rmse": float(error.square().mean().sqrt()),
                "rgb_max_error": float(error.abs().max())})
    if not plan["smoke_only"]:
        if any(r["position_rmse_m"] > .00015 for r in results["integration"] if r["substeps"] == plan["substeps"]):
            raise RuntimeError("Production integrator exceeds the declared 0.15 mm coordinate tolerance.")
        for row in results["rendering"]:
            limit = 3e-5 if row["bins"] == plan["render_bins"] else 2e-6
            if row["rgb_rmse"] > limit:
                raise RuntimeError("Volume quadrature exceeds its declared RGB tolerance.")
    return results


def run_accurate_study(plan, output, reference_root, resume=False):
    from .cli import source_fingerprint
    validate_plan(plan); plan = normalized(plan)
    torch.set_num_threads(plan["threads"])
    output, reference_root = Path(output).resolve(), Path(reference_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    reference_hashes = {c: hashlib.sha256((reference_root/f"{c}.json").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                        for c in plan["conditions"]}
    manifest = {"plan": plan, "source_fingerprint_sha256": fingerprint,
                "reference_hashes_normalized_lf": reference_hashes, "environment": _environment()}
    path = output/"manifest.json"
    if path.exists():
        if not resume:
            raise FileExistsError("Study exists; use --resume or a new directory.")
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Plan, source, environment or independent reference changed.")
    elif resume:
        raise FileNotFoundError("No study to resume.")
    write_json(path, manifest)
    audit_path = output/"numerical_audit.json"
    if not audit_path.exists():
        write_json(audit_path, numerical_audit(plan, reference_root))
    common = initial_state(plan)
    shared = construct(plan, "corrected", "six_single")
    shared_positions = shared.trajectory(common[None]).x.detach()
    jobs = [(v, "six_single", method) for v in plan["variants"]
            for method in (plan["methods"] if v == "corrected" else ["marginalized"])]
    jobs += [("corrected", scheme, "marginalized") for scheme in plan["schemes"] if scheme != "six_single"]
    records = []
    for condition in plan["conditions"]:
        data = inputs(plan, reference_root/f"{condition}.json", shared); data["condition"] = condition
        write_json(output/f"inputs_{condition}.json", {
            "reference_sha256_normalized_lf": data["reference_sha256_normalized_lf"],
            "positions_m_evaluation_only": data["x"].tolist(),
            "original_noise_sha256": data["original_noise_sha256"],
            "clean_rgb": {str(k): v.tolist() for k, v in data["clean"].items()},
            "noisy_rgb": {str(k): v.tolist() for k, v in data["noisy"].items()},
            "initial_frame_positions_m": shared_positions[0].tolist(),
            "noise_policy": "Original six primary-camera noise arrays reused pixelwise; extra primary/secondary seeds 304101/404101; reference motion RK4-96, new volume renderer"})
        for variant, scheme, method in jobs:
            if source_fingerprint() != fingerprint:
                raise RuntimeError("Source changed during study; keep inference source frozen.")
            label = f"{condition}__{variant}__{scheme}__{method}"
            destination = output/"cases"/label; destination.mkdir(parents=True, exist_ok=True)
            record_path = destination/"summary.json"
            if resume and record_path.exists():
                records.append(json.loads(record_path.read_text())); continue
            print("Accurate-reference study: "+label, flush=True)
            model = construct(plan, variant, scheme)
            observed = observations(model, data)
            start = nested_initial(model, common, plan)
            if not torch.allclose(model.trajectory(start[None]).x, shared_positions, atol=1e-14, rtol=1e-14):
                raise RuntimeError("Initial frame positions differ between controls.")
            checkpoint = destination/"fit.pt"
            began = time.perf_counter()
            if resume and checkpoint.exists():
                fitted = torch.load(checkpoint, weights_only=True)
            else:
                fitted = fit_trajectory(model, observed, start, method, TrajectoryFitConfig(**plan["optimizer"]),
                                        progress=lambda s: print(s, flush=True))
                torch.save(fitted, checkpoint)
            scores, point, images = point_and_interval_scores(model, fitted, data)
            write_json(destination/"point.json", point)
            torch.save(images, destination/"images.pt")
            record = {"case": label, "condition": condition, "variant": variant, "scheme": scheme, "method": method,
                "observed_slots": len(model.view_specs), "view_specs": model.view_specs,
                "observations_sha256": digest_tensor(observed.rgb), "source_fingerprint_sha256": fingerprint,
                "initial_frame_positions_sha256": digest_tensor(shared_positions),
                "initial_state_sha256": digest_tensor(start), "fit": fitted["record"], "scores": scores}
            if variant == "corrected" and scheme == "six_single":
                seed = plan["sampling_seed"]+(0 if condition == "no_load" else 1000)
                force_score, fields = force_distribution(model, fitted, data, plan["evaluation_samples"], seed)
                audit = audit_gaussian(model, observed, fitted, plan["audit_samples"], seed+100000)
                record.update(force_scores=force_score, local_gaussian_audit=audit)
                fields["inference"].update(source_fingerprint_sha256=fingerprint,
                    observations_sha256=record["observations_sha256"], local_gaussian_audit=audit)
                write_json(destination/"fields.json", fields)
            record["evaluation_seconds"] = time.perf_counter()-began
            write_json(record_path, record); records.append(record)
            print(f"  gap RMSE={scores['original_gap']['position_rmse_m']*1000:.3f} mm; "
                  f"stationary={fitted['record']['stationarity_screen_pass']}", flush=True)
    summary = {**manifest, "cases": records,
        "numerical_audit": json.loads(audit_path.read_text()),
        "scope": "two deterministic RK4-96 references, one toy known geometry; no certified stochastic-reference claim",
        "comparison": "all methods get identical continuous-reference images per case/scheme; component controls deliberately approximate image formation or dynamics",
        "initialization": "common prior-mean zero-load corrected rollout; nested-grid frame poses equal; no truth or image-based selection",
        "uncertainty": "local GN intervals and exact conditional Gaussian force; not a calibrated nonlinear posterior",
        "renderer": "projected EWA footprints with overlapping normal depth profiles; continuous absorption/emission volume surrogate, not exact 3D ray tracing"}
    write_json(output/"summary.json", summary)
    print(f"Completed comparison: {output/'summary.json'}", flush=True)
    return summary
