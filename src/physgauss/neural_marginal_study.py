"""Matched sparse-view comparison of marginalized forces and internal noise."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import torch
from .accurate_study import digest_tensor, point_and_interval_scores
from .benchmark import Config, Observations
from .camera import orbit
from .continuous_render import ContinuousGaussianRenderer, VolumeRenderConfig
from .force_study import field_scores
from .forces import ForceConfig
from .likelihood_audit import _environment
from .marginal_forces import MarginalConfig
from .marginal_inference import TrajectoryFitConfig
from .marginal_study import audit_gaussian
from .neural_marginal import InternalNoiseConfig, NeuralMarginalModel
from .neural_marginal_inference import fit_neural_marginal, force_mixture
from .stochastic import NeuralDiffusion, train_diffusion, calibration_law
from .stochastic_reference import coupled_reference, staggered_increments
from .study import normalized, write_json


CONDITIONS = {"no_load": (False, False), "external_pulse": (True, False),
              "stochastic": (False, True), "pulse_and_stochastic": (True, True)}
MODES = ("floor", "constant", "neural")


def default_plan(quick=False):
    return {"conditions": ["pulse_and_stochastic"] if quick else list(CONDITIONS),
        "seeds": [4101] if quick else [4101, 4102, 4103], "modes": list(MODES),
        "substeps": 2 if quick else 6, "observed_frames": [0, 3, 6, 14, 17, 20],
        "render_bins": 32 if quick else 128, "reference_render_bins": 128 if quick else 512,
        "reference_substeps": 32 if quick else 768,
        "reference_last_change_tolerance_m": .001 if quick else .00025,
        "production_stochastic_rmse_tolerance_m": .002 if quick else .0005,
        "optimizer": asdict(TrajectoryFitConfig(iterations=2 if quick else 20)),
        "force_prior": asdict(ForceConfig()), "trajectory_model": asdict(MarginalConfig()),
        "internal_rank": 3, "internal_spatial_length_m": .45,
        "calibration_seed": 918, "calibration_steps": 70 if quick else 350,
        "constant_calibration_seed": 105918, "constant_calibration_examples": 4096,
        "evaluation_samples": 8 if quick else 128, "audit_samples": 8 if quick else 128,
        "sampling_seed": 127100, "threads": 2, "smoke_only": quick}


def validate_plan(plan):
    for key, allowed in (("conditions", CONDITIONS), ("modes", MODES)):
        if not plan[key] or len(set(plan[key])) != len(plan[key]) or not set(plan[key]) <= set(allowed):
            raise ValueError("Invalid/repeated "+key)
    if not plan["seeds"] or len(set(plan["seeds"])) != len(plan["seeds"]) or any(not isinstance(s, int) or s < 0 for s in plan["seeds"]):
        raise ValueError("Need distinct nonnegative seeds.")
    Config(substeps=plan["substeps"], observed_frames=plan["observed_frames"], threads=plan["threads"]).validate()
    for k in ("evaluation_samples", "audit_samples", "constant_calibration_examples"):
        if plan[k] < 4:
            raise ValueError("Too few samples.")
    if plan["reference_substeps"] < 4 or plan["reference_substeps"] % 4 or plan["internal_rank"] != 3:
        raise ValueError("Need nested reference grids and the declared rank-three truth.")
    if plan["reference_last_change_tolerance_m"] <= 0:
        raise ValueError("Positive reference tolerance required.")
    if plan["reference_substeps"] % (4*plan["substeps"]):
        raise ValueError("Need nested Brownian grids for production refinement.")
    TrajectoryFitConfig(**plan["optimizer"]).validate()
    ForceConfig(**plan["force_prior"]).validate(); MarginalConfig(**plan["trajectory_model"]).validate()
    VolumeRenderConfig(bins=plan["render_bins"]).validate()
    VolumeRenderConfig(bins=plan["reference_render_bins"]).validate()


def calibrate(plan, path):
    path = Path(path)
    if path.exists():
        record = json.loads(path.read_text())
        if (record["network_training"]["seed"] != plan["calibration_seed"]
            or record["network_training"]["steps"] != plan["calibration_steps"]
            or record["constant_calibration"]["seed"] != plan["constant_calibration_seed"]
            or record["constant_calibration"]["examples"] != plan["constant_calibration_examples"]):
            raise ValueError("Stored calibration does not match the plan.")
        net = NeuralDiffusion()
        net.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in record["state_dict"].items()})
    else:
        net, stats = train_diffusion(seed=plan["calibration_seed"], steps=plan["calibration_steps"])
        generator = torch.Generator().manual_seed(plan["constant_calibration_seed"])
        features = torch.rand(plan["constant_calibration_examples"], 2, dtype=torch.float64, generator=generator)
        features = features*features.new_tensor([.7, 3.])-features.new_tensor([.25, 1.5])
        with torch.no_grad():
            predicted = net(features)
            truth = calibration_law(features)
            amplitude = float(predicted.square().mean().sqrt())
        record = {"architecture": "2 -> Linear(8) -> tanh -> Linear(1) -> .001 softplus + .0001",
            "network_training": stats, "constant_amplitude": amplitude,
            "constant_calibration": {"seed": plan["constant_calibration_seed"], "examples": len(features),
                "definition": "RMS neural amplitude on separate calibration features; same spatial correlation and rank",
                "neural_relative_sigma_rmse_vs_analytic_law": float((predicted/truth-1).square().mean().sqrt()),
                "constant_relative_sigma_rmse_vs_analytic_law": float((amplitude/truth-1).square().mean().sqrt())},
            "state_dict": {k: v.detach().tolist() for k, v in net.state_dict().items()},
            "qualification": "independent synthetic simulator supervision; weights frozen; no test motion or withheld images used"}
        write_json(path, record)
    net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    return net, record


def construct(plan, network, calibration, mode):
    c = Config(substeps=plan["substeps"], observed_frames=plan["observed_frames"], truth_stochastic=False,
               threads=plan["threads"])
    internal = InternalNoiseConfig(mode, plan["internal_rank"], plan["internal_spatial_length_m"], calibration["constant_amplitude"])
    m = NeuralMarginalModel(c, network, internal, force_config=ForceConfig(**plan["force_prior"]),
                           marginal_config=MarginalConfig(**plan["trajectory_model"]))
    m.set_views([(i, 0) for i in c.observed_frames])
    m.renderer = ContinuousGaussianRenderer(m.mesh, 24, VolumeRenderConfig(bins=plan["render_bins"]))
    return m


@torch.no_grad()
def production_refinement(plan, reference, pulse):
    # Evaluation-only oracle amplitude/analytic loading isolate numerical
    # integration error from network calibration and force-basis error.
    class Oracle(torch.nn.Module):
        def forward(self, features): return calibration_law(features)
    rows = []
    for substeps in (plan["substeps"], 2*plan["substeps"]):
        model = NeuralMarginalModel(Config(substeps=substeps), Oracle())
        raw = model.times.new_tensor([[math.log(19.5), math.log(.55), .11, -.035, .06, -.12, .04, .10]])
        theta = (raw-model.prior.mean)/model.prior.std
        increments = staggered_increments(reference["brownian_finest_increments"], model.nsteps)[None]
        def external(t, x, previous):
            strength = math.exp(-.5*((t-.5)/.08)**2) if pulse else 0.
            return strength*model.free[None, :, None]*x.new_tensor([.45, 0., -.30])
        state = model.internal_rollout(theta, theta.new_zeros(1, 15), increments, external_acceleration=external)
        error = model.trajectory(state).x[0, :, model.free]-reference["positions"][:, model.free]
        rows.append({"substeps": substeps, "coordinate_rmse_vs_fine_reference_m": float(error.square().mean().sqrt()),
                     "maximum_node_error_m": float(error.norm(dim=-1).max())})
    if rows[0]["coordinate_rmse_vs_fine_reference_m"] > plan["production_stochastic_rmse_tolerance_m"]:
        raise RuntimeError("Production stochastic integrator exceeds predeclared coordinate tolerance.")
    return {"rows": rows, "qualification": "evaluation-only true amplitude and analytic load, zero floor realization; not a posterior time-refinement study"}


@torch.no_grad()
def generate_data(plan, model, condition, seed, reference_root, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    reference_path = destination/"reference.pt"
    pulse, stochastic = CONDITIONS[condition]
    if reference_path.exists():
        reference = torch.load(reference_path, weights_only=True)
    elif stochastic:
        print(f"  coupled stochastic reference: {condition}, seed {seed}", flush=True)
        reference = coupled_reference(model.times, seed, pulse, plan["reference_substeps"], plan["reference_last_change_tolerance_m"])
        reference["audit"]["production_staggered_refinement"] = production_refinement(plan, reference, pulse)
        torch.save(reference, reference_path)
    else:
        source = Path(reference_root)/f"{condition}.json"
        original = json.loads(source.read_text())
        reference = {"positions": model.times.new_tensor(original["positions_m"]),
            "velocities": model.times.new_tensor(original["velocities_m_per_s"]),
            "audit": {"integrator": "independent RK4-96", "source_sha256_normalized_lf": hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                "RK4_48_vs_96_coordinate_rmse_m": original["RK4_48_vs_96_coordinate_rmse_m"]}}
        torch.save(reference, reference_path)
    data_file = destination/"inputs.json"
    if data_file.exists():
        stored = json.loads(data_file.read_text())
        clean = {int(k): model.times.new_tensor(v) for k, v in stored["clean_rgb"].items()}
        observed = model.times.new_tensor(stored["observed_rgb"])
    else:
        x = reference["positions"][None]
        renderer = ContinuousGaussianRenderer(model.mesh, 24, VolumeRenderConfig(bins=plan["reference_render_bins"]))
        clean = {}
        for offset in (0, 200):
            eye, rotation = orbit(model.times, offset_degrees=offset)
            clean[offset] = torch.cat([renderer(x[:, a:a+3], eye[a:a+3], rotation[a:a+3])[0] for a in range(0, len(model.times), 3)])
        selected = clean[0][model.view_frames]
        observed = selected+.015*torch.randn(selected.shape, dtype=selected.dtype, generator=torch.Generator().manual_seed(seed+100000))
        reused = False
        published = Path(reference_root).parents[1]/"accurate-reference"/f"inputs_{condition}.json"
        if not stochastic and seed == 4101 and not plan["smoke_only"] and published.exists():
            # Exact historical corrected input, including its floating residual.
            old = json.loads(published.read_text())
            torch.testing.assert_close(clean[0], model.times.new_tensor(old["clean_rgb"]["0"]), rtol=0., atol=0.)
            observed = model.times.new_tensor(old["noisy_rgb"]["0"])[model.view_frames]
            reused = True
        stored = {"condition": condition, "seed": seed, "observed_indices": model.view_frames.tolist(),
            "observed_rgb": observed.tolist(), "observations_sha256": digest_tensor(observed),
            "times_s": model.times.tolist(), "positions_m_evaluation_only": reference["positions"].tolist(),
            "clean_rgb": {str(k): v.tolist() for k, v in clean.items()},
            "reference_audit": reference["audit"], "reused_exact_v06_six_images": reused,
            "noise_seed": seed+100000, "noise_sigma": .015,
            "noise_policy": "common across all inference models; v0.6 deterministic seed-4101 input reused when present"}
        # Large image arrays stay portable without hundreds of thousands of
        # indentation-only lines; all floating values are retained exactly.
        temporary = data_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(stored, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        temporary.replace(data_file)
    return {"x": reference["positions"], "clean": clean, "condition": condition,
            "reference": reference, "stored": stored}, Observations(model.view_frames, observed, .015)


@torch.no_grad()
def force_results(model, fitted, condition, plan, seed):
    result = force_mixture(model, fitted, plan["evaluation_samples"], seed)
    prior = model.force_prior
    total = prior.total_force(result["coefficient_draws"], model.times)
    mean = prior.total_force(result["mean"][None], model.times)[0]
    pulse = torch.exp(-.5*((model.times-.5)/.08).square()) if CONDITIONS[condition][0] else torch.zeros_like(model.times)
    truth = pulse[:, None]*model.mesh.mass[model.free].sum()*model.times.new_tensor([.45, 0., -.30])
    score = field_scores(total, truth, "N")
    score["rmse_N"] = float((mean-truth).square().mean().sqrt())
    lo, hi = torch.quantile(total, model.times.new_tensor([.05, .95]), dim=0)
    fields = {"duration_s": 1., "times_s": model.times.tolist(), "inducing_times_s": prior.knots.tolist(),
        "force_prior": asdict(prior.config), "spatial_nodal_weights": prior.spatial_weights.tolist(),
        "external_total_mean_N": mean.tolist(), "external_total_q05_N": lo.tolist(), "external_total_q95_N": hi.tolist(),
        "truth_external_total_N_evaluation_only": truth.tolist(),
        "whitened_force_coefficient_mean": result["mean"].tolist(), "whitened_force_coefficient_covariance": result["covariance"].tolist(),
        "whitened_force_coefficient_draws": result["coefficient_draws"].tolist(),
        "conditional_coefficient_covariance": result["conditional_covariance"].tolist(),
        "between_trajectory_coefficient_covariance": result["between_covariance"].tolist(),
        "conditional_covariance_definition": "AVERAGE S(z) over valid local trajectory draws; it varies with state",
        "conditional_covariance_trace_min_max": result["conditional_covariance_trace_min_max"],
        "inference": {"status": "local_Gaussian_approximation_not_calibrated", "internal_mode": model.internal_config.mode,
            "stationarity_screen_pass": fitted["record"]["stationarity_screen_pass"], "invalid_trajectory_draws": len(result["invalid_indices"]),
            "requested_draws": plan["evaluation_samples"], "qualification": "frozen simulator-calibrated network; fixed force profile; full-rank floor retained"}}
    score.update(invalid_draw_indices=result["invalid_indices"], valid_draws=result["valid_samples"])
    return score, fields


def aggregate(records):
    result = []
    for condition in dict.fromkeys(r["condition"] for r in records):
        for mode in dict.fromkeys(r["mode"] for r in records):
            group = [r for r in records if r["condition"] == condition and r["mode"] == mode]
            result.append({"condition": condition, "mode": mode, "seeds": [r["seed"] for r in group],
                "mean_gap_position_rmse_m": sum(r["scores"]["original_gap"]["position_rmse_m"] for r in group)/len(group),
                "mean_gap_component_90_coverage": sum(r["scores"]["original_gap"]["component_90_coverage"] for r in group)/len(group),
                "mean_gap_component_90_width_m": sum(r["scores"]["original_gap"]["component_90_mean_width_m"] for r in group)/len(group),
                "mean_gap_component_crps_m": sum(r["scores"]["original_gap"]["component_gaussian_crps_m"] for r in group)/len(group),
                "mean_force_rmse_N": sum(r["force_scores"]["rmse_N"] for r in group)/len(group),
                "stationary_count": sum(r["fit"]["stationarity_screen_pass"] for r in group),
                "importance_screen_count": sum(r["local_gaussian_audit"]["screen_pass"] for r in group),
                "qualification": "three-seed exploratory averages; correlated component coverage is not established calibration"})
    return result


def run_study(plan, output, reference_root, resume=False):
    from .cli import source_fingerprint
    validate_plan(plan); plan = normalized(plan); torch.set_num_threads(plan["threads"])
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    network, calibration = calibrate(plan, output/"calibration.json")
    calibration_hash = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    root = Path(reference_root).resolve()
    references = {name: hashlib.sha256((root/f"{name}.json").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                  for name in ("no_load", "external_pulse")}
    manifest = {"plan": plan, "source_fingerprint_sha256": fingerprint, "environment": _environment(),
                "calibration_sha256_canonical_json": calibration_hash, "deterministic_reference_hashes": references}
    path = output/"manifest.json"
    if path.exists():
        if not resume: raise FileExistsError("Study exists; choose a new output or --resume.")
        if json.loads(path.read_text()) != manifest: raise ValueError("Source, environment, calibration, plan or reference changed.")
    elif resume: raise FileNotFoundError("No matching study to resume.")
    write_json(path, manifest)
    shared = construct(plan, network, calibration, "floor")
    with torch.no_grad():
        initial = shared.rollout(shared.times.new_zeros(1, 8), shared.times.new_zeros(1, 15))[0]
    initial_hash = digest_tensor(initial)
    records = []
    for condition in plan["conditions"]:
        for seed in plan["seeds"]:
            label = f"{condition}__{seed}"
            data, observed = generate_data(plan, shared, condition, seed, root, output/"datasets"/label)
            data_hash = digest_tensor(observed.rgb)
            for mode in plan["modes"]:
                if source_fingerprint() != fingerprint: raise RuntimeError("Numerical source changed during study.")
                destination = output/"cases"/f"{label}__{mode}"; destination.mkdir(parents=True, exist_ok=True)
                record_path = destination/"summary.json"
                if resume and record_path.exists(): records.append(json.loads(record_path.read_text())); continue
                print(f"Neural marginal study: {label}__{mode}", flush=True)
                model = construct(plan, network, calibration, mode)
                checkpoint = destination/"fit.pt"
                if resume and checkpoint.exists(): fitted = torch.load(checkpoint, weights_only=True)
                else:
                    fitted = fit_neural_marginal(model, observed, initial, TrajectoryFitConfig(**plan["optimizer"]),
                                                progress=lambda s: print(s, flush=True))
                    torch.save(fitted, checkpoint)
                scores, point, images = point_and_interval_scores(model, fitted, data)
                scores["interval_definition"] = "linear pushforward of local Gaussian mean+covariance metric; not exact posterior curvature or calibrated uncertainty"
                sample_seed = plan["sampling_seed"]+seed
                force_score, fields = force_results(model, fitted, condition, plan, sample_seed)
                audit = audit_gaussian(model, observed, fitted, plan["audit_samples"], sample_seed+100000)
                fields["inference"].update(source_fingerprint_sha256=fingerprint, observations_sha256=data_hash,
                    local_gaussian_audit=audit, calibration_sha256_canonical_json=calibration_hash)
                record = {"case": f"{label}__{mode}", "condition": condition, "seed": seed, "mode": mode,
                    "observations_sha256": data_hash, "initial_state_sha256": initial_hash,
                    "source_fingerprint_sha256": fingerprint, "internal_noise": asdict(model.internal_config),
                    "fit": fitted["record"], "scores": scores, "force_scores": force_score, "local_gaussian_audit": audit}
                torch.save(images, destination/"images.pt")
                write_json(destination/"point.json", point); write_json(destination/"fields.json", fields)
                write_json(record_path, record); records.append(record)
                print(f"  gap={1000*scores['original_gap']['position_rmse_m']:.3f} mm; "
                      f"coverage={scores['original_gap']['component_90_coverage']:.1%}; stationary={record['fit']['stationarity_screen_pass']}; "
                      f"weight ESS={audit['relative_weight_ess']:.3f}", flush=True)
    summary = {**manifest, "cases": records, "aggregates": aggregate(records),
        "scope": "one moving camera, six irregular timestamps; four synthetic conditions; three realization/image-noise seeds; fixed known geometry/cameras",
        "models": "all have the SAME full-rank floor and exactly marginalized structured external force; constant/neural add balanced internal rank-three noise",
        "inference": "state-dependent Q and log determinants differentiated; exact conditional force, approximate local Gaussian trajectory",
        "qualification": "three-seed exploratory comparison, not established calibration; no learned geometry/camera or test-video network training"}
    write_json(output/"summary.json", summary)
    print("Completed neural marginal study:", output/"summary.json", flush=True)
    return summary
