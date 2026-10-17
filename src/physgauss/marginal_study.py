"""Matched sparse-image experiment for exact Gaussian force elimination.

Truth is accessed only by orchestration after fitting. The inference classes
accept observations, known scene definitions and fixed priors only.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import torch
from .benchmark import Config, make_benchmark
from .camera import orbit
from .force_study import CONDITIONS, field_scores, truth_force_record
from .forces import ForceConfig, ForceSceneModel
from .inference import fit_map
from .likelihood_audit import _environment
from .marginal_forces import MarginalConfig, TrajectoryForceModel
from .marginal_inference import (TrajectoryFitConfig, fit_trajectory,
                                 recover_force_moments, trajectory_gaussian_draws)
from .metrics import evaluate, physics_diagnostics
from .physics import InvalidDeformation
from .study import normalized, write_json


METHODS = ("alternating", "joint", "marginalized")


def quick_marginal_plan():
    return {"base": asdict(Config.quick()), "seeds": [4101], "conditions": ["external_pulse"],
        "methods": list(METHODS), "force_prior": asdict(ForceConfig()),
        "trajectory_model": asdict(MarginalConfig()),
        "optimizer": asdict(TrajectoryFitConfig(iterations=3)), "evaluation_samples": 16,
        "audit_samples": 16, "sampling_seed": 92001}


def validate_plan(plan):
    Config(**plan["base"]).validate()
    ForceConfig(**plan["force_prior"]).validate()
    MarginalConfig(**plan["trajectory_model"]).validate()
    TrajectoryFitConfig(**plan["optimizer"]).validate()
    for key, allowed in (("conditions", CONDITIONS), ("methods", METHODS)):
        if not plan[key] or len(plan[key]) != len(set(plan[key])) or not set(plan[key]) <= set(allowed):
            raise ValueError("Invalid or repeated "+key)
    if not plan["seeds"] or len(plan["seeds"]) != len(set(plan["seeds"])) or any(
            not isinstance(s, int) or s < 0 for s in plan["seeds"]):
        raise ValueError("Need distinct nonnegative realization seeds.")
    if min(plan["evaluation_samples"], plan["audit_samples"]) < 4:
        raise ValueError("Need at least four evaluation and independent audit draws.")


@torch.no_grad()
def audit_gaussian(model, observations, fitted, count, seed):
    """Independent importance-weight concentration diagnostic, not correction."""
    draws = trajectory_gaussian_draws(fitted, count, seed)
    center = model.objective(fitted["mean"][None], observations)[0]
    ratios, invalid = [], 0
    for state in draws.split(4):
        # Isolate invalid states; never replace/resample them.
        for point in state:
            white = (point-fitted["mean"])@fitted["precision_cholesky"]
            try:
                value = -model.objective(point[None], observations)[0]+center+.5*white.square().sum()
            except (InvalidDeformation, ValueError):
                value = point.new_tensor(-torch.inf); invalid += 1
            ratios.append(value)
    ratios = torch.stack(ratios)
    if not bool(torch.isfinite(ratios).any()):
        return {"samples": count, "invalid_samples": invalid, "relative_weight_ess": 0.,
                "maximum_normalized_weight": 1., "screen_pass": False,
                "qualification": "all local Gaussian proposals invalid; no corrected posterior"}
    if bool(torch.isnan(ratios).any()):
        raise RuntimeError("Nonfinite importance ratio.")
    weights = torch.softmax(ratios, 0)
    relative = float(1/(weights.square().sum()*count))
    return {"samples": count, "invalid_samples": invalid, "relative_weight_ess": relative,
            "maximum_normalized_weight": float(weights.max()), "screen_pass": relative >= .1 and invalid == 0,
            "qualification": "independent full nonlinear marginal-target/proposal weight concentration; passing is not calibration or proof of adequate sampling"}


@torch.no_grad()
def evaluate_fit(model, fitted, benchmark, count, seed):
    coordinates = trajectory_gaussian_draws(fitted, count, seed)
    valid, invalid = [], []
    # Preserve and report every invalid Gaussian draw; do not silently resample.
    for i, point in enumerate(coordinates):
        try:
            model.residual(point[None]); valid.append(i)
        except (InvalidDeformation, ValueError):
            invalid.append(i)
    if len(valid) < 4:
        raise RuntimeError("Too few valid local Gaussian draws to evaluate; inspect the saved fit.")
    selected = coordinates[valid]
    groups = {k: [] for k in ("x", "v", "rgb", "novel_rgb")}
    novel = orbit(model.times, offset_degrees=100)
    for state in selected.split(4):
        trajectory = model.trajectory(state)
        for key, value in (("x", trajectory.x), ("v", trajectory.v),
                           ("rgb", model.renderer(trajectory.x, model.eye, model.rotation)),
                           ("novel_rgb", model.renderer(trajectory.x, *novel))):
            groups[key].append(value)
    prediction = {k: torch.cat(v) for k, v in groups.items()}
    residual = model.residual(selected)
    moments = recover_force_moments(model.elimination, residual)
    generator = torch.Generator().manual_seed(seed+1)
    coefficient_draws = model.elimination.sample_conditional(residual, generator)
    prior = model.force_prior
    total = prior.total_force(coefficient_draws, model.step_times)
    total_mean = prior.total_force(moments["mean"][None], model.step_times)[0]
    true_force, _ = truth_force_record(model.config, benchmark)
    true_total = true_force[0].sum(1)
    force_scores = field_scores(total, true_total, "N")
    force_scores["rmse_N"] = float((total_mean-true_total).square().mean().sqrt())
    force_scores["mean_note"] = "Rao-Blackwell coefficient mean; empirical intervals/CRPS from paired conditional force draws"
    score = evaluate(prediction, benchmark, model.mesh)
    score["definitions"]["uncertainty"] = "local Gauss-Newton trajectory Gaussian plus exact conditional forces; not calibrated"
    score["definitions"]["invalid_draw_policy"] = "no replacement; if invalid draws exist, scores condition on validity and are explicitly flagged"
    young = model.prior.decode(selected)[0]
    physics = physics_diagnostics(prediction, model.mesh, young)
    point = model.trajectory(fitted["mean"][None]).x[0]
    withheld = torch.ones(model.config.frames, dtype=torch.bool)
    withheld[benchmark.observations.indices] = False
    point_rmse = float((point[withheld][:, model.free]-benchmark.truth.x[0, withheld][:, model.free]).square().mean().sqrt())
    lo, hi = torch.quantile(total, total.new_tensor([.05, .95]), dim=0)
    force_mean = moments["mean"].tolist()
    force_covariance = moments["covariance"].tolist()
    fields = {"duration_s": model.config.duration, "times_s": model.step_times.tolist(),
        "inducing_times_s": prior.knots.tolist(), "force_prior": asdict(prior.config),
        "spatial_nodal_weights": prior.spatial_weights.tolist(),
        "external_total_mean_N": total_mean.tolist(), "external_total_q05_N": lo.tolist(), "external_total_q95_N": hi.tolist(),
        "truth_external_total_N_evaluation_only": true_total.tolist(),
        "whitened_force_coefficient_mean": force_mean,
        "whitened_force_coefficient_covariance": force_covariance,
        "whitened_force_coefficient_draws": coefficient_draws.tolist(),
        "conditional_coefficient_covariance": moments["conditional_covariance"].tolist(),
        "between_trajectory_coefficient_covariance": moments["between_covariance"].tolist(),
        "coefficient_draws_note": "paired local Gaussian trajectory draws and exact Gaussian force conditionals; moments use total-covariance decomposition",
        "inference": {"status": "local_Gaussian_approximation_not_calibrated", "stationarity_screen_pass": fitted["record"]["stationarity_screen_pass"],
            "invalid_trajectory_draws": len(invalid), "requested_draws": count,
            "qualification": "conditional full-rank trajectory-discrepancy model; fixed force profile; not the old low-rank SDE model"}}
    record = {"trajectory_scores": score, "point_withheld_position_rmse_m": point_rmse,
              "force_scores": force_scores, "physics": physics,
              "evaluation_samples_requested": count, "valid_evaluation_samples": len(valid),
              "invalid_evaluation_draw_indices": invalid,
              "force_coefficient_uncertainty": {"conditional_trace": float(moments["conditional_covariance"].trace()),
                  "between_trajectory_trace": float(moments["between_covariance"].trace())}}
    return record, fields, {"coordinates": coordinates, "valid_indices": torch.tensor(valid),
                            "force_coefficients_paired_with_valid_coordinates": coefficient_draws}


def historical_comparison(path, records, force_prior):
    if path is None:
        return {"status": "not supplied", "cases": []}
    path = Path(path)
    content = path.read_bytes()
    historical = json.loads(content)
    if historical["plan"]["force_prior"] != force_prior:
        raise ValueError("Historical force prior differs; do not label the comparison matched.")
    pairs = {(r["condition"], r["seed"]): r for r in records}
    cases = []
    for old in historical["cases"]:
        key = (old["condition"], old["seed"])
        if key not in pairs:
            continue
        current = pairs[key]
        if old["observations_sha256"] != current["observations_sha256"] or old["config"] != current["config"]:
            raise ValueError("Historical data/config mismatch; generate the same benchmark before comparing.")
        cases.append({"condition": key[0], "seed": key[1], "method": old["method"],
            "observations_sha256": old["observations_sha256"],
            "withheld_position_rmse_m": old["trajectory_scores"]["withheld"]["position_rmse_m"],
            "force_rmse_N": old["force_scores"]["external_total"]["rmse_N"],
            "all_mcmc_checks_pass": old["diagnostics"]["all_monitored_quantities_pass"],
            "seconds": old["seconds"], "latent_dimension": old["latent_dimension"]})
    if len(cases) != 3*len(pairs):
        raise ValueError("Historical three-model reference is incomplete for these cases.")
    return {"status": "same observations, geometry, mechanics and force prior; DIFFERENT process-noise model and inference",
            "summary_sha256_normalized_lf": hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest(),
            "source_fingerprint_sha256": historical["source_fingerprint_sha256"], "cases": cases}


def run_marginal_study(plan, output, resume=False, historical_summary=None):
    from .cli import source_fingerprint
    from .marginal_report import make_marginal_report
    validate_plan(plan)
    plan = normalized(plan)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    historical_digest = (hashlib.sha256(Path(historical_summary).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                         if historical_summary is not None else None)
    manifest = {"plan": plan, "source_fingerprint_sha256": fingerprint, "environment": _environment(),
                "historical_summary_sha256_normalized_lf": historical_digest}
    path = output/"manifest.json"
    if path.exists():
        if not resume:
            raise FileExistsError("Marginal study exists; use --resume or a new output.")
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Plan, source, environment or historical reference changed.")
    elif resume:
        raise FileNotFoundError("No matching marginal study exists to resume.")
    write_json(path, manifest)
    torch.set_num_threads(Config(**plan["base"]).threads)
    records = []
    for condition in plan["conditions"]:
        pulse, stochastic = CONDITIONS[condition]
        for seed in plan["seeds"]:
            config = Config(**dict(plan["base"], seed=seed, truth_stochastic=stochastic, truth_external_pulse=pulse))
            benchmark = make_benchmark(config)
            observed_hash = hashlib.sha256(benchmark.observations.rgb.numpy().tobytes()).hexdigest()
            model = TrajectoryForceModel(config, ForceConfig(**plan["force_prior"]), MarginalConfig(**plan["trajectory_model"]))
            root = output/"cases"/f"{condition}__{seed}"
            root.mkdir(parents=True, exist_ok=True)
            warm_path = root/"initialization.pt"
            print(f"Trajectory study: {condition}__{seed}", flush=True)
            if warm_path.exists() and resume:
                warm = torch.load(warm_path, weights_only=True)
                if warm["source_fingerprint"] != fingerprint or warm["observations_sha256"] != observed_hash:
                    raise ValueError("Warm-start provenance mismatch.")
            else:
                # Same observation-only existing force fit initializes EVERY method.
                forward = ForceSceneModel(config, force_config=model.force_prior.config)
                mean, _, seconds, value, gradient = fit_map(forward, benchmark.observations,
                    config.adam_steps, config.lbfgs_steps, progress=lambda s: print(s, flush=True))
                with torch.no_grad():
                    initial = model.rollout(mean[None, :8], mean[None, 8:])[0]
                warm = {"initial": initial, "forward_coordinates": mean, "seconds": seconds,
                        "forward_objective": value, "forward_gradient_norm": gradient,
                        "source_fingerprint": fingerprint, "observations_sha256": observed_hash,
                        "observed_rgb": benchmark.observations.rgb, "observed_indices": benchmark.observations.indices,
                        "observation_sigma": benchmark.observations.sigma}
                torch.save(warm, warm_path)
            for method in plan["methods"]:
                if source_fingerprint() != fingerprint:
                    raise RuntimeError("Source changed during marginal study.")
                destination = root/method; destination.mkdir(parents=True, exist_ok=True)
                if resume and (destination/"summary.json").exists():
                    records.append(json.loads((destination/"summary.json").read_text())); continue
                began = time.perf_counter()
                checkpoint = destination/"fit.pt"
                if resume and checkpoint.exists():
                    fitted = torch.load(checkpoint, weights_only=True)
                else:
                    fitted = fit_trajectory(model, benchmark.observations, warm["initial"], method,
                        TrajectoryFitConfig(**plan["optimizer"]), progress=lambda s: print(s, flush=True))
                    torch.save(fitted, checkpoint)
                sample_seed = plan["sampling_seed"]+seed
                evaluation, fields, draws = evaluate_fit(model, fitted, benchmark, plan["evaluation_samples"], sample_seed)
                audit = audit_gaussian(model, benchmark.observations, fitted, plan["audit_samples"], sample_seed+100000)
                record = {"condition": condition, "seed": seed, "method": method, "config": asdict(config),
                    "observations_sha256": observed_hash, "source_fingerprint_sha256": fingerprint,
                    "initialization": {k: warm[k] for k in ("seconds", "forward_objective", "forward_gradient_norm")},
                    "initial_state_sha256": hashlib.sha256(warm["initial"].numpy().tobytes()).hexdigest(),
                    "fit": fitted["record"], "local_gaussian_audit": audit, **evaluation,
                    "seconds_excluding_shared_initialization": time.perf_counter()-began}
                fields["inference"].update({"source_fingerprint_sha256": fingerprint,
                    "observations_sha256": observed_hash, "local_gaussian_audit": audit})
                torch.save(draws, destination/"draws.pt")
                write_json(destination/"fields.json", fields)
                write_json(destination/"summary.json", record)
                records.append(record)
                print(f"  withheld RMSE={record['point_withheld_position_rmse_m']*1000:.3f} mm; "
                    f"stationarity={record['fit']['stationarity_screen_pass']}; relative weight ESS={audit['relative_weight_ess']:.3g}", flush=True)
    summary = {**manifest, "cases": records,
        "historical_reference": historical_comparison(historical_summary, records, plan["force_prior"]),
        "scope": "known geometry/cameras; same full-rank velocity-discrepancy model for the three trajectory solvers",
        "inference": "exact Gaussian force elimination; approximate local Gauss-Newton trajectory/material posterior",
        "qualification": "single geometry; fixed hyperparameters; optimizer stationarity and weight-concentration diagnostics are not calibration",
        "old_model_difference": "new full-rank independent free-node velocity diffusion; old model has no such floor and uses optional balanced low-rank neural internal impulses"}
    write_json(output/"summary.json", summary)
    make_marginal_report(output, summary)
    print(f"Marginal report: {output/'report.html'}", flush=True)
    return summary
