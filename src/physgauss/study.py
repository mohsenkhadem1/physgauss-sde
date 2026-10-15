"""Matched linearized controls, adaptive noise inference, and seed/gap studies."""
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .benchmark import Config, SceneModel, make_benchmark
from .inference import fit, predict_samples
from .metrics import evaluate, interval_metrics, physics_diagnostics, uncertainty_ranking
from .scale_inference import draw_coordinates, infer_scale_posterior, summarize_scale
from .stochastic import train_diffusion


METHODS = {"physical": 0., "fixed_sde": 1., "adaptive_sde": None}


def normalized(value):
    return json.loads(json.dumps(value))


def write_json(path, value):
    """Replace an experiment record atomically to make interrupted studies resumable."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_plan(plan):
    required = {"base", "seeds", "schedules", "conditions", "scales",
                "spike_probability", "slab_std"}
    if set(plan) != required:
        raise ValueError(f"Plan keys must be exactly {sorted(required)}.")
    if not plan["seeds"] or len(plan["seeds"]) != len(set(plan["seeds"])):
        raise ValueError("Use distinct independent realization seeds.")
    if set(plan["conditions"]) - {"deterministic", "stochastic", "unmodeled_pulse"}:
        raise ValueError("Unknown ground-truth condition.")
    if not plan["conditions"] or len(plan["conditions"]) != len(set(plan["conditions"])):
        raise ValueError("Use distinct ground-truth conditions.")
    if not plan["schedules"]:
        raise ValueError("Need at least one observation schedule.")
    sizes = set()
    for name, schedule in plan["schedules"].items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Schedule names must be alphanumeric or underscore.")
        Config(**dict(plan["base"], observed_frames=schedule)).validate()
        sizes.add(len(schedule))
    if len(sizes) != 1:
        raise ValueError("Matched schedule comparisons must use the same observation budget.")
    from .scale_inference import scale_prior
    scale_prior(plan["scales"], plan["spike_probability"], plan["slab_std"])
    if 1. not in plan["scales"]:
        raise ValueError("The fixed unit-scale control must be present in the grid.")
    if any(not isinstance(seed, int) or seed < 0 for seed in plan["seeds"]):
        raise ValueError("Seeds must be nonnegative integers.")
    if plan["base"].get("diffusion_seed", 918) in plan["seeds"]:
        raise ValueError("Diffusion calibration seed must be separate from evaluation seeds.")
    return plan


def quick_plan():
    base = asdict(Config.quick())
    base.pop("observed_frames")
    base.pop("seed")
    return {"base": base, "seeds": [3101], "schedules": {"smoke": [0, 2, 6, 8]},
            "conditions": ["deterministic", "stochastic"],
            "scales": [0., .25, .5, 1., 2.], "spike_probability": .5, "slab_std": 1.}


def predict_mixture(model, coordinates, scales):
    """Batch by physical scale, preserving original sample order and correlations."""
    combined = None
    for scale in scales.unique(sorted=True):
        selected = scales == scale
        component_model = SceneModel(model.config, model.diffusion, process_scale=float(scale))
        predicted = predict_samples(component_model, coordinates[selected])
        if combined is None:
            combined = {key: value.new_empty((len(coordinates), *value.shape[1:]))
                        for key, value in predicted.items()}
        for key, value in predicted.items():
            combined[key][selected] = value
    return combined


def run_case(config, diffusion, grid, spike_probability, slab_std, output, fingerprint):
    """Only this orchestration layer accesses truth; scale inference gets observations."""
    start = time.perf_counter()
    benchmark = make_benchmark(config)
    reference_model = SceneModel(config)
    reference_fit = fit(reference_model, benchmark.observations, progress=None)
    model = SceneModel(config, diffusion)
    posterior = infer_scale_posterior(model, benchmark.observations, reference_fit.mean,
                                     grid, spike_probability, slab_std, progress=None)
    summary = {"config": asdict(config), "source_fingerprint_sha256": fingerprint,
               "reference_gauss_newton_decrement": reference_fit.newton_decrement,
               "scale_posterior": summarize_scale(posterior), "methods": {},
               "truth_parameters_evaluation_only": benchmark.truth_parameters,
               "inference": "matched common local image linearization with exact Gaussian marginalization per scale"}
    observed = torch.zeros(config.frames, dtype=torch.bool)
    observed[benchmark.observations.indices] = True
    free = ~model.mesh.pinned
    target = benchmark.truth.x[0, ~observed][:, free].flatten(1)
    with torch.no_grad():
        for method, fixed_scale in METHODS.items():
            coordinates, scales, components = draw_coordinates(
                posterior, config.posterior_samples, config.seed + 801, fixed_scale)
            samples = predict_mixture(model, coordinates, scales)
            scores = evaluate(samples, benchmark, model.mesh)
            scores["definitions"]["uncertainty"] = summary["inference"]
            positions = samples["x"][:, ~observed][:, :, free].flatten(2)
            coverage = {str(level): interval_metrics(positions, target, level)
                        for level in [.5, .8, .9, .95]}
            ranking = uncertainty_ranking(positions, target)
            scaled_coordinates = coordinates.clone()
            scaled_coordinates[:, 8:] *= scales[:, None]
            # Actual nonlinear observation renders versus the *same* linearized model.
            linear_rgb = posterior.reference_rgb.flatten()[None] + benchmark.observations.sigma * (
                (scaled_coordinates - posterior.reference[None]) @ posterior.jacobian.T)
            actual_rgb = samples["rgb"][:, benchmark.observations.indices].flatten(1)
            linearization_rmse = ((actual_rgb - linear_rgb) / benchmark.observations.sigma).square().mean().sqrt()
            ppc = ((actual_rgb.mean(0) - benchmark.observations.rgb.flatten())
                    / benchmark.observations.sigma).square().mean()
            young, _, _, _ = model.prior.decode(coordinates)
            summary["methods"][method] = {
                "scores": scores, "coverage_curve_withheld": coverage,
                "ranking_withheld": ranking,
                "observation_linearization_rmse_in_noise_sigma": float(linearization_rmse),
                "observed_mean_render_reduced_squared_residual": float(ppc),
                "sampled_scale_counts": {str(c.scale): int((scales == c.scale).sum()) for c in posterior.components},
                "posterior_samples": len(coordinates), "failed_samples": 0,
                "physics": physics_diagnostics(samples, model.mesh, young),
            }
    summary["seconds"] = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "posterior.pt"
    temporary_checkpoint = output / "posterior.pt.tmp"
    torch.save({"reference": posterior.reference, "reference_rgb": posterior.reference_rgb,
                "jacobian": posterior.jacobian, "weights": posterior.posterior_weights,
                "means": torch.stack([c.mean for c in posterior.components]),
                "precision_cholesky": torch.stack([c.precision_cholesky for c in posterior.components]),
                "source_fingerprint": fingerprint}, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)
    # A summary is the completion marker, written only after the posterior exists.
    write_json(output / "summary.json", summary)
    return summary


def aggregate_cases(cases):
    """Each group has one replicate per seed, never one replicate per pixel/frame."""
    groups = {}
    for case in cases:
        for method, record in case["methods"].items():
            key = (case["condition"], case["schedule"], method)
            groups.setdefault(key, []).append((case, record))
    records = []
    for (condition, schedule, method), group in sorted(groups.items()):
        seeds = [case["seed"] for case, _ in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Repeated seed within one condition/schedule/method group.")
        metrics = {}
        for key in ["position_rmse_m", "position_crps_m", "trajectory_energy_score_m",
                    "component_90_coverage", "component_90_mean_width_m", "rgb_psnr_db", "novel_view_psnr_db"]:
            values = np.array([r["scores"]["withheld"][key] for _, r in group])
            metrics[key] = {"mean": float(values.mean()),
                            "seed_std": float(values.std(ddof=1)) if len(values) > 1 else None}
        metrics["normalized_ause"] = {"mean": float(np.mean([r["ranking_withheld"]["normalized_ause"] for _, r in group]))}
        linear_error = np.array([r["observation_linearization_rmse_in_noise_sigma"] for _, r in group])
        metrics["linearization_rmse_sigma"] = {"mean": float(linear_error.mean()),
            "seed_std": float(linear_error.std(ddof=1)) if len(linear_error) > 1 else None}
        records.append({"condition": condition, "schedule": schedule, "method": method,
                        "independent_realizations": len(group), "metrics": metrics,
                        "mean_probability_zero_noise": float(np.mean([
                            c["scale_posterior"]["probability_zero_process_noise"] for c, _ in group])),
                        "mean_scale": float(np.mean([c["scale_posterior"]["posterior_mean_scale"] for c, _ in group]))})
    return records


def run_study(plan, output, resume=False):
    from .cli import source_fingerprint
    from .study_report import make_study_report
    validate_plan(plan)
    plan = normalized(plan)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    manifest = {"plan": plan, "source_fingerprint_sha256": fingerprint}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Study exists. Use --resume with matching source/plan or choose a new output.")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Study source or plan changed; use a new output directory.")
    elif resume:
        raise FileNotFoundError("There is no study manifest to resume.")
    write_json(manifest_path, manifest)
    config_for_calibration = Config(**dict(plan["base"], observed_frames=next(iter(plan["schedules"].values()))))
    torch.set_num_threads(config_for_calibration.threads)
    print("Calibrating diffusion once, independently of all study observations...", flush=True)
    diffusion, calibration = train_diffusion(config_for_calibration.diffusion_seed,
                                            config_for_calibration.diffusion_steps)
    cases = []
    count = len(plan["conditions"]) * len(plan["schedules"]) * len(plan["seeds"])
    for condition in plan["conditions"]:
        for schedule, observed_frames in plan["schedules"].items():
            for seed in plan["seeds"]:
                if source_fingerprint() != fingerprint:
                    raise RuntimeError("Source changed during study; stop to avoid mixed-provenance results.")
                label = f"{condition}__{schedule}__{seed}"
                print(f"[{len(cases)+1}/{count}] {label}", flush=True)
                config = Config(**dict(plan["base"], seed=seed, observed_frames=observed_frames,
                                       truth_stochastic=condition == "stochastic",
                                       truth_external_pulse=condition == "unmodeled_pulse"))
                case_path = output / "cases" / label
                if resume and (case_path / "summary.json").exists():
                    case = json.loads((case_path / "summary.json").read_text())
                    if case["source_fingerprint_sha256"] != fingerprint or case["config"] != normalized(asdict(config)):
                        raise ValueError("A completed case has mismatched provenance.")
                else:
                    case = run_case(config, diffusion, plan["scales"], plan["spike_probability"],
                                    plan["slab_std"], case_path, fingerprint)
                case.update({"condition": condition, "schedule": schedule, "seed": seed})
                if condition != "unmodeled_pulse":
                    target_noise = float(condition == "stochastic")
                    probability = 1 - case["scale_posterior"]["probability_zero_process_noise"]
                    case["process_presence_brier_evaluation_only"] = (probability - target_noise) ** 2
                write_json(case_path / "summary.json", case)
                cases.append(case)
                s = case["scale_posterior"]
                print(f"  P(no noise)={s['probability_zero_process_noise']:.3f}, "
                      f"mean scale={s['posterior_mean_scale']:.3f}, {case['seconds']:.1f}s", flush=True)
                write_json(output / "progress.json", {**manifest, "completed": len(cases), "total": count})
    summary = {**manifest, "scope": "one known geometry, independent forcing/image-noise realizations; not cross-scene calibration",
               "inference": "three controls share one local observation linearization; adaptive scale is marginalized, not joint-MAP optimized",
               "diffusion_calibration": calibration,
               "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "numpy": np.__version__, "platform": platform.platform()},
               "case_count": len(cases), "cases": cases, "groups": aggregate_cases(cases)}
    write_json(output / "summary.json", summary)
    make_study_report(output, summary)
    print(f"Study report: {output / 'report.html'}", flush=True)
    return summary
