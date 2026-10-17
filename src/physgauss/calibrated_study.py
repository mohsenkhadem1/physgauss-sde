"""Frozen operating-region calibration and independent sparse-view validation.

The old 12 benchmark datasets are reused exactly for paired model comparison.
New trajectory seeds are reserved for interval calibration and final evaluation.
Only RGB enters each reconstruction. Truth enters scores AFTER fitting.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from .accurate_study import digest_tensor, point_and_interval_scores
from .calibrated_diffusion import amplitude_metrics
from .likelihood_audit import _environment
from .marginal_inference import TrajectoryFitConfig
from .marginal_study import audit_gaussian
from .neural_marginal_inference import fit_neural_marginal
from .neural_marginal_study import (default_plan as base_plan, calibrate, construct,
    generate_data, run_study, force_results, validate_plan)
from .physics import edge_features
from .study import write_json
from .uncertainty_validation import trajectory_score, apply_conformal


def default_plan(quick=False):
    plan = base_plan(quick)
    if not quick: plan["optimizer"]["iterations"] = 40
    plan.update(calibration_method="trajectory_innovations_v2", operating_calibration={
        "training_seeds": [62001, 62002] if quick else list(range(62001, 62013)),
        "validation_seeds": [63001] if quick else list(range(63001, 63005)),
        "substeps": 8 if quick else 48, "innovation_seed": 72001,
        "validation_innovation_seed": 74001, "network_seed": 73001,
        "replicates": 16 if quick else 128, "steps": 30 if quick else 2500, "learning_rate": .01},
        interval_calibration_seeds=[81001, 81002] if quick else list(range(81001, 81020)),
        interval_test_seeds=[82001, 82002] if quick else list(range(82001, 82021)),
        interval_modes=["constant", "neural"], interval_alpha=.5 if quick else .1,
        posterior_audit_samples=16 if quick else 512)
    return plan


def check_splits(plan):
    groups = [set(plan["seeds"]), set(plan["operating_calibration"]["training_seeds"]),
        set(plan["operating_calibration"]["validation_seeds"]),
        set(plan["interval_calibration_seeds"]), set(plan["interval_test_seeds"])]
    for i, group in enumerate(groups):
        if not group or any(group & other for other in groups[:i]):
            raise ValueError("All calibration, validation and benchmark trajectory seeds must be disjoint.")
    for key in ("training_seeds", "validation_seeds"):
        values = plan["operating_calibration"][key]
        if len(values) != len(set(values)): raise ValueError("Repeated network calibration trajectory seed.")
    for key in ("interval_calibration_seeds", "interval_test_seeds"):
        if len(plan[key]) != len(set(plan[key])): raise ValueError("Repeated trajectory seed.")


def copy_paired_inputs(source, destination, plan):
    """Portable exact inputs or raw checkpoints; no optimization output imported."""
    import shutil
    source, destination = Path(source), Path(destination)
    for condition in plan["conditions"]:
        for seed in plan["seeds"]:
            label = f"{condition}__{seed}"
            old = source/"datasets"/label; new = destination/"datasets"/label
            if not old.exists(): raise FileNotFoundError(old)
            new.mkdir(parents=True, exist_ok=True)
            for filename in ("inputs.json", "reference.pt"):
                if (old/filename).exists():
                    if (new/filename).exists():
                        if (old/filename).read_bytes() != (new/filename).read_bytes():
                            raise ValueError("Paired inputs changed.")
                    else: shutil.copyfile(old/filename, new/filename)
            if not (new/"reference.pt").exists():
                reference = json.loads((old/"reference.json").read_text())
                torch.save({"positions": torch.tensor(reference["positions_m"], dtype=torch.float64),
                    "velocities": torch.tensor(reference["velocities_m_per_s"], dtype=torch.float64),
                    "audit": reference["audit"]}, new/"reference.pt")


def case_condition(seed):
    # Independent predeclared Bernoulli experiment condition, not chosen by result.
    return ("pulse_and_stochastic" if torch.rand((), generator=torch.Generator().manual_seed(seed+800000)) < .5
            else "stochastic")


@torch.no_grad()
def evaluation_record(model, fitted, data, condition, seed, mode):
    position = model.trajectory(fitted["mean"][None]).x[0]
    mapping = model.position_map.flatten(0, 1)
    transformed = torch.linalg.solve_triangular(fitted["precision_cholesky"], mapping.T, upper=False)
    sd = transformed.square().sum(0).sqrt().reshape(21, -1, 3)
    error = position[:, model.free]-data["x"][:, model.free]
    gap = (model.times > .3+1e-8)&(model.times < .7-1e-8)
    return {"seed": seed, "condition": condition, "mode": mode,
        "gap_max_standardized_error": trajectory_score(error[gap], sd[gap]),
        "raw_component_90_coverage": float((error[gap].abs() <= 1.6448536269514722*sd[gap]).double().mean()),
        "gap_mean_sd_m": float(sd[gap].mean()), "gap_rmse_m": float(error[gap].square().mean().sqrt()),
        "gap_absolute_errors_m": error[gap].abs().tolist(), "gap_standard_deviations_m": sd[gap].tolist(),
        "positions_m": position.tolist(), "stationary": fitted["record"]["stationarity_screen_pass"]}


def run_validation(plan, output, reference_root, resume=False):
    from .cli import source_fingerprint
    validate_plan(plan); check_splits(plan); torch.set_num_threads(plan["threads"])
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    network, calibration = calibrate(plan, output/"calibration.json")
    fingerprint = source_fingerprint()
    manifest = {"plan": plan, "source_fingerprint_sha256": fingerprint, "environment": _environment(),
        "calibration_sha256": hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
        "condition_assignment": "independent Bernoulli(0.5), generator seed trajectory_seed+800000",
        "target": "entire 7-frame missing-motion gap, all 4 free nodes and xyz; unit of replication is whole independent trajectory"}
    path = output/"manifest.json"
    if path.exists():
        if not resume or json.loads(path.read_text()) != manifest:
            raise ValueError("Validation exists or source/settings changed.")
    write_json(path, manifest)
    shared = construct(plan, network, calibration, "floor")
    with torch.no_grad(): initial = shared.rollout(shared.times.new_zeros(1, 8), shared.times.new_zeros(1, 15))[0]
    records = []
    for split, seeds in (("calibration", plan["interval_calibration_seeds"]), ("test", plan["interval_test_seeds"])):
        # Complete and freeze calibration quantiles BEFORE any test cases run.
        if split == "test":
            frozen = []
            from .uncertainty_validation import conformal_quantile
            for mode in plan["interval_modes"]:
                scores = [r["gap_max_standardized_error"] for r in records if r["mode"] == mode]
                frozen.append({"mode": mode, "sd_multiplier": conformal_quantile(scores, plan["interval_alpha"]),
                               "scores": scores})
            write_json(output/"frozen_intervals.json", {"calibration_seeds": plan["interval_calibration_seeds"],
                "alpha": plan["interval_alpha"], "quantiles": frozen,
                "qualification": "frozen before held-out test fitting; no interval or model tuning on test results"})
        for seed in seeds:
            if source_fingerprint() != fingerprint: raise RuntimeError("Source changed during validation.")
            condition = case_condition(seed); label = f"{condition}__{seed}"
            data, observed = generate_data(plan, shared, condition, seed, reference_root, output/"datasets"/label)
            for mode in plan["interval_modes"]:
                destination = output/"cases"/f"{label}__{mode}"; destination.mkdir(parents=True, exist_ok=True)
                record_file = destination/"summary.json"
                if resume and record_file.exists(): records.append(json.loads(record_file.read_text())); continue
                print(f"Interval {split}: {label}__{mode}", flush=True)
                model = construct(plan, network, calibration, mode)
                checkpoint = destination/"fit.pt"
                if resume and checkpoint.exists(): fitted = torch.load(checkpoint, weights_only=True)
                else:
                    fitted = fit_neural_marginal(model, observed, initial, TrajectoryFitConfig(**plan["optimizer"]),
                                                progress=lambda s: print(s, flush=True))
                    torch.save(fitted, checkpoint)
                record = evaluation_record(model, fitted, data, condition, seed, mode)
                record.update(split=split, observations_sha256=digest_tensor(observed.rgb),
                    initial_state_sha256=digest_tensor(initial), fit=fitted["record"],
                    local_gaussian_audit=audit_gaussian(model, observed, fitted, plan["audit_samples"], seed+900000))
                if mode == "neural":
                    features = edge_features(data["reference"]["positions"][None], data["reference"]["velocities"][None], model.mesh)[0].reshape(-1, 2)
                    record["evaluation_only_amplitude_calibration"] = amplitude_metrics(network, features, seed+950000)
                # Force uncertainty remains explicitly labelled approximate.
                force_score, fields = force_results(model, fitted, condition, plan, seed+960000)
                record["local_force_scores"] = force_score
                write_json(destination/"fields.json", fields)
                write_json(record_file, record); records.append(record)
                print(f"  gap={1000*record['gap_rmse_m']:.3f} mm; max standardized={record['gap_max_standardized_error']:.3f}; "
                      f"stationary={record['stationary']}", flush=True)
    result = {**manifest, "cases": records, "interval_validation": apply_conformal(records,
        plan["interval_calibration_seeds"], plan["interval_test_seeds"], plan["interval_alpha"]),
        "qualification": "frequentist trajectory prediction calibration on a fixed synthetic scene/mixture; local nonlinear posterior and force uncertainty separately audited, not established by conformal coverage"}
    write_json(output/"summary.json", result)
    return result


def run_calibrated(plan, output, reference_root, paired_source=None, resume=False, stage="all"):
    check_splits(plan); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if stage in ("all", "comparison"):
        if paired_source is not None: copy_paired_inputs(paired_source, output/"comparison", plan)
        run_study(plan, output/"comparison", reference_root, resume)
    if stage in ("all", "validation"):
        run_validation(plan, output/"validation", reference_root, resume)
