"""Partition the frozen validation protocol across reproducible CPU workers.

No change to numerical algorithms or seeds. Calibration partitions must finish
and freeze intervals before any test worker starts. Full precisions remain
worker-local; portable solver coordinates regenerate them without optimization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from physgauss.accurate_study import digest_tensor
from physgauss.calibrated_study import check_splits, case_condition, evaluation_record
from physgauss.cli import source_fingerprint
from physgauss.likelihood_audit import _environment
from physgauss.marginal_inference import TrajectoryFitConfig
from physgauss.marginal_study import audit_gaussian
from physgauss.neural_marginal_study import calibrate, construct, generate_data, force_results
from physgauss.neural_marginal_inference import fit_neural_marginal
from physgauss.calibrated_diffusion import amplitude_metrics
from physgauss.physics import edge_features
from physgauss.study import write_json
from physgauss.uncertainty_validation import conformal_quantile, apply_conformal


def worker(plan, calibration_file, output, split, partition, partitions, frozen=None):
    check_splits(plan); torch.set_num_threads(plan["threads"])
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(calibration_file, output/"calibration.json")
    network, calibration = calibrate(plan, output/"calibration.json")
    calibration_hash = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    fingerprint = source_fingerprint()
    if split == "test":
        freeze = json.loads(Path(frozen).read_text())
        if freeze["calibration_sha256"] != calibration_hash or freeze["source_fingerprint_sha256"] != fingerprint:
            raise ValueError("Frozen calibration source/weights mismatch.")
        if freeze["calibration_seeds"] != plan["interval_calibration_seeds"]: raise ValueError("Incomplete calibration.")
    shared = construct(plan, network, calibration, "floor")
    with torch.no_grad(): initial = shared.rollout(shared.times.new_zeros(1, 8), shared.times.new_zeros(1, 15))[0]
    seeds = plan["interval_calibration_seeds" if split == "calibration" else "interval_test_seeds"][partition::partitions]
    for seed in seeds:
        condition = case_condition(seed); label = f"{condition}__{seed}"
        data, observed = generate_data(plan, shared, condition, seed,
            "examples/simulation-view-audit/high_accuracy_reference", output/"datasets"/label)
        for mode in plan["interval_modes"]:
            if source_fingerprint() != fingerprint: raise RuntimeError("Source changed.")
            folder = output/"cases"/f"{label}__{mode}"; folder.mkdir(parents=True, exist_ok=True)
            print(f"Partition {split}/{partition}: {label}/{mode}", flush=True)
            model = construct(plan, network, calibration, mode)
            fitted = fit_neural_marginal(model, observed, initial, TrajectoryFitConfig(**plan["optimizer"]))
            row = evaluation_record(model, fitted, data, condition, seed, mode)
            row.update(split=split, observations_sha256=digest_tensor(observed.rgb), initial_state_sha256=digest_tensor(initial),
                fit=fitted["record"], local_gaussian_audit=audit_gaussian(model, observed, fitted, plan["audit_samples"], seed+900000))
            if mode == "neural":
                with torch.no_grad(): features = edge_features(data["reference"]["positions"][None], data["reference"]["velocities"][None], model.mesh)[0].reshape(-1, 2)
                row["evaluation_only_amplitude_calibration"] = amplitude_metrics(network, features, seed+950000)
            force_score, fields = force_results(model, fitted, condition, plan, seed+960000)
            row["local_force_scores"] = force_score
            write_json(folder/"fields.json", fields); write_json(folder/"summary.json", row)
            write_json(folder/"solver_state.json", {"coordinates": fitted["mean"].tolist(),
                "conditional_force_mean_at_MAP": fitted["conditional_force_mean"].tolist(), "record": fitted["record"]})
    write_json(output/f"execution_{split}_{partition}.json", {"split": split, "partition": partition, "partitions": partitions,
        "seeds": seeds, "source_fingerprint_sha256": fingerprint, "calibration_sha256": calibration_hash,
        "environment": _environment(), "frozen_intervals_sha256": hashlib.sha256(Path(frozen).read_bytes()).hexdigest() if frozen else None,
        "worker_script_sha256_normalized_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()})


def merge(plan, root, calibration_file, final=False):
    root = Path(root); check_splits(plan)
    calibration = json.loads(Path(calibration_file).read_text())
    calibration_hash = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    fingerprint = source_fingerprint()
    executions = [json.loads(p.read_text()) for p in sorted(root.glob("execution_*.json"))]
    if any(r["source_fingerprint_sha256"] != fingerprint or r["calibration_sha256"] != calibration_hash for r in executions):
        raise ValueError("Mixed worker source/calibration.")
    environment = executions[0]["environment"]
    for row in executions:
        if any(row["environment"][key] != environment[key] for key in ("python", "torch", "numpy")):
            raise ValueError("Worker numerical runtimes differ.")
    records = [json.loads(p.read_text()) for p in root.glob("cases/*/summary.json")]
    seeds = plan["interval_calibration_seeds"]+(plan["interval_test_seeds"] if final else [])
    lookup = {(r["seed"], r["mode"]): r for r in records}
    expected = {(seed, mode) for seed in seeds for mode in plan["interval_modes"]}
    if len(records) != len(expected) or set(lookup) != expected: raise ValueError("Incomplete or duplicated partition.")
    ordered = [lookup[seed, mode] for seed in seeds for mode in plan["interval_modes"]]
    if not final:
        quantiles = []
        for mode in plan["interval_modes"]:
            scores = [lookup[seed, mode]["gap_max_standardized_error"] for seed in plan["interval_calibration_seeds"]]
            quantiles.append({"mode": mode, "scores": scores, "sd_multiplier": conformal_quantile(scores, plan["interval_alpha"])})
        write_json(root/"frozen_intervals.json", {"calibration_seeds": plan["interval_calibration_seeds"],
            "alpha": plan["interval_alpha"], "quantiles": quantiles, "calibration_sha256": calibration_hash,
            "source_fingerprint_sha256": fingerprint, "qualification": "frozen after all calibration workers and before any test job"})
        return
    frozen_path = root/"frozen_intervals.json"; frozen = json.loads(frozen_path.read_text())
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    if any(e["frozen_intervals_sha256"] != frozen_hash for e in executions if e["split"] == "test"):
        raise ValueError("Test workers used different frozen quantiles.")
    manifest = {"plan": plan, "source_fingerprint_sha256": fingerprint, "environment": environment,
        "calibration_sha256": calibration_hash, "condition_assignment": "independent Bernoulli(0.5), generator seed trajectory_seed+800000",
        "target": "entire 7-frame missing-motion gap, all 4 free nodes and xyz; unit of replication is whole independent trajectory",
        "execution_partition_environments": executions}
    result = {**manifest, "cases": ordered, "interval_validation": apply_conformal(ordered,
        plan["interval_calibration_seeds"], plan["interval_test_seeds"], plan["interval_alpha"]),
        "qualification": "fixed protocol executed in independent CPU partitions; split-conformal position sets, not Bayesian posterior validation"}
    write_json(root/"manifest.json", manifest); write_json(root/"summary.json", result)
    shutil.copyfile(calibration_file, root/"calibration.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/calibrated_noise_study.json"))
    parser.add_argument("--calibration", type=Path, default=Path("examples/calibrated-noise-setup/calibration.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["calibration", "test"])
    parser.add_argument("--partition", type=int, default=0); parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--frozen", type=Path); parser.add_argument("--merge", action="store_true"); parser.add_argument("--final", action="store_true")
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    if args.merge: merge(plan, args.output, args.calibration, args.final)
    else: worker(plan, args.calibration, args.output, args.split, args.partition, args.partitions, args.frozen)
