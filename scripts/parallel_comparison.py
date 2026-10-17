"""Partition all 36 matched benchmark fits using the fixed shared initializer."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import torch
from physgauss.accurate_study import digest_tensor, point_and_interval_scores
from physgauss.calibrated_study import copy_paired_inputs
from physgauss.cli import source_fingerprint
from physgauss.likelihood_audit import _environment
from physgauss.marginal_inference import TrajectoryFitConfig
from physgauss.marginal_study import audit_gaussian
from physgauss.neural_marginal_study import calibrate, construct, generate_data, force_results, aggregate
from physgauss.neural_marginal_inference import fit_neural_marginal
from physgauss.study import write_json


def worker(plan, output, partition, partitions):
    output = Path(output); output.mkdir(parents=True, exist_ok=True); torch.set_num_threads(plan["threads"])
    shutil.copyfile("examples/calibrated-noise-setup/calibration.json", output/"calibration.json")
    network, calibration = calibrate(plan, output/"calibration.json")
    fingerprint = source_fingerprint(); shared = construct(plan, network, calibration, "floor")
    fixed = json.loads(Path("examples/calibrated-noise-setup/initial_state.json").read_text())
    initial = shared.times.new_tensor(fixed["coordinates"])
    with torch.no_grad():
        torch.testing.assert_close(initial, shared.rollout(shared.times.new_zeros(1, 8), shared.times.new_zeros(1, 15))[0], rtol=1e-10, atol=1e-12)
    write_json(output/"initial_state.json", fixed)
    pairs = [(condition, seed) for condition in plan["conditions"] for seed in plan["seeds"]][partition::partitions]
    for condition, seed in pairs:
        label = f"{condition}__{seed}"
        copy_paired_inputs("examples/neural-marginal", output, {"conditions": [condition], "seeds": [seed]})
        data, observed = generate_data(plan, shared, condition, seed,
            "examples/simulation-view-audit/high_accuracy_reference", output/"datasets"/label)
        for mode in plan["modes"]:
            if source_fingerprint() != fingerprint: raise RuntimeError("Source changed.")
            print(f"Comparison partition {partition}: {label}/{mode}", flush=True)
            folder = output/"cases"/f"{label}__{mode}"; folder.mkdir(parents=True, exist_ok=True)
            model = construct(plan, network, calibration, mode)
            fitted = fit_neural_marginal(model, observed, initial, TrajectoryFitConfig(**plan["optimizer"]))
            scores, point, images = point_and_interval_scores(model, fitted, data)
            scores["interval_definition"] = "linear pushforward of local Gaussian mean+covariance metric; not exact posterior curvature or calibrated uncertainty"
            sample_seed = plan["sampling_seed"]+seed
            force_score, fields = force_results(model, fitted, condition, plan, sample_seed)
            audit = audit_gaussian(model, observed, fitted, plan["audit_samples"], sample_seed+100000)
            calibration_hash = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
            fields["inference"].update(source_fingerprint_sha256=fingerprint, observations_sha256=digest_tensor(observed.rgb),
                local_gaussian_audit=audit, calibration_sha256_canonical_json=calibration_hash)
            row = {"case": f"{label}__{mode}", "condition": condition, "seed": seed, "mode": mode,
                "observations_sha256": digest_tensor(observed.rgb), "initial_state_sha256": digest_tensor(initial),
                "source_fingerprint_sha256": fingerprint, "internal_noise": asdict(model.internal_config),
                "fit": fitted["record"], "scores": scores, "force_scores": force_score, "local_gaussian_audit": audit}
            write_json(folder/"point.json", point); write_json(folder/"fields.json", fields); write_json(folder/"summary.json", row)
            write_json(folder/"solver_state.json", {"coordinates": fitted["mean"].tolist(),
                "conditional_force_mean_at_MAP": fitted["conditional_force_mean"].tolist(), "record": fitted["record"]})
            torch.save(images, folder/"images.pt")
    write_json(output/f"execution_comparison_{partition}.json", {"partition": partition, "partitions": partitions, "pairs": pairs,
        "environment": _environment(), "source_fingerprint_sha256": fingerprint,
        "worker_script_sha256_normalized_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()})


def merge(plan, output):
    output = Path(output)
    records = [json.loads(p.read_text()) for p in output.glob("cases/*/summary.json")]
    expected = {f"{c}__{s}__{m}" for c in plan["conditions"] for s in plan["seeds"] for m in plan["modes"]}
    if len(records) != 36 or {r["case"] for r in records} != expected: raise ValueError("Incomplete comparison.")
    rows = {r["case"]: r for r in records}
    ordered = [rows[f"{c}__{s}__{m}"] for c in plan["conditions"] for s in plan["seeds"] for m in plan["modes"]]
    executions = [json.loads(p.read_text()) for p in sorted(output.glob("execution_comparison_*.json"))]
    fingerprint = source_fingerprint()
    if any(r["source_fingerprint_sha256"] != fingerprint for r in records+executions): raise ValueError("Mixed source.")
    calibration = json.loads((output/"calibration.json").read_text())
    root = Path("examples/simulation-view-audit/high_accuracy_reference")
    manifest = {"plan": plan, "source_fingerprint_sha256": fingerprint, "environment": executions[0]["environment"],
        "calibration_sha256_canonical_json": hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
        "deterministic_reference_hashes": {name: hashlib.sha256((root/f"{name}.json").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in ("no_load", "external_pulse")}, "execution_partition_environments": executions}
    write_json(output/"manifest.json", manifest)
    write_json(output/"summary.json", {**manifest, "cases": ordered, "aggregates": aggregate(ordered),
        "scope": "36 matched fits, original four conditions and three paired seeds; same six original observed images",
        "models": "all exactly marginalize external force; floor/constant/neural, fixed same original spatial basis and stable volume emission",
        "qualification": "independent CPU partitions, shared serialized prior initialization, no outcome-based selection"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/calibrated_noise_frozen_basis_study.json"))
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--partitions", type=int, default=4); parser.add_argument("--merge", action="store_true")
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    if args.merge: merge(plan, args.output)
    else: worker(plan, args.output, args.partition, args.partitions)
