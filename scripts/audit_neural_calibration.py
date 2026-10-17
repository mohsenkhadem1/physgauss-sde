"""Evaluation-only audit of frozen diffusion on published true frame states.

No calibration, inference, or primary score is updated. Tests whether a network
validated on broad independent features behaves adaptively on this benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from physgauss.cli import source_fingerprint
from physgauss.neural_marginal_study import calibrate, construct
from physgauss.physics import edge_features
from physgauss.stochastic import calibration_law
from physgauss.study import write_json


@torch.no_grad()
def audit(study, output):
    study, output = Path(study).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError("Choose an unused audit directory.")
    manifest = json.loads((study/"manifest.json").read_text())
    if manifest["source_fingerprint_sha256"] != source_fingerprint():
        raise ValueError("Numerical source differs from the publication.")
    plan = manifest["plan"]; torch.set_num_threads(plan["threads"])
    network, calibration = calibrate(plan, study/"calibration.json")
    if hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest() != manifest["calibration_sha256_canonical_json"]:
        raise ValueError("Calibrated weights/statistics differ from the study.")
    model = construct(plan, network, calibration, "neural")
    rows, all_true, all_pred, source_hashes = [], [], [], {}
    for condition in ("stochastic", "pulse_and_stochastic"):
        for seed in plan["seeds"]:
            label = f"{condition}__{seed}"
            path = study/"datasets"/label/"reference.json"
            data = json.loads(path.read_text())
            x, velocity = [model.times.new_tensor(data[k])[None] for k in ("positions_m", "velocities_m_per_s")]
            features = edge_features(x, velocity, model.mesh)[0]
            truth = calibration_law(features); predicted = network(features)
            constant = calibration["constant_amplitude"]
            outside = ((features[..., 0] < -.25)|(features[..., 0] > .45)
                       |(features[..., 1] < -1.5)|(features[..., 1] > 1.5))
            rows.append({"condition": condition, "seed": seed, "frame_edge_states": truth.numel(),
                "analytic_true_amplitude_mean": float(truth.mean()), "neural_amplitude_mean": float(predicted.mean()),
                "analytic_true_amplitude_cv": float(truth.std()/truth.mean()),
                "neural_amplitude_cv": float(predicted.std()/predicted.mean()),
                "neural_relative_sigma_rmse": float((predicted/truth-1).square().mean().sqrt()),
                "constant_relative_sigma_rmse": float((constant/truth-1).square().mean().sqrt()),
                "out_of_training_feature_range_fraction": float(outside.double().mean())})
            all_true.append(truth.flatten()); all_pred.append(predicted.flatten())
            source_hashes[label] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    actual, predicted = torch.cat(all_true), torch.cat(all_pred)
    output.mkdir(parents=True)
    write_json(output/"summary.json", {"source_fingerprint_sha256": source_fingerprint(),
        "calibration_sha256_canonical_json": manifest["calibration_sha256_canonical_json"], "cases": rows,
        "pooled_neural_relative_sigma_rmse": float((predicted/actual-1).square().mean().sqrt()),
        "pooled_constant_relative_sigma_rmse": float((calibration["constant_amplitude"]/actual-1).square().mean().sqrt()),
        "reference_files_sha256_normalized_lf": source_hashes,
        "scope": "secondary post-fit evaluation on known true states at 21 saved frames, including base edges; correlated samples; no training/tuning or primary score changes; oracle noise amplitude is known only in this synthetic audit"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(actual.numpy(), predicted.numpy(), s=8, alpha=.25, color="#007c91")
    bounds = [min(float(actual.min()), float(predicted.min())), max(float(actual.max()), float(predicted.max()))]
    axes[0].plot(bounds, bounds, "--", color="black", label="Perfect amplitude calibration")
    axes[0].axhline(calibration["constant_amplitude"], color="#c9782a", ls=":", label="Constant control")
    axes[0].set(xlabel="Analytic true edge amplitude", ylabel="Frozen neural prediction", title="True frame-state calibration audit")
    axes[0].legend(fontsize=8)
    x = range(len(rows))
    axes[1].plot(x, [100*r["neural_relative_sigma_rmse"] for r in rows], "o-", color="#007c91", label="On true benchmark states")
    axes[1].axhline(100*calibration["constant_calibration"]["neural_relative_sigma_rmse_vs_analytic_law"], color="gray", ls="--", label="Separate broad calibration features")
    axes[1].set_xticks(list(x), [f"{'S' if r['condition']=='stochastic' else 'P+S'} / {r['seed']}" for r in rows], rotation=25, fontsize=8)
    axes[1].set(ylabel="Relative amplitude RMSE [%]", title="Calibration distribution matters")
    axes[1].legend(fontsize=8)
    fig.suptitle("Evaluation only: no network retraining, inference changes or result selection")
    fig.tight_layout(); fig.savefig(output/"calibration.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({"pooled_neural_relative_sigma_rmse": float((predicted/actual-1).square().mean().sqrt()),
                      "per_case": rows}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); audit(args.study, args.output)
