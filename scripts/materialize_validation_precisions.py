"""Rebuild worker-local Gaussian factors from exact portable states and RGB.

No fitting, truth or withheld images used. Cross-platform floating-point
agreement is checked with explicit tolerances; source and weights must match.
"""
import argparse
import json
from pathlib import Path
import torch
from physgauss.cli import source_fingerprint
from physgauss.neural_marginal_study import calibrate, construct
from physgauss.benchmark import Observations


def materialize(root):
    root = Path(root); manifest = json.loads((root/"manifest.json").read_text())
    if manifest["source_fingerprint_sha256"] != source_fingerprint(): raise ValueError("Source differs.")
    plan = manifest["plan"]; torch.set_num_threads(plan["threads"])
    network, calibration = calibrate(plan, root/"calibration.json")
    for case in sorted((root/"cases").iterdir()):
        if (case/"fit.pt").exists(): continue
        row = json.loads((case/"summary.json").read_text())
        state = json.loads((case/"solver_state.json").read_text())
        model = construct(plan, network, calibration, row["mode"])
        inputs = json.loads((root/"datasets"/f"{row['condition']}__{row['seed']}"/"inputs.json").read_text())
        observation = Observations(model.view_frames, model.times.new_tensor(inputs["observed_rgb"]), .015)
        mean = model.times.new_tensor(state["coordinates"])
        precision, details = model.fisher_precision(mean, observation)
        chol = torch.linalg.cholesky(precision)
        z = mean[None].requires_grad_(); value = model.objective(z, observation).sum()
        gradient, = torch.autograd.grad(value, z)
        decrement = float((gradient@torch.cholesky_solve(gradient.T, chol)).clamp_min(0).sqrt())
        if abs(float(value.detach())-row["fit"]["marginal_objective"]) > 1e-7 or abs(decrement-row["fit"]["marginal_gaussian_metric_decrement"]) > 1e-5:
            raise ValueError("Rebuilt normalized target/precision failed.")
        torch.save({"mean": mean, "precision_cholesky": chol, "conditional_force_mean": model.times.new_tensor(state["conditional_force_mean_at_MAP"]),
            "record": row["fit"], "recovery": {"no_optimization": True, "objective": float(value.detach()), "decrement": decrement,
                "runtime_environment": {"torch": str(torch.__version__)}, **details}}, case/"fit.pt")
        print("Rebuilt", case.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args(); materialize(args.study)
