"""Recover a local joint-scene proposal from compact published solver states.

Recomputes one covariance, without re-optimizing or reading evaluation truth.
Use the recorded numerical source; large raw precision factors need not live
in Git. The result is still a local Gaussian approximation, not calibration.
"""
import argparse
import json
from pathlib import Path
import torch
from physgauss.cli import source_fingerprint
from physgauss.neural_marginal_study import calibrate, construct
from physgauss.accurate_study import digest_tensor
from physgauss.benchmark import Observations


def recover(study, case, output):
    study, output = Path(study).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError("Choose an unused checkpoint filename.")
    manifest = json.loads((study/"manifest.json").read_text())
    if manifest["source_fingerprint_sha256"] != source_fingerprint():
        raise ValueError("Use the numerical implementation recorded by the study.")
    allowed = {p.name for p in (study/"cases").iterdir() if p.is_dir()}
    if case not in allowed: raise ValueError("Choose an exact published case name.")
    folder = study/"cases"/case
    record = json.loads((folder/"summary.json").read_text())
    plan = manifest["plan"]; torch.set_num_threads(plan["threads"])
    network, calibration = calibrate(plan, study/"calibration.json")
    model = construct(plan, network, calibration, record["mode"])
    data = json.loads((study/"datasets"/f"{record['condition']}__{record['seed']}"/"inputs.json").read_text())
    rgb = model.times.new_tensor(data["observed_rgb"])
    if digest_tensor(rgb) != record["observations_sha256"]:
        raise ValueError("Observation hash differs.")
    observation = Observations(model.view_frames, rgb, .015)
    if (folder/"solver_state.json").exists():
        point = json.loads((folder/"solver_state.json").read_text())
        mean = model.times.new_tensor(point["coordinates"])
    else:
        mean = torch.load(folder/"fit.pt", weights_only=True)["mean"]
    precision, details = model.fisher_precision(mean, observation)
    chol = torch.linalg.cholesky(precision)
    state = mean[None].requires_grad_()
    value = model.objective(state, observation).sum()
    gradient, = torch.autograd.grad(value, state)
    decrement = float((gradient@torch.cholesky_solve(gradient.T, chol)).clamp_min(0).sqrt())
    if abs(float(value.detach())-record["fit"]["marginal_objective"]) > 1e-8:
        raise ValueError("Saved normalized target failed reproduction.")
    if abs(decrement-record["fit"]["marginal_gaussian_metric_decrement"]) > 1e-6:
        raise ValueError("Local precision/stationarity failed reproduction.")
    with torch.no_grad():
        conditional = model.elimination_at(mean[None]).conditional_mean(model.residual(mean[None]))[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean.detach(), "precision_cholesky": chol.detach(), "conditional_force_mean": conditional,
                "record": record["fit"], "recovery": {"case": case, "source_fingerprint_sha256": source_fingerprint(),
                    "objective_reproduced": True, "decrement_reproduced": decrement, **details}}, output)
    print("Recovered local checkpoint:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); recover(args.study, args.case, args.output)
