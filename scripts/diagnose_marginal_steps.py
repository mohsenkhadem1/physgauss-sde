"""Audit stalled marginal GN directions against actual renderer depth order.

This is post-fit numerical diagnosis, not parameter tuning or another fit.
"""
import argparse
import csv
import json
from pathlib import Path
import torch
from physgauss.benchmark import Config, Observations
from physgauss.camera import project
from physgauss.forces import ForceConfig
from physgauss.marginal_forces import MarginalConfig, TrajectoryForceModel
from physgauss.physics import InvalidDeformation
from physgauss.study import write_json


@torch.no_grad()
def depths(model, state, observations):
    x = model.trajectory(state[None]).x[:, observations.indices]
    nodes = x[:, :, model.mesh.cells]
    centers = torch.einsum("pk,bftkd->bftpd", model.renderer.bary, nodes).flatten(2, 3)
    _, depth, _ = project(centers, model.eye[observations.indices], model.rotation[observations.indices], model.config.image_size)
    return depth[0]


def diagnose(raw, output):
    raw, output = Path(raw), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new diagnostic directory.")
    summary = json.loads((raw/"summary.json").read_text())
    output.mkdir(parents=True)
    rows, records = [], []
    torch.set_num_threads(Config(**summary["plan"]["base"]).threads)
    for r in summary["cases"]:
        if r["method"] != "marginalized":
            continue
        pair = f"{r['condition']}__{r['seed']}"
        model = TrajectoryForceModel(Config(**r["config"]), ForceConfig(**summary["plan"]["force_prior"]),
                                     MarginalConfig(**summary["plan"]["trajectory_model"]))
        warm = torch.load(raw/"cases"/pair/"initialization.pt", weights_only=True)
        obs = Observations(warm["observed_indices"], warm["observed_rgb"], warm["observation_sigma"])
        fit = torch.load(raw/"cases"/pair/"marginalized"/"fit.pt", weights_only=True)
        state = fit["mean"].detach().requires_grad_()
        loss = model.objective(state[None], obs)[0]
        gradient, = torch.autograd.grad(loss, state)
        direction = -torch.cholesky_solve(gradient[:, None], fit["precision_cholesky"])[:, 0].detach()
        initial = float(loss.detach())
        expected_slope = float(gradient.detach()@direction)
        base_depth = depths(model, state.detach(), obs)
        base_order = base_depth.argsort(-1)
        found = []
        for exponent in range(41):
            step = .5**exponent
            point = state.detach()+step*direction
            try:
                with torch.no_grad():
                    actual = float(model.objective(point[None], obs)[0])-initial
                    candidate_order = depths(model, point, obs).argsort(-1)
                changed = int((candidate_order != base_order).sum())
                armijo = actual <= 1e-4*step*expected_slope
                rows.append({"condition": r["condition"], "seed": r["seed"], "step_exponent": exponent,
                    "step": step, "objective_change": actual, "predicted_linear_change": step*expected_slope,
                    "changed_order_entries": changed, "armijo_decrease": armijo, "valid": True})
                if armijo:
                    found.append(exponent)
            except (InvalidDeformation, ValueError):
                rows.append({"condition": r["condition"], "seed": r["seed"], "step_exponent": exponent,
                    "step": step, "objective_change": None, "predicted_linear_change": step*expected_slope,
                    "changed_order_entries": None, "armijo_decrease": False, "valid": False})
        mask = torch.triu(torch.ones(base_depth.shape[-1], base_depth.shape[-1], dtype=torch.bool), diagonal=1)
        differences = (base_depth[:, :, None]-base_depth[:, None, :]).abs()
        near_ties = int((differences[:, mask] <= 1e-12).sum())
        records.append({"condition": r["condition"], "seed": r["seed"], "fit_stop_reason": r["fit"]["stop_reason"],
            "directional_derivative": expected_slope, "minimum_Armijo_exponent_in_scan": min(found) if found else None,
            "near_tied_depth_pairs_at_1e_minus_12_m": near_ties,
            "interpretation": "depth-order changes can make a branch derivative fail to predict finite-step objective; scan alone does not prove the only cause of failure"})
    with (output/"step_scan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output/"summary.json", {"numerical_source_fingerprint_sha256": summary["source_fingerprint_sha256"],
        "scope": "post-fit step/depth-order diagnosis; no inference retuning", "cases": records})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.raw, args.output)
