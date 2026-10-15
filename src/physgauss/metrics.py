"""Predictive scores, componentwise intervals, and mechanical diagnostics."""
import math
import torch
from .physics import material_response


def crps(samples, target):
    """Empirical-distribution CRPS, averaged over scalar targets."""
    first = (samples - target[None]).abs().mean(0)
    second = (samples[:, None] - samples[None]).abs().mean((0, 1))
    return (first - .5 * second).mean()


def energy_score(samples, target):
    """Samples [S,T,D], target [T,D]; preserves the joint node coordinates."""
    first = torch.linalg.vector_norm(samples - target[None], dim=-1).mean(0)
    second = torch.linalg.vector_norm(samples[:, None] - samples[None], dim=-1).mean((0, 1))
    return (first - .5 * second).mean()


def interval_metrics(samples, target, probability=.9):
    alpha = (1 - probability) / 2
    lo, hi = torch.quantile(samples, samples.new_tensor([alpha, 1 - alpha]), dim=0)
    return {"coverage": float(((target >= lo) & (target <= hi)).to(samples.dtype).mean()),
            "mean_width_m": float((hi - lo).mean())}


def uncertainty_ranking(samples, target):
    """Ranking-only diagnostics on scalar coordinates; not calibration tests.

    AUSE integrates normalized remaining absolute error above the oracle curve
    over retained fractions 0.05,...,1. Spearman uses average ranks for ties.
    """
    import numpy as np
    error = (samples.mean(0) - target).abs().flatten().cpu().numpy()
    uncertainty = samples.std(0, unbiased=False).flatten().cpu().numpy()
    n = len(error)
    order = np.argsort(uncertainty, kind="stable")
    oracle = np.sort(error)
    fractions = np.linspace(.05, 1., 20)
    counts = np.maximum(1, np.ceil(fractions * n).astype(int))
    risk = np.array([error[order[:k]].mean() for k in counts])
    oracle_risk = np.array([oracle[:k].mean() for k in counts])
    relative = (risk - oracle_risk) / max(float(error.mean()), 1e-12)
    ause = np.sum(.5 * (relative[:-1] + relative[1:]) * np.diff(fractions))

    def average_rank(values):
        indices = np.argsort(values, kind="stable")
        sorted_values = values[indices]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
        ends = np.r_[starts[1:], len(values)]
        ranks = np.empty(len(values), dtype=float)
        for a, b in zip(starts, ends):
            ranks[indices[a:b]] = .5 * (a + b - 1)
        return ranks

    a, b = average_rank(error), average_rank(uncertainty)
    rho = None if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
    return {"normalized_ause": float(ause), "spearman_error_uncertainty": rho,
            "retained_fractions": fractions.tolist(), "remaining_mae_m": risk.tolist(),
            "oracle_remaining_mae_m": oracle_risk.tolist()}


def evaluate(samples, benchmark, mesh):
    truth = benchmark.truth.x[0]
    free = ~mesh.pinned
    observed = torch.zeros(benchmark.config.frames, dtype=torch.bool)
    observed[benchmark.observations.indices] = True
    observed_list = benchmark.observations.indices.tolist()
    left, right = max(zip(observed_list[:-1], observed_list[1:]), key=lambda pair: pair[1] - pair[0])
    gap = torch.zeros_like(observed)
    gap[left + 1:right] = True
    result = {}
    for name, mask in [("observed", observed), ("withheld", ~observed), ("long_gap", gap)]:
        if not bool(mask.any()):
            continue
        positions = samples["x"][:, mask][:, :, free].flatten(2)
        target = truth[mask][:, free].flatten(1)
        delta = positions.mean(0) - target
        image_mean = samples["rgb"][:, mask].mean(0)
        image_error = (image_mean - benchmark.truth_rgb[mask]).square().mean()
        novel_error = (samples["novel_rgb"][:, mask].mean(0)
                       - benchmark.truth_novel_rgb[mask]).square().mean()
        intervals = interval_metrics(positions, target)
        result[name] = {
            "frames": int(mask.sum()),
            "position_rmse_m": float(delta.square().mean().sqrt()),
            "position_crps_m": float(crps(positions, target)),
            "trajectory_energy_score_m": float(energy_score(positions, target)),
            "component_90_coverage": intervals["coverage"],
            "component_90_mean_width_m": intervals["mean_width_m"],
            "rgb_psnr_db": float(-10 * torch.log10(image_error.clamp_min(1e-15))),
            "novel_view_psnr_db": float(-10 * torch.log10(novel_error.clamp_min(1e-15))),
        }
    result["definitions"] = {
        "uncertainty": "local joint Gauss-Newton Laplace; finite posterior samples; no calibration guarantee",
        "coverage": "central componentwise intervals on free-node positions only; pinned nodes excluded",
        "energy_score": "Euclidean joint free-node position score per timestamp, averaged over timestamps",
        "novel_view": "same times, known camera path offset by 100 degrees; evaluation-only images",
        "rgb": "posterior mean render versus noise-free truth; PSNR data range 1",
        "long_gap": f"strictly between observed frame {left} and {right}",
    }
    return result


def physics_diagnostics(samples, mesh, young):
    with torch.no_grad():
        force, elastic, jac = material_response(samples["x"], mesh, young[:, None])
        net = force.sum(-2)
        torque = torch.linalg.cross(samples["x"], force).sum(-2)
        kinetic = .5 * (samples["v"].square() * mesh.mass[None, None, :, None]).sum((-2, -1))
        gravitational = .6 * (samples["x"][..., 1] * mesh.mass[None, None]).sum(-1)
        energy = kinetic + elastic + gravitational
    return {
        "fixed_total_mass_kg": float(mesh.mass.sum()),
        "minimum_det_F": float(jac.min()),
        "maximum_internal_net_force_n": float(net.abs().max()),
        "maximum_internal_net_torque_nm": float(torque.abs().max()),
        "maximum_support_position_error_m": float((samples["x"][:, :, mesh.pinned]
                                                   - mesh.rest[mesh.pinned]).abs().max()),
        "mechanical_energy_mean_j": energy.mean(0).tolist(),
        "energy_note": "Gravity potential included. Drag dissipates and stochastic forcing exchanges energy; total energy need not be constant.",
    }
