"""Line-searched marginal inference with covariance-aware Gaussian metric."""
import math
import time
import torch
from .marginal_inference import TrajectoryFitConfig, trajectory_gaussian_draws
from .physics import InvalidDeformation


def fit_neural_marginal(model, observations, initial, settings=TrajectoryFitConfig(), progress=print):
    settings.validate()
    began = time.perf_counter()
    mean = initial.detach().clone(); history = []; reason = "iteration_budget"
    for iteration in range(settings.iterations):
        state = mean[None].requires_grad_()
        objective = model.objective(state, observations).sum()
        gradient, = torch.autograd.grad(objective, state)
        gradient = gradient[0].detach()
        precision, details = model.fisher_precision(mean, observations)
        chol = torch.linalg.cholesky(precision)
        direction = -torch.cholesky_solve(gradient[:, None], chol)[:, 0]
        decrement = float((-gradient@direction).clamp_min(0).sqrt())
        current = float(objective.detach())
        item = {"iteration": iteration, "objective": current, "step_decrement": decrement,
                "gradient_norm": float(gradient.norm()), **details}
        history.append(item)
        if progress:
            progress(f"  {iteration+1}/{settings.iterations}: objective={current:.4f}, decrement={decrement:.4g}")
        if decrement < settings.decrement_tolerance:
            reason = "step_decrement"; break
        accepted = False
        for exponent in range(settings.backtracks):
            scale = .5**exponent
            candidate = mean+scale*direction
            try:
                with torch.no_grad():
                    value = float(model.objective(candidate[None], observations)[0])
            except (InvalidDeformation, ValueError):
                continue
            if math.isfinite(value) and value <= current+1e-4*scale*float(gradient@direction):
                mean = candidate.detach(); item["accepted_scale"] = scale
                accepted = True; break
        if not accepted:
            reason = "line_search_failed"; break
    state = mean[None].requires_grad_()
    objective = model.objective(state, observations).sum()
    gradient, = torch.autograd.grad(objective, state)
    gradient = gradient[0].detach()
    precision, details = model.fisher_precision(mean, observations)
    chol = torch.linalg.cholesky(precision)
    decrement = float((gradient@torch.cholesky_solve(gradient[:, None], chol)[:, 0]).clamp_min(0).sqrt())
    with torch.no_grad():
        elimination = model.elimination_at(mean[None])
        residual = model.residual(mean[None])
        conditional = elimination.conditional_mean(residual)
        offset = .5*model.force_dimension*math.log(2*math.pi)-elimination.cholesky[0].diagonal().log().sum()
        error = float((model.objective(mean[None], observations, conditional)-objective.detach()-offset).abs().max())
    return {"mean": mean, "precision_cholesky": chol, "conditional_force_mean": conditional[0],
        "record": {"method": "marginalized", "internal_mode": model.internal_config.mode,
            "state_dimension": model.state_dimension, "marginal_objective": float(objective.detach()),
            "marginal_gradient_norm": float(gradient.norm()), "marginal_gaussian_metric_decrement": decrement,
            "stationarity_screen_pass": decrement < settings.decrement_tolerance, "stop_reason": reason,
            "iterations": len(history), "history": history,
            "profile_marginal_identity_abs_error": error,
            "seconds_including_final_covariance": time.perf_counter()-began,
            "metric": "Gaussian residual mean and covariance information plus prior; exact centered-drag normalizer curvature",
            "uncertainty": "local Gaussian using this metric, not exact posterior Hessian or calibrated nonlinear uncertainty",
            **details}}


@torch.no_grad()
def force_mixture(model, fitted, count, seed):
    draws = trajectory_gaussian_draws(fitted, count, seed)
    valid, invalid = [], []
    for i, z in enumerate(draws):
        try:
            model.residual(z[None]); valid.append(i)
        except (InvalidDeformation, ValueError):
            invalid.append(i)
    if len(valid) < 4:
        raise RuntimeError("Too few valid local trajectories; never silently resample.")
    means, conditional_covariances, coefficients = [], [], []
    generator = torch.Generator().manual_seed(seed+1)
    for batch in draws[valid].split(8):
        elimination = model.elimination_at(batch)
        mean = elimination.conditional_mean(model.residual(batch))
        eps = torch.randn(mean.shape, dtype=mean.dtype, generator=generator)
        sample = mean+torch.linalg.solve_triangular(elimination.cholesky.transpose(-1, -2), eps[..., None], upper=True).squeeze(-1)
        means.append(mean); conditional_covariances.append(elimination.conditional_covariance); coefficients.append(sample)
    means, conditional_covariances, coefficients = [torch.cat(v) for v in (means, conditional_covariances, coefficients)]
    # Both components vary with the trajectory. Averaging S(z) is essential.
    conditional = conditional_covariances.mean(0)
    between = torch.cov(means.T)
    return {"mean": means.mean(0), "covariance": conditional+between,
        "conditional_covariance": conditional, "between_covariance": between,
        "coefficient_draws": coefficients, "invalid_indices": invalid, "valid_samples": len(valid),
        "conditional_covariance_trace_min_max": [float(conditional_covariances.diagonal(dim1=-2, dim2=-1).sum(-1).min()),
                                                  float(conditional_covariances.diagonal(dim1=-2, dim2=-1).sum(-1).max())]}
