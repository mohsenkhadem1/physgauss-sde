"""Matched explicit, alternating and analytically collapsed trajectory fits."""
from dataclasses import dataclass
import math
import time
import torch
from .physics import InvalidDeformation


@dataclass
class TrajectoryFitConfig:
    iterations: int = 20
    decrement_tolerance: float = .05
    backtracks: int = 18

    def validate(self):
        if self.iterations < 1 or self.backtracks < 1 or not math.isfinite(self.decrement_tolerance) or self.decrement_tolerance <= 0:
            raise ValueError("Need positive iteration, backtrack and convergence settings.")
        return self


def fit_trajectory(model, observations, initial, method, settings=TrajectoryFitConfig(), progress=print):
    if method not in ("marginalized", "joint", "alternating"):
        raise ValueError("Unknown trajectory inference method.")
    settings.validate()
    began = time.perf_counter()
    mean = initial.detach().clone()
    with torch.no_grad():
        a = model.elimination.conditional_mean(model.residual(mean[None]))[0]
    history = []
    reason = "iteration_budget"
    for iteration in range(settings.iterations):
        if method == "alternating":
            with torch.no_grad():
                a = model.elimination.conditional_mean(model.residual(mean[None]))[0]
        state = mean[None].requires_grad_()
        if method == "marginalized":
            loss = model.objective(state, observations).sum()
            gradient, = torch.autograd.grad(loss, state)
            gradient = gradient[0].detach()
        else:
            coefficient = a[None].requires_grad_()
            loss = model.objective(state, observations, coefficient).sum()
            gs, ga = torch.autograd.grad(loss, (state, coefficient))
            gradient = torch.cat((gs[0], ga[0])).detach() if method == "joint" else gs[0].detach()
        marginal, explicit, cross = model.information(mean, observations)
        if method == "joint":
            precision = torch.cat((torch.cat((explicit, cross), 1),
                                   torch.cat((cross.T, model.elimination.precision), 1)), 0)
        else:
            precision = marginal if method == "marginalized" else explicit
        chol = torch.linalg.cholesky(precision)
        direction = -torch.cholesky_solve(gradient[:, None], chol)[:, 0]
        decrement = float((-gradient@direction).clamp_min(0).sqrt())
        current = float(loss.detach())
        item = {"iteration": iteration, "objective": current, "step_decrement": decrement,
                "gradient_norm": float(gradient.norm())}
        history.append(item)
        if progress:
            progress(f"  {method} {iteration+1}/{settings.iterations}: objective={current:.3f}, decrement={decrement:.4g}")
        if decrement < settings.decrement_tolerance:
            reason = "step_decrement"; break
        accepted = False
        for backtrack in range(settings.backtracks):
            scale = .5**backtrack
            candidate = mean+scale*direction[:model.state_dimension]
            candidate_a = a+scale*direction[model.state_dimension:] if method == "joint" else a
            try:
                with torch.no_grad():
                    value = float(model.objective(candidate[None], observations,
                        None if method == "marginalized" else candidate_a[None])[0])
            except (InvalidDeformation, ValueError):
                continue
            if value <= current+1e-4*scale*float(gradient@direction):
                mean, a = candidate.detach(), candidate_a.detach()
                item["accepted_scale"] = scale
                accepted = True; break
        if not accepted:
            reason = "line_search_failed"; break
    optimization_seconds = time.perf_counter()-began
    # Compare all methods in the SAME marginal density and covariance.
    state = mean[None].requires_grad_()
    final = model.objective(state, observations).sum()
    gradient, = torch.autograd.grad(final, state)
    gradient = gradient[0].detach()
    marginal, explicit, cross = model.information(mean, observations)
    chol = torch.linalg.cholesky(marginal)
    decrement = float((gradient@torch.cholesky_solve(gradient[:, None], chol)[:, 0]).clamp_min(0).sqrt())
    schur = explicit-cross@torch.cholesky_solve(cross.T, model.elimination.cholesky)
    schur_error = float((schur-marginal).abs().max()/marginal.abs().max())
    with torch.no_grad():
        residual = model.residual(mean[None])
        conditional_a = model.elimination.conditional_mean(residual)[0]
        joint_at_conditional = model.objective(mean[None], observations, conditional_a[None])[0]
        expected_offset = (.5*model.force_dimension*torch.log(mean.new_tensor(2*torch.pi))
                           -model.elimination.cholesky.diagonal().log().sum())
        identity_error = float((joint_at_conditional-final.detach()-expected_offset).abs())
        data_nll = float(.5*model.image_residual(mean[None], observations).square().sum())
    return {"mean": mean, "precision_cholesky": chol,
        "conditional_force_mean": conditional_a,
        "record": {"method": method, "state_dimension": model.state_dimension,
            "optimized_dimension": model.state_dimension+(model.force_dimension if method == "joint" else 0),
            "force_block": "analytic Gaussian conditional" if method != "joint" else "explicit optimized coefficients",
            "marginal_objective": float(final.detach()), "observed_image_nll_without_constant": data_nll,
            "marginal_gradient_norm": float(gradient.norm()), "marginal_gauss_newton_decrement": decrement,
            "stationarity_screen_pass": decrement < settings.decrement_tolerance,
            "stop_reason": reason, "iterations": len(history), "history": history,
            "schur_precision_relative_max_error": schur_error,
            "profile_marginal_identity_abs_error": identity_error,
            "optimization_seconds": optimization_seconds, "seconds_including_final_covariance": time.perf_counter()-began,
            "uncertainty": "local Gauss-Newton Gaussian over trajectory/material; exact conditional Gaussian force, not calibrated nonlinear posterior"}}


def trajectory_gaussian_draws(fitted, count, seed):
    gen = torch.Generator().manual_seed(seed)
    mean = fitted["mean"]
    eps = torch.randn(count, len(mean), dtype=mean.dtype, generator=gen)
    return mean[None]+torch.linalg.solve_triangular(fitted["precision_cholesky"].T, eps.T, upper=True).T


def recover_force_moments(elimination, residuals):
    """Total covariance = E[conditional covariance] + Cov[conditional mean]."""
    means = elimination.conditional_mean(residuals)
    if len(means) < 2:
        raise ValueError("Need at least two trajectory draws for mixture covariance.")
    between = torch.cov(means.T)
    return {"mean": means.mean(0), "covariance": elimination.conditional_covariance+between,
            "conditional_covariance": elimination.conditional_covariance, "between_covariance": between,
            "conditional_means": means}
