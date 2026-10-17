"""Trajectory-level split conformal intervals and nonlinear posterior audits.

Conformal coverage is a repeated-case prediction property under exchangeability,
not validation of a Bayesian posterior or of force/parameter uncertainty.
"""
import math
import torch


def conformal_quantile(scores, alpha=.1):
    values = torch.as_tensor(scores, dtype=torch.float64)
    if values.ndim != 1 or not len(values) or not 0 < alpha < 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("Need finite scalar scores and 0 < alpha < 1.")
    rank = math.ceil((len(values)+1)*(1-alpha))
    return float(values.sort().values[rank-1]) if rank <= len(values) else math.inf


def trajectory_score(error, standard_deviation):
    if error.shape != standard_deviation.shape or bool((standard_deviation <= 0).any()):
        raise ValueError("Matching shapes and positive uncertainty scales required.")
    return float((error.abs()/standard_deviation).max())


def wilson_interval(successes, trials, z=1.959963984540054):
    if trials < 1 or not 0 <= successes <= trials: raise ValueError("Invalid binomial counts.")
    p = successes/trials; denominator = 1+z*z/trials
    center = (p+z*z/(2*trials))/denominator
    radius = z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/denominator
    return [max(0., center-radius), min(1., center+radius)]


def apply_conformal(records, calibration_seeds, test_seeds, alpha=.1):
    if set(calibration_seeds) & set(test_seeds): raise ValueError("Calibration/test leakage.")
    result = []
    for mode in dict.fromkeys(r["mode"] for r in records):
        subset = [r for r in records if r["mode"] == mode]
        calibration = [r for r in subset if r["seed"] in calibration_seeds]
        test = [r for r in subset if r["seed"] in test_seeds]
        if len(calibration) != len(calibration_seeds) or len(test) != len(test_seeds):
            raise ValueError("Require exactly one independent trajectory per seed per mode.")
        if ({r["seed"] for r in calibration} != set(calibration_seeds)
                or {r["seed"] for r in test} != set(test_seeds)):
            raise ValueError("Duplicate or missing independent trajectory.")
        factor = conformal_quantile([r["gap_max_standardized_error"] for r in calibration], alpha)
        covered = sum(r["gap_max_standardized_error"] <= factor for r in test)
        raw = sum(r["gap_max_standardized_error"] <= 1.6448536269514722 for r in test)
        component = []
        for row in test:
            if "gap_absolute_errors_m" in row:
                error = torch.tensor(row["gap_absolute_errors_m"])
                width = torch.tensor(row["gap_standard_deviations_m"])*factor
                component.append(float((error <= width).double().mean()))
        result.append({"mode": mode, "alpha": alpha, "calibration_cases": len(calibration), "test_cases": len(test),
            "quantile_rank": math.ceil((len(calibration)+1)*(1-alpha)), "sd_multiplier": factor,
            "raw_simultaneous_covered": raw, "conformal_simultaneous_covered": covered,
            "conformal_trajectory_coverage": covered/len(test),
            "coverage_95_wilson_interval": wilson_interval(covered, len(test)),
            "raw_mean_component_coverage": sum(r["raw_component_90_coverage"] for r in test)/len(test),
            "conformal_mean_component_coverage": sum(component)/len(component) if component else None,
            "raw_mean_width_m": sum(r["gap_mean_sd_m"] for r in test)/len(test)*2*1.6448536269514722,
            "conformal_mean_width_m": sum(r["gap_mean_sd_m"] for r in test)/len(test)*2*factor,
            "mean_gap_rmse_m": sum(r["gap_rmse_m"] for r in test)/len(test),
            "scope": "simultaneous coverage of all 84 free-node xyz coordinates at 7 gap frames; one independent trajectory/image-noise draw per seed; 50:50 pulse/no-pulse stochastic mixture; fixed known scene",
            "qualification": "split-conformal prediction sets conditional on frozen fitted calibration dataset; marginal guarantee averages over calibration and new cases under exchangeability; no guarantee conditional on each video or for force/parameter posteriors"})
    return result


def finite_difference_precision(model, observations, fitted, epsilon=.01, batch_size=4):
    """Observed nonlinear curvature in well-scaled local-Gaussian coordinates.

    Central differences of exact gradients, not differences of objective values.
    A separate epsilon refinement must be reported; indefinite curvature fails
    explicitly. No eigenvalue clipping or silent Fisher substitution.
    """
    mean = fitted["mean"]; chol = fitted["precision_cholesky"]
    dimension = len(mean)
    transform = torch.linalg.solve_triangular(chol.T, torch.eye(dimension, dtype=mean.dtype), upper=True)
    columns = []
    for start in range(0, dimension, batch_size):
        shifts = epsilon*transform[:, start:start+batch_size].T
        gradients = []
        for sign in (1, -1):
            points = (mean[None]+sign*shifts).detach().requires_grad_()
            loss = model.objective(points, observations).sum()
            derivative, = torch.autograd.grad(loss, points)
            gradients.append(derivative.detach()@transform)
        columns.append(((gradients[0]-gradients[1])/(2*epsilon)).T)
    hessian = torch.cat(columns, 1)
    asymmetry = float((hessian-hessian.T).norm()/hessian.norm())
    hessian = .5*(hessian+hessian.T)
    white_chol, info = torch.linalg.cholesky_ex(hessian)
    if int(info): raise RuntimeError("Observed posterior curvature is indefinite; improve mode convergence.")
    # H_native = L H_white L^T.
    native = chol@hessian@chol.T
    result = dict(fitted); result["precision_cholesky"] = torch.linalg.cholesky(.5*(native+native.T))
    return result, {"epsilon_white": epsilon, "gradient_batch_size": batch_size,
        "relative_antisymmetry": asymmetry, "white_eigenvalue_min": float(torch.linalg.eigvalsh(hessian).min()),
        "white_eigenvalue_max": float(torch.linalg.eigvalsh(hessian).max()),
        "qualification": "finite-difference observed Hessian; still local Gaussian, not full posterior"}, hessian
