"""Observation-only MAP smoothing and local Gauss-Newton Laplace inference.

The SDE latent variables are standardized Brownian increments at every solver
step. Inference conditions the *joint* scene-and-noise state on all observed
images. It is a local Gaussian approximation, not exact or multimodal sampling.
"""
from dataclasses import dataclass
import time
import torch
from torch import Tensor
from .benchmark import Observations, SceneModel


@dataclass
class Fit:
    mean: Tensor
    precision_cholesky: Tensor
    history: list
    elapsed_seconds: float
    objective: float
    gradient_norm: float
    newton_decrement: float = float("nan")


def objective(model: SceneModel, observations: Observations, z: Tensor):
    prediction = model.predict(z, observations.indices)
    residual = (prediction - observations.rgb[None]) / observations.sigma
    return .5 * residual.square().sum() + .5 * z.square().sum()


def fit_map(model: SceneModel, observations: Observations, adam_steps=100,
            lbfgs_steps=20, progress=print):
    start = time.perf_counter()
    z = torch.zeros(1, model.dimension, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=.035)
    history = []
    best_loss, best_z = float("inf"), z.detach().clone()
    for step in range(adam_steps):
        optimizer.zero_grad()
        loss = objective(model, observations, z)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite objective; no invalid inference result is saved.")
        history.append(float(loss.detach()))
        if history[-1] < best_loss:
            best_loss, best_z = history[-1], z.detach().clone()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z], max_norm=2000)
        optimizer.step()
        if progress and (step == 0 or (step + 1) % 25 == 0):
            progress(f"  MAP {step + 1}/{adam_steps}: objective={history[-1]:.3f}")
    # Start line search from the best evaluated Adam iterate, not an unchecked last update.
    with torch.no_grad():
        z.copy_(best_z)
    if lbfgs_steps:
        optimizer2 = torch.optim.LBFGS([z], lr=.7, max_iter=lbfgs_steps,
                                      line_search_fn="strong_wolfe", tolerance_grad=1e-5)

        def closure():
            optimizer2.zero_grad()
            loss = objective(model, observations, z)
            loss.backward()
            history.append(float(loss.detach()))
            return loss

        optimizer2.step(closure)
    final = objective(model, observations, z)
    grad, = torch.autograd.grad(final, z)
    elapsed = time.perf_counter() - start
    return z.detach()[0], history, elapsed, float(final.detach()), float(grad.norm())


def laplace_precision(model, observations, mean, chunk=12, delta=1e-3, progress=print):
    """Central finite-difference image Jacobian; H = J^T R^-1 J + I.

    Coordinates include Brownian increments for the SDE. The identity is the
    common standardized normal prior, not an arbitrary uncertainty temperature.
    The Gauss-Newton approximation drops likelihood residual second derivatives.
    """
    columns = []
    dimension = model.dimension
    with torch.no_grad():
        for start in range(0, dimension, chunk):
            end = min(start + chunk, dimension)
            count = end - start
            shift = torch.zeros(count, dimension, dtype=mean.dtype)
            shift[torch.arange(count), torch.arange(start, end)] = delta
            candidates = torch.cat((mean[None] + shift, mean[None] - shift))
            predicted = model.predict(candidates, observations.indices).flatten(1)
            columns.append(((predicted[:count] - predicted[count:])
                            / (2 * delta * observations.sigma)).T)
            if progress and (end == dimension or start == 0 or end % (chunk * 5) == 0):
                progress(f"  uncertainty Jacobian {end}/{dimension}")
        jac = torch.cat(columns, dim=1)
        precision = jac.T @ jac + torch.eye(dimension, dtype=mean.dtype)
        chol = torch.linalg.cholesky(precision)
    return chol


def fit(model, observations, progress=print):
    c = model.config
    mean, history, seconds, loss, grad = fit_map(model, observations, c.adam_steps,
                                               c.lbfgs_steps, progress)
    start = time.perf_counter()
    chol = laplace_precision(model, observations, mean, c.finite_difference_chunk, progress=progress)
    # A local covariance should be centered near a stationary solution. Refine
    # with the same Gauss-Newton metric and an objective-decreasing line search.
    changed = False
    for _ in range(3):
        variable = mean[None].clone().requires_grad_()
        value = objective(model, observations, variable)
        gradient, = torch.autograd.grad(value, variable)
        direction = -torch.cholesky_solve(gradient.T, chol).T[0]
        decrement = float((-gradient[0] @ direction).clamp_min(0).sqrt())
        if decrement < .02:
            break
        accepted = False
        for backtrack in range(12):
            scale = .5 ** backtrack
            candidate = mean + scale * direction
            with torch.no_grad():
                candidate_value = float(objective(model, observations, candidate[None]))
            if candidate_value < float(value.detach()) + 1e-4 * scale * float(gradient[0] @ direction):
                mean = candidate
                history.append(candidate_value)
                changed = accepted = True
                break
        if not accepted:
            break
    if changed:
        chol = laplace_precision(model, observations, mean, c.finite_difference_chunk, progress=progress)
    variable = mean[None].clone().requires_grad_()
    final = objective(model, observations, variable)
    gradient, = torch.autograd.grad(final, variable)
    decrement = float((gradient @ torch.cholesky_solve(gradient.T, chol)).clamp_min(0).sqrt().detach())
    return Fit(mean.detach(), chol, history, seconds + time.perf_counter() - start,
               float(final.detach()), float(gradient.norm()), decrement)


def sample_coordinates(fit_result: Fit, samples, seed):
    gen = torch.Generator().manual_seed(seed)
    eps = torch.randn(samples, len(fit_result.mean), generator=gen, dtype=fit_result.mean.dtype)
    perturbation = torch.linalg.solve_triangular(fit_result.precision_cholesky.T,
                                                 eps.T, upper=True).T
    return fit_result.mean[None] + perturbation


def predict_samples(model, coordinates, chunk=8):
    from .camera import orbit
    positions, velocities, images, novel_images = [], [], [], []
    with torch.no_grad():
        for z in coordinates.split(chunk):
            trajectory = model.trajectory(z)
            positions.append(trajectory.x)
            velocities.append(trajectory.v)
            images.append(model.renderer(trajectory.x, model.eye, model.rotation))
            novel_images.append(model.renderer(trajectory.x, *orbit(model.times, offset_degrees=100)))
    return {"x": torch.cat(positions), "v": torch.cat(velocities),
            "rgb": torch.cat(images), "novel_rgb": torch.cat(novel_images)}
