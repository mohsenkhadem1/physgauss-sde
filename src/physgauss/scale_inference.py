"""Observation-only inference over a discrete spike-and-slab noise-scale prior.

All components share ONE local image linearization. Standard normal scene and
Brownian variables are analytically marginalized for scale evidence. This is
exact for that linearized model, and approximate for the nonlinear renderer.
It avoids selecting noise scale by joint MAP, which ignores latent prior volume.
"""
from dataclasses import dataclass
import math
import torch
from torch import Tensor
from .benchmark import SceneModel
from .inference import image_jacobian


@dataclass
class LinearComponent:
    scale: float
    mean: Tensor
    precision_cholesky: Tensor
    negative_log_evidence: float


@dataclass
class ScalePosterior:
    components: list
    prior_weights: Tensor
    posterior_weights: Tensor
    reference: Tensor
    reference_rgb: Tensor
    jacobian: Tensor


def scale_prior(scales=(0., .125, .25, .5, 1., 2., 4.), spike_probability=.5, slab_std=1.):
    """Point mass at zero plus binned half-normal probability on positive scales.

    Positive grid values represent bins separated by geometric midpoints. The
    final bin contains the entire tail. This is a declared discrete prior, not
    fitted to held-out scores or a continuous-evidence integration guarantee.
    """
    if (len(scales) < 2 or scales[0] != 0 or list(scales) != sorted(set(scales))
            or any(not math.isfinite(s) or s < 0 for s in scales)):
        raise ValueError("Scale grid must be sorted, distinct, finite, and start at zero.")
    if not 0 < spike_probability < 1 or slab_std <= 0 or not math.isfinite(slab_std):
        raise ValueError("Need a nondegenerate spike probability and a positive finite slab std.")
    positive = list(scales[1:])
    boundaries = [0.] + [math.sqrt(a * b) for a, b in zip(positive[:-1], positive[1:])] + [math.inf]
    cdf = lambda a: math.erf(a / (math.sqrt(2) * slab_std))
    masses = [spike_probability] + [(1 - spike_probability) * (cdf(b) - cdf(a))
                                   for a, b in zip(boundaries[:-1], boundaries[1:])]
    result = torch.tensor(masses, dtype=torch.float64)
    if not bool((result > 0).all()):
        raise ValueError("Grid has numerically zero prior masses; reduce its range.")
    return result / result.sum()


def linear_component(jacobian, data, scene_dimensions, scale):
    """Integrate z~N(0,I), y=A_scale z+epsilon, epsilon~N(0,I).

    Evidence omits only M/2*log(2pi), common to all scale components.
    Cholesky calculations are in latent dimension, never observation dimension.
    """
    if not 0 < scene_dimensions <= jacobian.shape[1] or scale < 0 or not math.isfinite(scale):
        raise ValueError("Invalid scene dimension or scale.")
    d = jacobian.shape[1]
    multipliers = torch.ones(d, dtype=jacobian.dtype)
    multipliers[scene_dimensions:] = scale
    a = jacobian * multipliers
    precision = torch.eye(d, dtype=a.dtype) + a.T @ a
    chol = torch.linalg.cholesky(precision)
    mean = torch.cholesky_solve((a.T @ data)[:, None], chol)[:, 0]
    # Stable equivalent of y^T C^-1 y: no subtraction of large close numbers.
    residual = data - a @ mean
    logdet = 2 * chol.diagonal().log().sum()
    nle = .5 * (residual.square().sum() + mean.square().sum() + logdet)
    return LinearComponent(float(scale), mean, chol, float(nle))


def scale_posterior_from_linearization(jacobian, data, scene_dimensions,
                                     scales=(0., .125, .25, .5, 1., 2., 4.),
                                     spike_probability=.5, slab_std=1.):
    weights = scale_prior(scales, spike_probability, slab_std).to(jacobian.dtype)
    components = [linear_component(jacobian, data, scene_dimensions, s) for s in scales]
    nle = jacobian.new_tensor([c.negative_log_evidence for c in components])
    posterior = torch.softmax(weights.log() - nle, dim=0)
    return components, weights, posterior


def infer_scale_posterior(model: SceneModel, observations, scene_reference,
                         scales=(0., .125, .25, .5, 1., 2., 4.),
                         spike_probability=.5, slab_std=1., progress=print):
    if model.diffusion is None or model.process_scale != 1.:
        raise ValueError("The common linearization needs the unit-scale stochastic model.")
    reference = torch.zeros(model.dimension, dtype=torch.float64)
    reference[:8] = scene_reference
    with torch.no_grad():
        rgb = model.predict(reference[None], observations.indices)[0]
        jac = image_jacobian(model, observations, reference,
                             chunk=model.config.finite_difference_chunk, progress=progress)
        # f(z_ref)+J_scene(z_scene-z_ref)+alpha*J_noise*w, prior E[z]=0.
        data = ((observations.rgb - rgb) / observations.sigma).flatten() + jac @ reference
        components, prior, posterior = scale_posterior_from_linearization(
            jac, data, 8, scales, spike_probability, slab_std)
    return ScalePosterior(components, prior, posterior, reference, rgb, jac)


def draw_coordinates(posterior: ScalePosterior, samples, seed, fixed_scale=None):
    generator = torch.Generator().manual_seed(seed)
    if fixed_scale is None:
        indices = torch.multinomial(posterior.posterior_weights, samples, replacement=True,
                                    generator=generator)
    else:
        matches = [i for i, c in enumerate(posterior.components) if c.scale == fixed_scale]
        if len(matches) != 1:
            raise ValueError("Requested fixed scale is absent from the prior grid.")
        indices = torch.full((samples,), matches[0], dtype=torch.long)
    eps = torch.randn(samples, len(posterior.reference), dtype=posterior.reference.dtype,
                      generator=generator)
    coordinates = torch.empty_like(eps)
    scales = torch.empty(samples, dtype=eps.dtype)
    for i, component in enumerate(posterior.components):
        mask = indices == i
        if not bool(mask.any()):
            continue
        delta = torch.linalg.solve_triangular(component.precision_cholesky.T,
                                               eps[mask].T, upper=True).T
        coordinates[mask] = component.mean[None] + delta
        scales[mask] = component.scale
    return coordinates, scales, indices


def summarize_scale(posterior):
    grid = posterior.reference.new_tensor([c.scale for c in posterior.components])
    nle = grid.new_tensor([c.negative_log_evidence for c in posterior.components])
    # Predeclared sensitivity summaries, never used to select a best test result.
    sensitivity = {}
    slab = posterior.prior_weights[1:] / posterior.prior_weights[1:].sum()
    for spike in [.25, .5, .75]:
        weights = torch.cat((grid.new_tensor([spike]), (1 - spike) * slab))
        probabilities = torch.softmax(weights.log() - nle, dim=0)
        sensitivity[str(spike)] = {"probability_zero_process_noise": float(probabilities[0]),
                                    "posterior_mean_scale": float(probabilities @ grid)}
    return {
        "grid": grid.tolist(), "prior_weights": posterior.prior_weights.tolist(),
        "posterior_weights": posterior.posterior_weights.tolist(),
        "probability_zero_process_noise": float(posterior.posterior_weights[0]),
        "posterior_mean_scale": float(grid @ posterior.posterior_weights),
        "spike_prior_sensitivity": sensitivity,
        "component_negative_log_evidence_up_to_common_constant": [c.negative_log_evidence for c in posterior.components],
        "approximation": "exact scale marginalization for one common local image linearization; approximate for nonlinear physics/rendering",
    }
