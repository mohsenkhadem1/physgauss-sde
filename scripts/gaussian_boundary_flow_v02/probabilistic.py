"""Matched inference baselines for Gaussian-boundary identification.

The mixture is over unconstrained physical-program logits. Like the conditional
flow, sampled logits must be propagated through the same parameter transform and
mechanical solver before averaging predicted motion. It is not a mixture over
rendered images or a claim of a calibrated posterior without validation.
"""
import math

import torch
from torch import nn


def _mlp(context_dim: int, width: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(context_dim, width), nn.SiLU(),
        nn.Linear(width, width), nn.SiLU(),
        nn.Linear(width, width), nn.SiLU(),
        nn.Linear(width, output_dim),
    )


class GaussianMixtureRegressor(nn.Module):
    """Conditional K-component diagonal Gaussian mixture over latent parameters.

    Defaults: six parameters, four components, and 141,713 trainable parameters
    for a 144-D context. The original 192-wide flow has 141,894 parameters, so
    representation size differs by <0.2%. Training/inference budgets still need
    to be recorded separately; matched capacity alone is not matched compute.

    Log standard deviations are clipped to [-7, 3] for finite density evaluation.
    Densities are with respect to unconstrained logits, not transformed SI units.
    """

    def __init__(self, context_dim: int, parameter_dim: int = 6,
                 components: int = 4, width: int = 221,
                 log_scale_bounds: tuple[float, float] = (-7.0, 3.0)):
        super().__init__()
        if min(context_dim, parameter_dim, components, width) < 1:
            raise ValueError('Positive context, parameter, component and width sizes required')
        low, high = log_scale_bounds
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError('Finite increasing log-scale bounds required')
        self.context_dim = context_dim
        self.parameter_dim = parameter_dim
        self.components = components
        self.log_scale_bounds = (float(low), float(high))
        self.network = _mlp(context_dim, width, components * (1 + 2 * parameter_dim))

    def forward(self, context: torch.Tensor):
        """Return mixing logits [B,K], means and log-scales [B,K,D]."""
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError('Expected context [batch, context_dim]')
        raw = self.network(context).reshape(-1, self.components, 1 + 2 * self.parameter_dim)
        logits = raw[..., 0]
        means = raw[..., 1:1 + self.parameter_dim]
        log_scales = raw[..., 1 + self.parameter_dim:].clamp(*self.log_scale_bounds)
        return logits, means, log_scales

    def log_prob(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Stable mixture log-density, one scalar per case [B]."""
        if target.ndim != 2 or target.shape != (len(context), self.parameter_dim):
            raise ValueError('Expected target [batch, parameter_dim]')
        logits, means, log_scales = self(context)
        residual = (target[:, None, :] - means) * torch.exp(-log_scales)
        component = -.5 * (residual.square() + math.log(2 * math.pi)).sum(-1)
        component = component - log_scales.sum(-1)
        return torch.logsumexp(torch.log_softmax(logits, dim=-1) + component, dim=-1)

    def loss(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(target, context).mean()

    def mean(self, context: torch.Tensor) -> torch.Tensor:
        """Analytic latent mean, not generally the mean of a physical rollout."""
        logits, means, _ = self(context)
        return (logits.softmax(-1)[..., None] * means).sum(1)

    @torch.no_grad()
    def sample(self, context: torch.Tensor, count: int = 32,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Return reproducible [B,count,D] samples when a seeded generator is given."""
        if not isinstance(count, int) or count < 1:
            raise ValueError('Positive integer sample count required')
        logits, means, log_scales = self(context)
        choices = torch.multinomial(logits.softmax(-1), count, replacement=True,
                                    generator=generator)
        index = choices[..., None].expand(-1, -1, self.parameter_dim)
        selected_mean = means.gather(1, index)
        selected_scale = log_scales.gather(1, index).exp()
        noise = torch.randn(selected_mean.shape, dtype=context.dtype,
                            device=context.device, generator=generator)
        return selected_mean + selected_scale * noise


class ParameterRegressor(nn.Module):
    """Point estimate baseline with a comparable MLP parameter budget.

    With default context_dim=144 and six parameters, width=230 yields 140,996
    parameters, within 1% of the original flow. Its forward output can be trained
    with a trajectory/task loss; the supplied convenience loss is latent MSE only.
    """

    def __init__(self, context_dim: int, parameter_dim: int = 6, width: int = 230):
        super().__init__()
        if min(context_dim, parameter_dim, width) < 1:
            raise ValueError('Positive context, parameter and width sizes required')
        self.context_dim = context_dim
        self.parameter_dim = parameter_dim
        self.network = _mlp(context_dim, width, parameter_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError('Expected context [batch, context_dim]')
        return self.network(context)

    def loss(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        prediction = self(context)
        if target.shape != prediction.shape:
            raise ValueError('Target shape must match predicted latent shape')
        return (prediction - target).square().mean()
