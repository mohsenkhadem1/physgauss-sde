"""Small full-covariance mixture for correlated physical-program uncertainty."""
import math

import torch
from torch import nn

from scripts.gaussian_boundary_flow_v02.probabilistic import _mlp


class FullCovarianceMixtureRegressor(nn.Module):
    """Two Gaussian components with 6 means and 21 Cholesky parameters each.

    At context_dim144, parameter_dim6 and width220 this has141,516 parameters,
    versus141,713 for the four-component diagonal mixture. The triangular factor
    has positive diagonal scales and bounded row-normalized off-diagonals; this
    avoids extreme triangular conditioning while permitting strong correlations.
    Density is over unconstrained physical logits, not transformed physical units.
    """
    def __init__(self, context_dim: int, parameter_dim: int = 6,
                 components: int = 2, width: int = 220,
                 log_scale_bounds: tuple[float, float] = (-7., 3.),
                 off_diagonal_limit: float = 10.):
        super().__init__()
        if min(context_dim, parameter_dim, components, width) < 1:
            raise ValueError('Positive dimensions required')
        low, high = log_scale_bounds
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError('Finite increasing log-scale bounds required')
        if not math.isfinite(off_diagonal_limit) or off_diagonal_limit <= 0:
            raise ValueError('Finite positive off-diagonal bound required')
        self.context_dim = context_dim; self.parameter_dim = parameter_dim
        self.components = components; self.log_scale_bounds = (float(low), float(high))
        self.off_diagonal_limit = float(off_diagonal_limit)
        self.triangle_size = parameter_dim*(parameter_dim+1)//2
        self.network = _mlp(context_dim, width,
                            components*(1+parameter_dim+self.triangle_size))
        rows, columns = torch.tril_indices(parameter_dim, parameter_dim)
        self.register_buffer('triangle_rows', rows)
        self.register_buffer('triangle_columns', columns)
        self.register_buffer('diagonal_locations', (rows == columns).nonzero().flatten())

    def forward(self, context):
        """Return component logits[B,K], means[B,K,D], scale_tril[B,K,D,D]."""
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError('Expected context[batch,context_dim]')
        packed = self.network(context).reshape(-1, self.components,
                    1+self.parameter_dim+self.triangle_size)
        logits = packed[..., 0]; means = packed[..., 1:1+self.parameter_dim]
        raw = packed[..., 1+self.parameter_dim:]
        log_scale = raw[..., self.diagonal_locations].clamp(*self.log_scale_bounds)
        unit = raw.new_zeros((*raw.shape[:2], self.parameter_dim, self.parameter_dim))
        bounded = self.off_diagonal_limit*torch.tanh(raw/self.off_diagonal_limit)
        unit[..., self.triangle_rows, self.triangle_columns] = bounded
        diagonal = torch.arange(self.parameter_dim, device=context.device)
        unit[..., diagonal, diagonal] = 1.
        factor = log_scale.exp()[..., :, None]*unit
        return logits, means, factor

    def log_prob(self, target, context):
        if target.ndim != 2 or target.shape != (len(context), self.parameter_dim):
            raise ValueError('Expected target[batch,parameter_dim]')
        logits, means, factor = self(context)
        delta = target[:, None, :]-means
        whitened = torch.linalg.solve_triangular(factor, delta[..., None], upper=False).squeeze(-1)
        component = -.5*(whitened.square().sum(-1)+self.parameter_dim*math.log(2*math.pi))
        component -= factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return torch.logsumexp(torch.log_softmax(logits, -1)+component, dim=-1)

    def loss(self, target, context):
        return -self.log_prob(target, context).mean()

    def mean(self, context):
        logits, means, _ = self(context)
        return (logits.softmax(-1)[..., None]*means).sum(1)

    @torch.no_grad()
    def sample(self, context, count=32, generator=None):
        if not isinstance(count, int) or count < 1:
            raise ValueError('Positive integer sample count required')
        logits, means, factor = self(context)
        choices = torch.multinomial(logits.softmax(-1), count, replacement=True, generator=generator)
        means = means.gather(1, choices[..., None].expand(-1, -1, self.parameter_dim))
        factor = factor.gather(1, choices[..., None, None].expand(-1, -1, self.parameter_dim, self.parameter_dim))
        noise = torch.randn(means.shape, dtype=context.dtype, device=context.device, generator=generator)
        return means+(factor@noise[..., None]).squeeze(-1)
