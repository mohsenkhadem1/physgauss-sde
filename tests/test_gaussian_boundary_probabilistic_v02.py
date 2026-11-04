import math

import pytest
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from scripts.gaussian_boundary_flow.flow import ConditionalFlow
from scripts.gaussian_boundary_flow_v02.probabilistic import (
    GaussianMixtureRegressor, ParameterRegressor,
)


def test_mixture_log_density_matches_torch_distribution():
    torch.manual_seed(64)
    model = GaussianMixtureRegressor(5, 3, components=4, width=12).double()
    context = torch.randn(7, 5, dtype=torch.float64)
    target = torch.randn(7, 3, dtype=torch.float64)
    logits, means, log_scales = model(context)
    reference = MixtureSameFamily(Categorical(logits=logits),
                                 Independent(Normal(means, log_scales.exp()), 1))
    assert torch.allclose(model.log_prob(target, context), reference.log_prob(target),
                          atol=1e-12, rtol=1e-12)


def test_log_density_is_finite_for_tiny_scales_and_unbalanced_weights():
    model = GaussianMixtureRegressor(2, 2, width=4).double()
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        bias = model.network[-1].bias.reshape(4, 5)
        bias[:, 0] = torch.tensor([1000., -1000., -1000., -1000.])
        bias[:, 3:] = -1000.
    value = model.log_prob(torch.tensor([[1., 2.]], dtype=torch.float64),
                           torch.zeros(1, 2, dtype=torch.float64))
    assert torch.isfinite(value).all()


def test_sampling_reproduces_and_matches_known_mixture_moments():
    model = GaussianMixtureRegressor(2, 1, components=2, width=4).double()
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        model.network[-1].bias.copy_(torch.tensor([
            math.log(.25), -2., math.log(.5),
            math.log(.75), 2., math.log(.5),
        ], dtype=torch.float64))
    context = torch.zeros(1, 2, dtype=torch.float64)
    a = model.sample(context, 30000, torch.Generator().manual_seed(177))
    b = model.sample(context, 30000, torch.Generator().manual_seed(177))
    assert torch.equal(a, b)
    assert a.shape == (1, 30000, 1)
    # Analytic mean 1; variance = within .25 + between 3.
    assert abs(float(a.mean()) - 1.) < .04
    assert abs(float(a.var()) - 3.25) < .07
    assert torch.allclose(model.mean(context), torch.tensor([[1.]], dtype=torch.float64))


def test_log_prob_gradient_matches_finite_difference_and_reaches_all_heads():
    torch.manual_seed(983)
    model = GaussianMixtureRegressor(3, 2, width=8).double()
    c = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(4, 2, dtype=torch.float64)
    loss = model.loss(y, c)
    loss.backward()
    assert torch.isfinite(c.grad).all()
    grad = model.network[-1].bias.grad.reshape(4, 5)
    assert all(grad[:, region].abs().sum() > 0 for region in (0, slice(1, 3), slice(3, 5)))
    shift = torch.zeros_like(c); shift[0, 1] = 1e-5
    finite = (model.loss(y, c.detach() + shift) - model.loss(y, c.detach() - shift)) / 2e-5
    assert torch.allclose(c.grad[0, 1], finite, atol=1e-9, rtol=1e-5)


def test_capacity_and_mean_baseline_contract():
    flow = ConditionalFlow(144)
    mixture = GaussianMixtureRegressor(144)
    point = ParameterRegressor(144)
    count = lambda model: sum(p.numel() for p in model.parameters())
    assert abs(count(mixture) / count(flow) - 1) < .002
    assert abs(count(point) / count(flow) - 1) < .01
    context = torch.zeros(3, 144)
    assert point(context).shape == (3, 6)
    assert mixture.sample(context).shape == (3, 32, 6)
    assert torch.isfinite(point.loss(torch.ones(3, 6), context))


def test_invalid_shapes_and_counts_rejected():
    model = GaussianMixtureRegressor(3)
    with pytest.raises(ValueError): model(torch.zeros(2, 4))
    with pytest.raises(ValueError): model.log_prob(torch.zeros(2, 5), torch.zeros(2, 3))
    with pytest.raises(ValueError): model.sample(torch.zeros(2, 3), count=0)
    with pytest.raises(ValueError): GaussianMixtureRegressor(3, components=0)
    with pytest.raises(ValueError): GaussianMixtureRegressor(3, log_scale_bounds=(3., -7.))
