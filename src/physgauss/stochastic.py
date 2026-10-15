"""Spatially correlated internal noise and independent diffusion calibration.

The neural amplitude is trained on a synthetic conditional-innovation task,
not on withheld test states. This is simulator supervision, not evidence that
diffusion is identifiable from one video. Noise acts as central edge impulses.
"""
import torch
from torch import nn
from .physics import Mesh


def spatial_basis(mesh: Mesh, rank=3, length_scale=0.45):
    centers = mesh.rest[mesh.edges].mean(-2)
    d2 = (centers[:, None] - centers[None]).square().sum(-1)
    kernel = torch.exp(-d2 / (2 * length_scale ** 2))
    values, vectors = torch.linalg.eigh(kernel)
    if not 1 <= rank <= len(values):
        raise ValueError("Noise rank must lie between one and the edge count.")
    basis = vectors[:, -rank:] * values[-rank:].clamp_min(0).sqrt()
    # Unit marginal variance after truncation; rank controls spatial modes.
    return basis / basis.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-10)


def calibration_law(features):
    strain, speed = features.unbind(-1)
    return .0017 * (1 + 2 * torch.sqrt(strain.square() + .001 ** 2)
                    + .18 * torch.sqrt(speed.square() + .001 ** 2))


class NeuralDiffusion(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)).to(dtype)

    def forward(self, features):
        # Nonnegative force-noise amplitude; inputs are objective edge invariants.
        return .001 * torch.nn.functional.softplus(self.net(features).squeeze(-1)) + .0001


def train_diffusion(seed=918, steps=350, dtype=torch.float64):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net = NeuralDiffusion(dtype)
        gen = torch.Generator().manual_seed(seed + 1)
        features = torch.rand(4096, 2, generator=gen, dtype=dtype)
        features = features * features.new_tensor([.7, 3.0]) - features.new_tensor([.25, 1.5])
        innovations = calibration_law(features) * torch.randn(4096, generator=gen, dtype=dtype)
        optimizer = torch.optim.Adam(net.parameters(), lr=.025)
        for _ in range(steps):
            sigma = net(features)
            loss = (.5 * (innovations / sigma).square() + sigma.log()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        validation = torch.rand(1024, 2, generator=gen, dtype=dtype)
        validation = validation * features.new_tensor([.7, 3.0]) - features.new_tensor([.25, 1.5])
        with torch.no_grad():
            ratio = net(validation) / calibration_law(validation)
        stats = {"supervision": "independent synthetic conditional Gaussian innovations",
                 "seed": seed, "train_examples": 4096, "validation_examples": 1024,
                 "steps": steps, "relative_sigma_rmse": float((ratio - 1).square().mean().sqrt())}
        net.eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        return net, stats
