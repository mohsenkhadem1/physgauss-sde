"""Continuous, permutation-invariant projected Gaussian volume rendering.

Retains the EWA footprint approximation, but integrates positive extinction
through overlapping Gaussian depth profiles instead of sorting whole splats.
This is a declared image-formation change, not exact 3D ray tracing.
"""
from dataclasses import dataclass
import math
import torch
from .camera import project
from .physics import deformation
from .render import GaussianRenderer


@dataclass
class VolumeRenderConfig:
    bins: int = 128
    near_m: float = 1.3
    far_m: float = 3.8

    def validate(self):
        if self.bins < 8 or not (math.isfinite(self.near_m) and math.isfinite(self.far_m)) or self.near_m >= self.far_m:
            raise ValueError("Need >=8 depth bins and ordered finite integration bounds.")
        return self


class ContinuousGaussianRenderer(GaussianRenderer):
    def __init__(self, mesh, size=24, config=VolumeRenderConfig()):
        super().__init__(mesh, size)
        self.config = config.validate()
        # Two outer slabs include the tails exactly. For the declared scene
        # their mass is audited; no moving bounds, sorting, or visibility clamp.
        self.boundaries = torch.linspace(config.near_m, config.far_m, config.bins-1, dtype=mesh.rest.dtype)

    def components(self, x, eye, rotation):
        centers = torch.einsum("pk,bftkd->bftpd", self.bary, x[..., self.mesh.cells, :])
        batch, frames, cells, primitives, _ = centers.shape
        centers = centers.reshape(batch, frames, -1, 3)
        f = deformation(x, self.mesh)
        covariance = f@self.sigma0@f.transpose(-1, -2)
        covariance = covariance.unsqueeze(-3).expand(batch, frames, cells, primitives, 3, 3).reshape(batch, frames, -1, 3, 3)
        mean, depth, jac = project(centers, eye, rotation, self.size)
        camera_covariance = rotation[None, :, None]@covariance@rotation[None, :, None].transpose(-1, -2)
        cov2 = jac@camera_covariance@jac.transpose(-1, -2)+.30*torch.eye(2, dtype=x.dtype)
        a, off, d = cov2[..., 0, 0], cov2[..., 0, 1], cov2[..., 1, 1]
        delta = self.pixels[None, None, None]-mean[..., None, None, :]
        dx, dy = delta.unbind(-1)
        mahal = (d[..., None, None]*dx.square()-2*off[..., None, None]*dx*dy+a[..., None, None]*dy.square())/(a*d-off.square())[..., None, None]
        alpha = .60*torch.exp(-.5*mahal)
        optical_depth = -torch.log1p(-alpha)
        sd_depth = camera_covariance[..., 2, 2].sqrt()
        z = (self.boundaries[None, None, :, None]-depth[:, :, None])/(math.sqrt(2)*sd_depth[:, :, None])
        cdf = .5*(1+torch.erf(z))
        cdf = torch.cat((torch.zeros_like(cdf[:, :, :1]), cdf, torch.ones_like(cdf[:, :, :1])), 2)
        fractions = cdf[:, :, 1:]-cdf[:, :, :-1]
        return optical_depth, fractions

    def __call__(self, x, eye, rotation):
        optical, fractions = self.components(x, eye, rotation)
        return self.integrate(optical, fractions, self.colors)

    @staticmethod
    def integrate(optical, fractions, colors):
        extinction = torch.einsum("btsg,btghw->btshw", fractions, optical)
        source = torch.einsum("btsg,btghw,gc->btshwc", fractions, optical, colors)
        # The exact zero-density limit emits zero. The tiny denominator only
        # prevents 0/0 in empty rays; it is far below the quadrature tolerance.
        color = source/extinction[..., None].clamp_min(1e-30)
        before = torch.cumsum(extinction, 2)-extinction
        weights = torch.exp(-before)*(-torch.expm1(-extinction))
        return (weights[..., None]*color).sum(2)+torch.exp(-extinction.sum(2))[..., None]*.035
