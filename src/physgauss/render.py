"""Small differentiable anisotropic Gaussian renderer with alpha compositing.

Gaussian centers are barycentric material points. Covariance F Sigma0 F^T
describes their extent; it is deliberately separate from posterior uncertainty.
This dense reference implementation favors clarity over real-time performance.
"""
import torch
from torch import Tensor
from .camera import project
from .physics import Mesh, deformation


class GaussianRenderer:
    def __init__(self, mesh: Mesh, size=24):
        if size < 8:
            raise ValueError("Image size must be at least eight pixels.")
        self.mesh, self.size = mesh, size
        self.bary = mesh.rest.new_tensor([[.25, .25, .25, .25],
                                         [.55, .15, .15, .15],
                                         [.15, .55, .15, .15],
                                         [.15, .15, .55, .15],
                                         [.15, .15, .15, .55]])
        rest_centers = torch.einsum("pk,tkd->tpd", self.bary, mesh.rest[mesh.cells])
        rest_min, rest_max = mesh.rest.amin(0), mesh.rest.amax(0)
        colors = .18 + .72 * (rest_centers - rest_min) / (rest_max - rest_min)
        self.colors = colors.reshape(-1, 3)
        self.sigma0 = torch.eye(3, dtype=mesh.rest.dtype) * (0.105 ** 2)
        yy, xx = torch.meshgrid(torch.arange(size, dtype=mesh.rest.dtype),
                                torch.arange(size, dtype=mesh.rest.dtype), indexing="ij")
        self.pixels = torch.stack((xx, yy), dim=-1)

    def __call__(self, x: Tensor, eye: Tensor, rotation: Tensor):
        centers = torch.einsum("pk,bftkd->bftpd", self.bary, x[..., self.mesh.cells, :])
        b, t, cells, primitives, _ = centers.shape
        centers = centers.reshape(b, t, -1, 3)
        f = deformation(x, self.mesh)
        cov = f @ self.sigma0 @ f.transpose(-1, -2)
        cov = cov.unsqueeze(-3).expand(b, t, cells, primitives, 3, 3).reshape(b, t, -1, 3, 3)
        mean, depth, jac = project(centers, eye, rotation, self.size)
        cam_cov = rotation[None, :, None] @ cov @ rotation[None, :, None].transpose(-1, -2)
        cov2 = jac @ cam_cov @ jac.transpose(-1, -2)
        cov2 = cov2 + torch.eye(2, dtype=x.dtype) * .30
        a, off, d = cov2[..., 0, 0], cov2[..., 0, 1], cov2[..., 1, 1]
        det = a * d - off.square()
        delta = self.pixels[None, None, None] - mean[..., None, None, :]
        dx, dy = delta.unbind(-1)
        mahal = (d[..., None, None] * dx.square() - 2 * off[..., None, None] * dx * dy
                 + a[..., None, None] * dy.square()) / det[..., None, None]
        alpha = .60 * torch.exp(-.5 * mahal)
        order = depth.argsort(-1)
        alpha = alpha.gather(2, order[..., None, None].expand_as(alpha))
        colors = self.colors[None, None].expand(b, t, -1, -1)
        colors = colors.gather(2, order[..., None].expand_as(colors))
        transmission = torch.cumprod(torch.cat((torch.ones_like(alpha[:, :, :1]),
                                                1 - alpha), dim=2), dim=2)
        rgb = torch.einsum("btghw,btgc->bthwc", alpha * transmission[:, :, :-1], colors)
        return rgb + transmission[:, :, -1, :, :, None] * .035
