"""Bounded float64 tensor contraction for the existing residual atlas.

The interpolation coordinates, exact-node handling and prior checks use the
original CPU basis. Only its tensor contraction moves to the selected device.
This provides numerical acceleration, not new physical/model admission.
"""
import numpy as np
import torch
from experiments.campaign_nonlinear_residual_atlas_v01 import barycentric_basis


class TorchResidualAtlas:
    def __init__(self, atlas, *, device):
        self.atlas = atlas
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Explicit CPU or CUDA device required')
        if self.device.type == 'cuda' and self.device.index is None:
            raise ValueError('Explicit CUDA index required')
        self.table = torch.tensor(atlas.delta.copy(), dtype=torch.float64,
                                  device=self.device)
        self.fingerprint = atlas.fingerprint

    def predict(self, unit, labels, *, chunk=4096):
        u = np.asarray(unit, float)
        selected = tuple(labels)
        if u.ndim != 2 or u.shape[1] != 2 or not np.isfinite(u).all() or not selected:
            raise ValueError('Finite unit[batch,2] and nonempty labels')
        if type(chunk) is not int or chunk < 1:
            raise ValueError('Bounded positive query chunk')
        indices = [self.atlas.labels.index(v) for v in selected]
        result = np.empty((len(u), len(selected), self.atlas.delta.shape[3]))
        with torch.inference_mode():
            table = self.table[:, :, indices, :].reshape(
                len(self.atlas.axis)**2, -1)
            for start in range(0, len(u), chunk):
                q = u[start:start + chunk]
                a = barycentric_basis(q[:, 0], self.atlas.axis)
                b = barycentric_basis(q[:, 1], self.atlas.axis)
                product = (a[:, :, None] * b[:, None, :]).reshape(len(q), -1)
                weights = torch.as_tensor(product, dtype=torch.float64,
                                          device=self.device)
                value = (weights @ table).cpu().numpy()
                result[start:start + len(q)] = value.reshape(
                    len(q), len(selected), self.atlas.delta.shape[3])
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite accelerated atlas response')
        return result
