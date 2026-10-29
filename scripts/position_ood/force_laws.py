"""Unseen environmental force laws for a separate frozen-model stress test."""
import numpy as np
from scipy import sparse
from scripts.position_ood.smooth_fixture import SmoothMaterialFixture


class ForceLawFixture(SmoothMaterialFixture):
    def __init__(self, *args, force_law, **kwargs):
        self.force_law = force_law
        super().__init__(*args, **kwargs)

    def spatial_field(self, x, tangent=False):
        law = self.force_law
        weights = self.volume_weights
        if law['kind'] == 'anisotropic_harmonic':
            delta = x - np.asarray(law['center'])
            matrix = np.asarray(law['stiffness_N_per_m'])
            gradient = weights[:, None] * (delta @ matrix.T)
            energy = .5 * np.sum(delta * gradient)
            blocks = weights[:, None, None] * matrix[None] if tangent else None
        elif law['kind'] == 'two_gaussian':
            energy = 0.
            gradient = np.zeros_like(x)
            blocks = np.zeros((len(x), 3, 3)) if tangent else None
            for component in law['components']:
                delta = x - np.asarray(component['center'])
                s2 = component['width_m'] ** 2
                value = component['amplitude_J'] * weights * np.exp(-np.sum(delta**2, axis=1)/(2*s2))
                energy += value.sum()
                gradient -= value[:, None] * delta/s2
                if tangent:
                    blocks += value[:, None, None] * (delta[:, :, None]*delta[:, None, :]/s2**2-np.eye(3)[None]/s2)
        else:
            raise ValueError('Undeclared environmental force law')
        hessian = None
        if tangent:
            indices = np.arange(x.size).reshape(-1, 3)
            rows = np.broadcast_to(indices[:, :, None], blocks.shape)
            cols = np.broadcast_to(indices[:, None, :], blocks.shape)
            hessian = sparse.coo_matrix((blocks.ravel(), (rows.ravel(), cols.ravel())),
                                       shape=(x.size, x.size)).tocsr()
        return float(energy), gradient, hessian
