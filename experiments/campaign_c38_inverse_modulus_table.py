"""Three-node inverse-modulus interpolation; no simulation or observation I/O.

This numerical helper creates neither a physical certificate nor a posterior.
Withheld direct responses must remain withheld when this approximation fails.
The caller owns support/load/mesh/point identities and their admission checks.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InverseModulusTable:
    young_Pa: np.ndarray
    displacement_m: np.ndarray

    def __post_init__(self):
        E = np.array(self.young_Pa, dtype=float, copy=True)
        u = np.array(self.displacement_m, dtype=float, copy=True)
        if E.shape != (3,) or not np.isfinite(E).all() or np.any(E <= 0) or np.any(np.diff(E) <= 0):
            raise ValueError('Three strictly increasing positive material nodes required')
        if u.ndim < 3 or u.shape[0] != 3 or u.shape[-1] != 3 or not np.isfinite(u).all():
            raise ValueError('Three finite displacement fields with Cartesian final axis required')
        E.setflags(write=False)
        u.setflags(write=False)
        object.__setattr__(self, 'young_Pa', E)
        object.__setattr__(self, 'displacement_m', u)

    def weights(self, young_Pa):
        E = np.asarray(young_Pa, dtype=float)
        if not np.isfinite(E).all() or np.any(E < self.young_Pa[0]) or np.any(E > self.young_Pa[-1]):
            raise ValueError('No material extrapolation or silent clipping')
        # Dimensionless inverse modulus avoids unnecessarily small differences.
        nodes = self.young_Pa[0] / self.young_Pa
        x = self.young_Pa[0] / E
        weights = np.ones(E.shape + (3,))
        for i in range(3):
            for j in range(3):
                if j != i:
                    weights[..., i] *= (x - nodes[j]) / (nodes[i] - nodes[j])
        return weights

    def predict(self, young_Pa):
        return np.tensordot(self.weights(young_Pa), self.displacement_m, axes=([-1], [0]))

    def validate_withheld(self, young_Pa, direct_displacement_m, *, tolerance_m):
        """Pointwise Euclidean errors, with no replacement or refit of nodes.

        All query material values must be strictly interior and distinct from
        the table nodes. A passed finite check is not a continuous-E bound.
        """
        E = np.asarray(young_Pa, dtype=float)
        truth = np.asarray(direct_displacement_m, dtype=float)
        if E.ndim != 1 or not len(E) or len(np.unique(E)) != len(E):
            raise ValueError('Distinct explicitly withheld material values required')
        if np.any(E <= self.young_Pa[0]) or np.any(E >= self.young_Pa[-1]) or np.any(E[:, None] == self.young_Pa):
            raise ValueError('Validation materials must be separate interior nodes')
        prediction = self.predict(E)
        if truth.shape != prediction.shape or not np.isfinite(truth).all():
            raise ValueError('Matching finite separately supplied direct responses required')
        if not np.isfinite(tolerance_m) or tolerance_m <= 0:
            raise ValueError('Prospectively chosen positive interpolation tolerance required')
        error = np.linalg.norm(prediction - truth, axis=-1)
        maxima = error.reshape(len(E), -1).max(axis=1)
        return dict(young_Pa=E.tolist(), maximum_point_error_m=maxima.tolist(),
                    maximum_error_m=float(maxima.max()), tolerance_m=float(tolerance_m),
                    passed=bool(np.all(maxima <= tolerance_m)),
                    finite_validation_only=True, validation_promoted_to_table=False,
                    per_point_error_m=error)
