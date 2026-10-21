"""Object-level split-conformal bands around frozen deformation predictions.

The scalar score is the maximum standardized coordinate error over a fixed
target field, potentially including both known-force programs. Coordinates and
frames within one object are NOT independent calibration examples. A band
scale is calibrated using independent complete objects, after the predictor is
frozen. This does not calibrate Bayesian material probabilities or provide
conditional coverage for an individual support, material, or selected action.

Method: Angelopoulos and Bates, arXiv:2107.07511, sections1.1 and2.3.
https://arxiv.org/abs/2107.07511
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def object_score(truth, mean, latent_std, *, numerical_floor_m=1e-9):
    """One scalar per independent object, not per coordinate/frame/program.

    The 1nm floor prevents division by numerical zero. It is not an estimate
    of physical discrepancy or sensor noise. Global scale calibration can
    absorb misspecification for the stipulated target population marginally.
    """
    truth, mean, std = [np.asarray(x,dtype=float) for x in (truth,mean,latent_std)]
    if (not truth.size or truth.shape!=mean.shape or truth.shape!=std.shape
            or any(not np.isfinite(x).all() for x in [truth,mean,std])
            or (std<0).any() or not np.isfinite(numerical_floor_m) or numerical_floor_m<=0):
        raise ValueError('Finite matching nonempty fields and nonnegative standard deviations required')
    return float(np.max(np.abs(truth-mean)/np.maximum(std,numerical_floor_m)))


@dataclass(frozen=True)
class BandCalibration:
    scale: float
    scores: tuple
    alpha: float
    order_statistic_rank: int
    numerical_floor_m: float

    def metadata(self):
        return dict(calibration_objects=len(self.scores),score='max absolute standardized error over the entire fixed target field',
            alpha=self.alpha,order_statistic_rank=self.order_statistic_rank,
            scale=self.scale if math.isfinite(self.scale) else None,
            infinite_band=not math.isfinite(self.scale),minimum_scale=1.,numerical_floor_m=self.numerical_floor_m,
            band_formula='mean +/- scale * max(latent_std, numerical_floor_m)',
            assumptions='Frozen predictor; independent/exchangeable calibration and evaluation objects from the same stipulated population; fixed score/target before calibration.',
            claim='Marginal simultaneous target-field coverage for a new object; no per-material/support conditional coverage or calibrated Bayesian posterior claim.',
            source='https://arxiv.org/abs/2107.07511')

    def contains(self, truth, mean, latent_std):
        return object_score(truth,mean,latent_std,numerical_floor_m=self.numerical_floor_m)<=self.scale

    def bounds(self, mean, latent_std):
        object_score(mean,mean,latent_std,numerical_floor_m=self.numerical_floor_m)
        mean,std=np.asarray(mean,dtype=float),np.asarray(latent_std,dtype=float)
        half_width=self.scale*np.maximum(std,self.numerical_floor_m)
        return mean-half_width,mean+half_width


def calibrate(scores, *, alpha=.1, numerical_floor_m=1e-9):
    values=np.asarray(scores,dtype=float)
    if (values.ndim!=1 or not len(values) or not np.isfinite(values).all()
            or (values<0).any() or not 0<alpha<1
            or not np.isfinite(numerical_floor_m) or numerical_floor_m<=0):
        raise ValueError('Finite nonnegative object scores and alpha in(0,1) required')
    rank=math.ceil((len(values)+1)*(1-alpha))
    # Never clip an out-of-range finite-sample rank to the largest score.
    # At such a stringent alpha the standard procedure returns the whole set.
    scale=math.inf if rank>len(values) else max(1.,float(np.sort(values)[rank-1]))
    return BandCalibration(scale,tuple(values),float(alpha),rank,float(numerical_floor_m))
