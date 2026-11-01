"""Proper marginal scores and empirical coverage for a declared finite belief.

Coverage here is a measured outcome, not a claim of calibrated continuous
parameter inference. We retain all target coordinates, including weak signals.
"""
import numpy as np


def marginal_scores(values,weights,target):
    weights=np.asarray(weights,float)
    values=np.asarray(values,float)
    target=np.asarray(target,float)
    if weights.ndim!=1 or not len(weights) or not np.isfinite(weights).all():
        raise ValueError('Finite nonempty one-dimensional belief required')
    if values.ndim<2 or values.shape[0]!=len(weights) or values.shape[1:]!=target.shape or target.size==0:
        raise ValueError('Response sample and target shapes must match exactly')
    values=values.reshape(len(weights),-1)
    target=target.reshape(-1)
    if values.shape[1]!=len(target) or not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError('Finite matching response coordinates required')
    if (weights<0).any() or not np.isclose(weights.sum(),1):raise ValueError('Normalized belief required')
    weights=weights/weights.sum()
    order=np.argsort(values,axis=0,kind='stable')
    sorted_values=np.take_along_axis(values,order,axis=0)
    sorted_weights=weights[order]
    cumulative=np.cumsum(sorted_weights,axis=0)
    half_pairwise=(sorted_weights*sorted_values*(2*cumulative-sorted_weights-1)).sum(0)
    crps=weights@abs(values-target)-half_pairwise
    lower=sorted_values[(cumulative>=.05).argmax(axis=0),np.arange(len(target))]
    upper=sorted_values[(cumulative>=.95).argmax(axis=0),np.arange(len(target))]
    return dict(crps_mean=float(np.mean(crps)),central90_coordinate_coverage=float(np.mean((target>=lower)&(target<=upper))),
                mean_interval_width=float(np.mean(upper-lower)),
                interval_rule='Finite-belief inverse CDF; empirical coordinate coverage, not independent-coordinate confidence')
