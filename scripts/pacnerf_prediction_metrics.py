"""Independent NumPy ensemble scores; evaluation only, never inference input."""
import numpy as np


def prediction_scores(draws,truth):
    draws=np.asarray(draws,dtype=np.float64);truth=np.asarray(truth,dtype=np.float64)
    levels=[]
    for level in [.5,.8,.9,.95]:
        lo,hi=np.quantile(draws,[(1-level)/2,(1+level)/2],axis=0)
        included=(truth>=lo)&(truth<=hi);width=hi-lo
        interval=width+2/(1-level)*(np.maximum(lo-truth,0)+np.maximum(truth-hi,0))
        levels.append({'nominal':level,'coverage':float(included.mean()),
            'coverage_xyz':included.mean(tuple(range(included.ndim-1))).tolist(),
            'mean_width_m':float(width.mean()),'mean_interval_score_m':float(interval.mean())})
    ordered=np.sort(draws,axis=0)
    weights=(2*np.arange(1,len(draws)+1)-len(draws)-1).reshape((-1,)+(1,)*(draws.ndim-1))
    crps=np.abs(draws-truth).mean(0)-(weights*ordered).sum(0)/(len(draws)**2)
    return {'mean_crps_m':float(crps.mean()),'coverage_curve':levels,'interval_draws':len(draws),
        'qualification':'empirical ensemble prediction scores; descriptive over correlated coordinates of deterministic trajectories, not independent calibration datasets'}
