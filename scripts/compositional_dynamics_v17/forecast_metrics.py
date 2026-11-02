"""Common surface metrics with explicit sampling and clock conventions."""
import numpy as np
from scripts.compositional_dynamics_v17.sequence_metrics import matched_sequence_metrics


def reference_surface_weights(queries,dimensions):
    q=np.asarray(queries,float);dims=np.asarray(dimensions,float)
    if q.shape!=(98,3) or dims.shape!=(3,) or np.any(dims<=0) or not np.isfinite(dims).all():raise ValueError('Declared98point boxsurface and positive dimensions required')
    weights=np.zeros(len(q));axis_weight={0.:.125,.25:.25,.5:.25,.75:.25,1.:.125}
    if not np.all(np.isin(q,list(axis_weight))) or len(np.unique(q,axis=0))!=98:raise ValueError('Exact unique5grid surface queries required')
    for i,point in enumerate(q):
        for axis in range(3):
            if point[axis] in (0.,1.):
                other=[j for j in range(3) if j!=axis]
                weights[i]+=float(np.prod(dims[other]))*axis_weight[point[other[0]]]*axis_weight[point[other[1]]]
    if np.any(weights<=0) or not np.isclose(weights.sum(),2*(dims[0]*dims[1]+dims[0]*dims[2]+dims[1]*dims[2]),atol=1e-14,rtol=0):raise ValueError('Complete closed surface quadrature required')
    return weights


def compare_surface_forecast(rest,queries,dimensions,reference_times,reference_positions,prediction_times,prediction_positions,*,frame_dt_s=.01):
    rt=np.asarray(reference_times,float);pt=np.asarray(prediction_times,float)
    nominal=np.arange(len(rt))*frame_dt_s
    if len(pt)>len(rt) or not np.isfinite(rt).all() or not np.isfinite(pt).all() or not np.allclose(rt,nominal,atol=1e-12,rtol=0) or not np.allclose(pt,nominal[:len(pt)],atol=1e-12,rtol=0):
        raise ValueError('Declared simultaneous frames required; no temporal alignment or interpolation')
    weights=reference_surface_weights(queries,dimensions)
    result=matched_sequence_metrics(rest,nominal,reference_positions,nominal[:len(pt)],prediction_positions,weights,characteristic_length_m=float(min(dimensions)))
    result['surface_centroid_error_m']=result.pop('com_error_m',None)
    for row in result['rows']:row['surface_centroid_error_m']=row.pop('com_error_m')
    result['weight_scope']='Rest-area trapezoidal quadrature on98unique material surfacepoints. Surface centroid is NOT volume center of mass.'
    result['clock_roundoff_max_s']=float(max(np.max(abs(rt-nominal),initial=0),np.max(abs(pt-nominal[:len(pt)]),initial=0)))
    result['clock_scope']='Public integer frame clock; atmost1e-12s floating point representation difference accepted. No resampling, phase shift, trajectory fitting or dropped frames.'
    result['deformation_scope']='Proper rigid alignment only for separate shape/deformation diagnostics; world-position errors unchanged.'
    return result
