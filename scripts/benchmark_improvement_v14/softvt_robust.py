"""v14b validity-triggered rigid/contact fallback; frozen finite-base path unchanged."""
import numpy as np
from scripts.benchmark_improvement_v14.softvt_predict import predict as frozen_predict
from scripts.contact_benchmark_v12.model_selection import rigid_nodal
from scripts.contact_benchmark_v12.contact_rollout import rollout


def predict(data,base_prediction=None,config=None):
    # Reuse frozen validation and exact finite-base computation without changes.
    original,metadata=frozen_predict(data,base_prediction,config)
    base=np.asarray(base_prediction,float)
    if np.isfinite(base).all():return original,metadata
    failed=np.flatnonzero(~np.isfinite(base).all(axis=(1,2))).tolist()
    meta=dict(method='validity_triggered_rigid_contact_fallback',trigger='Any nonfinite original nodal frame; no target/error/case-identity selection',original_nonfinite_frames=failed,original_metadata=metadata,fallback_applied=False)
    try:
        rigid=rigid_nodal(data)
        result,contact=rollout(data,rigid,config['scene'])
        if result.shape!=base.shape or not np.isfinite(result).all():raise ValueError('Rigid/contact fallback nonfinite or malformed')
        initial=base[0] if np.isfinite(base[0]).all() else np.asarray(data['nodes'])
        result[0]=initial
        if not np.isfinite(result).all():raise ValueError('Initial reference nonfinite')
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
        return original,dict(meta,fallback_error=f'{type(error).__name__}: {error}',failure_retained=True)
    return result,dict(meta,fallback_applied=True,failure_retained=False,contact=contact,initial_reference='Original frame0'if np.isfinite(base[0]).all()else'Known initial nodes because original frame0 nonfinite',scope='Entire trajectory uses prefix-calibrated rigid model with static contact; no deformable accuracy claim. Finite original trajectories use unchanged frozen v14 predictor.')
