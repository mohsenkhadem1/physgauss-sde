"""Terminal observed-marker centroid correction transported by known controller pose."""
import numpy as np
from scripts.contact_benchmark_v12.object_frame import control_frames, _observed_prediction


def predict(data,base_prediction=None,config=None):
    if 'evaluation' in data:raise ValueError('Evaluator forbidden')
    base=np.asarray(base_prediction,float);end=int(data['train_end']);last=end-1
    if len(data['observed'])!=end or len(data['observed_mask'])!=end:raise ValueError('Strict observed prefix required')
    projected=_observed_prediction(data,base[last:last+1])[0]
    valid=np.asarray(data['observed_mask'][last],bool)&np.isfinite(data['observed'][last]).all(1)&np.isfinite(projected).all(1)
    if valid.sum()<3:return base.copy(),dict(fallback='Insufficient terminal shared markers')
    frames=control_frames(data)
    delta=np.mean(data['observed'][last,valid]-projected[valid],axis=0)
    local_delta=delta@frames[last][0].T
    out=base.copy()
    for t in range(end,len(base)):out[t]+=local_delta@frames[t][0]
    out[0]=base[0]
    return out,dict(method='terminal_marker_centroid_transport',fit_end=end,terminal_frame=last,shared_markers=int(valid.sum()),world_delta_m=delta.tolist(),controller_delta_m=local_delta.tolist(),initial_preserved=True,scope='Constant translation residual only; no rotation or deformation correction, no velocity or force inputs')
