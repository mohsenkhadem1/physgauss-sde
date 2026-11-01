"""Low-dimensional positive-volume affine strain residual driven by controller positions."""
import numpy as np
from scipy.linalg import polar
from scripts.contact_benchmark_v12.object_frame import control_frames, _observed_prediction


def predict(data,base_prediction=None,config=None):
    if 'evaluation' in data:raise ValueError('Evaluator forbidden')
    base=np.asarray(base_prediction,float);end=int(data['train_end'])
    if len(data['observed'])!=end or not np.isfinite(base).all():raise ValueError('Finite anchor and strict prefix required')
    frames=control_frames(data);projected=_observed_prediction(data,base[:end]);strain=[];indices=[]
    for t in range(end):
        valid=data['observed_mask'][t]&np.isfinite(data['observed'][t]).all(1)
        if valid.sum()<6:continue
        R,b=frames[t];x=projected[t,valid]@R.T;y=data['observed'][t,valid]@R.T
        x-=x.mean(0);y-=y.mean(0)
        A=np.linalg.lstsq(x,y,rcond=1e-6)[0]
        _,S=polar(A);w,V=np.linalg.eigh(S);logS=(V*np.log(np.clip(w,.5,2.)))@V.T
        strain.append(logS.reshape(9));indices.append(t)
    if len(indices)<6:return base.copy(),dict(fallback='Insufficient affine observations')
    # Known controller displacement only; no clock extrapolation or force labels.
    controls=np.asarray(data['controls']).mean(1);F=controls-controls[0]
    scale=np.maximum(np.std(F[:end],axis=0),.001);Z=np.c_[np.ones(len(F)),F/scale]
    X=Z[indices];penalty=np.eye(4)*.1;penalty[0,0]=.001
    coefficients=np.linalg.solve(X.T@X+penalty,X.T@np.asarray(strain))
    out=base.copy();bounds=[]
    for t in range(1,len(base)):
        R,b=frames[t];L=(Z[t]@coefficients).reshape(3,3);L=(L+L.T)/2
        values,V=np.linalg.eigh(L);values=np.clip(values,-np.log(2.),np.log(2.));S=(V*np.exp(values))@V.T
        center=base[t].mean(0);out[t]=((base[t]-center)@R.T@S)@R+center
        bounds.append(np.exp(values).tolist())
    out[0]=base[0]
    return out,dict(method='controller_conditioned_affine_logstrain',fit_end=end,fit_frames=indices,coefficients=coefficients.tolist(),feature_scale=scale.tolist(),stretch_bounds=[.5,2.],initial_preserved=True,scope='Empirical six-dimensional symmetric strain correction, positive-determinant affine transform of anchor; not material identification or repaired inherited inversions')
