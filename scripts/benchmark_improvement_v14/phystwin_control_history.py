"""Direct support-space positional ridge with causal controller history."""
import numpy as np
from scipy.spatial import cKDTree
from scripts.external_benchmark.evaluate import features
LAGS=(0.,.033,.067,.1,.2,.4)

def marker_fit(data,end,tau):
    if 'evaluation'in data or end>len(data['observed']):raise ValueError('Strict prefix required')
    F=features(data)
    if tau:
        history=F.copy()
        for t in range(1,len(F)):
            decay=np.exp(-(data['times'][t]-data['times'][t-1])/tau)
            history[t]=decay*history[t-1]+(1-decay)*F[t]
        F=np.c_[F,history-F]
    mean=F[:end].mean(0);std=F[:end].std(0);std[std<1e-5]=1
    Z=np.c_[np.ones(len(F)),(F-mean)/std];target=data['observed'][:end]-data['observed_initial'][None];mask=data['observed_mask'][:end]&np.isfinite(target).all(2)
    pred=np.zeros((len(F),target.shape[1],3))
    # Group equal visibility patterns so one linear solve handles many tracks.
    patterns,groups=np.unique(mask.T,axis=0,return_inverse=True)
    for group,use in enumerate(patterns):
        if not use.any():continue
        ids=np.flatnonzero(groups==group);X=Z[:end][use];Y=target[use][:,ids].reshape(use.sum(),-1)
        coef=np.linalg.solve(X.T@X+.1*np.eye(Z.shape[1]),X.T@Y)
        pred[:,ids]=(Z@coef).reshape(len(F),len(ids),3)
    return pred

def choose_lag(data):
    end=int(data['train_end']);cut=max(3,int(.7*end));valid=data['observed_mask'][cut:end]&np.isfinite(data['observed'][cut:end]).all(2);scores=[]
    for tau in LAGS:
        pred=marker_fit(data,cut,tau)+data['observed_initial'][None]
        scores.append(float(np.mean((pred[cut:end][valid]-data['observed'][cut:end][valid])**2)))
    return LAGS[int(np.argmin(scores))],dict(inner_cut=cut,fit_end=end,lag_grid_seconds=list(LAGS),validation_rmse_mm=(np.sqrt(scores)*1000).tolist())

def support(data,displacement,initial_reference):
    initial=data['support_geometry'];d,ids=cKDTree(data['observed_initial']).query(initial,k=min(6,len(data['observed_initial'])))
    weights=np.exp(-3*(d/np.maximum(d[:,-1,None],1e-9))**2)/np.maximum(d,1e-6);weights/=weights.sum(1,keepdims=True)
    pred=initial[None]+np.einsum('pk,tpkc->tpc',weights,displacement[:,ids]);pred[0]=initial_reference
    return pred
