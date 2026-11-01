"""Same Monte Carlo predictive-risk criterion without a quadratic particle Gram.

Computes posterior target moments directly. This permits finer quadrature without
turning the method into a coarse-particle acquisition heuristic for speed.
"""
import numpy as np
from experiments.gaussian_active_design import normalized_weights,whitened,predictive_risk


def score_query(targets,predictions,covariance,weights,*,fantasies=128,seed=0):
    weights=normalized_weights(weights,len(predictions))
    if fantasies<2:raise ValueError('At least two fantasy observations required')
    targets=np.asarray(targets,dtype=float).reshape(len(weights),-1)
    if not targets.shape[1] or not np.isfinite(targets).all():raise ValueError('Finite design response required')
    white,_=whitened(predictions,covariance)
    white=white-weights@white
    rng=np.random.default_rng(seed)
    ancestors=rng.choice(len(weights),size=fantasies,p=weights)
    observed=white[ancestors]+rng.normal(size=(fantasies,white.shape[1]))
    distance=np.maximum(np.sum(observed**2,axis=1)[:,None]+np.sum(white**2,axis=1)[None]-2*observed@white.T,0.)
    logprior=np.full(len(weights),-np.inf);positive=weights>0;logprior[positive]=np.log(weights[positive])
    logw=logprior[None]-.5*distance
    posterior=np.exp(logw-logw.max(axis=1,keepdims=True));posterior/=posterior.sum(axis=1,keepdims=True)
    centered=targets-weights@targets
    second=np.mean(centered**2,axis=1)
    postmean=posterior@centered
    after=np.maximum(posterior@second-np.mean(postmean**2,axis=1),0.)
    before=predictive_risk(targets,weights)
    gains=before-after
    return dict(expected_risk_reduction=float(gains.mean()),score_standard_error=float(gains.std(ddof=1)/np.sqrt(fantasies)),
                expected_effective_sample_size=float(np.mean(1/np.sum(posterior**2,axis=1))),
                fantasies=fantasies,seed=seed,risk_before=before)
