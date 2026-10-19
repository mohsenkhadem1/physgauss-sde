"""Learn a fixed sampling reference from discarded observed-data warmup.

The fitted guide/mode and physical target remain unchanged. The regularized
covariance and centre affect only the affine sampling coordinates. No learning
occurs after the final sampler starts retaining posterior draws.
"""
import copy
import torch
from pacnerf_hybrid import conditional_reference_transform


@torch.no_grad()
def adapt_reference(reference,target,preconditioning,core,shrinkage=.5):
    if not 0<shrinkage<=1:raise ValueError('Invalid reference covariance shrinkage')
    white=preconditioning['white'].flatten(0,1)
    if len(white)<16:raise ValueError('Insufficient reference-learning states')
    average=white.mean(0)
    covariance=(1-shrinkage)*torch.cov(white.T)+shrinkage*torch.eye(white.shape[1],dtype=white.dtype)
    native_covariance=target.transform@covariance@target.transform.T
    native_covariance=(native_covariance+native_covariance.T)/2
    result=copy.deepcopy(reference)
    result['sampling_mean']=target.coordinates(average[None])[0]
    result['sampling_precision']=torch.cholesky_inverse(torch.linalg.cholesky(native_covariance))
    transform=conditional_reference_transform(result['sampling_precision'],core)
    previous_native=target.coordinates(preconditioning['position'])
    initial=torch.linalg.solve(transform,(previous_native-result['sampling_mean']).T).T
    error=float((result['sampling_mean']+initial@transform.T-previous_native).abs().max())
    if error>1e-8:raise ValueError('Reference learning changed the initial physical/native states')
    result['initial_white']=initial
    result['reference_learning']={'discarded_draws':len(white),'warmup_per_chain':preconditioning['completed_warmup'],
        'draws_per_chain':preconditioning['white'].shape[1],'shrinkage':shrinkage,
        'maximum_native_state_preservation_error':error,'future_geometry_used':False,
        'scope':'Proposal-only mean/covariance from discarded preconditioning states; fitted-mode stationarity refers to mean/precision, not sampling_mean/sampling_precision; physical posterior unchanged'}
    result['fit']['reference_learning']=result['reference_learning']
    return result
