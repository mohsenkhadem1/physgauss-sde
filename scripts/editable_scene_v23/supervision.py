"""Mixed-supervision contract: missing labels do not become fabricated targets."""
import numpy as np
import torch

OBSERVATIONS=('rgb','depth','tracks','camera_poses')
ACTIONS=('tool_positions','applied_forces')

def causal_example(sequence,cut):
    """Action horizon is known; visual/physical observations end at cut.

    Return evaluator/training targets separately. Do not pass targets to encoder.
    Camera trajectory after cut is a render query, not an observed visual input.
    """
    if not isinstance(cut,int) or cut<1:raise ValueError('Positive prefix length required')
    observations={k:np.array(sequence[k][:cut],copy=True)for k in OBSERVATIONS if k in sequence}
    if not observations or any(len(x)!=cut for x in observations.values()):raise ValueError('Incomplete prefix')
    actions={k:np.array(sequence[k],copy=True)for k in ACTIONS if k in sequence}
    targets={k:np.array(sequence[k][cut:],copy=True)for k in OBSERVATIONS if k in sequence}
    return dict(observations=observations,actions=actions,cut=cut),targets

def available_loss(predictions,targets,weights=None):
    """Finite-label masked MSE prototype, with per-modality normalization.

    Images/geometry must be normalized by the caller; this is not a calibrated
    likelihood, a complete photometric objective, or permission to mix SI units.
    """
    weights={}if weights is None else weights;terms={}
    for key,target in targets.items():
        if key not in predictions:continue
        pred=predictions[key]
        if pred.shape!=target.shape:raise ValueError('Supervision shape mismatch')
        valid=torch.isfinite(target)
        if valid.any():
            if not torch.isfinite(pred[valid]).all():raise ValueError('Nonfinite supervised prediction')
            terms[key]=(pred[valid]-target[valid]).square().mean()*weights.get(key,1.)
    if not terms:raise ValueError('No available supervision')
    return sum(terms.values()),terms
