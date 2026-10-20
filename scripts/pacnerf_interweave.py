"""Exact material/noise update while holding the physical trajectory fixed.

The other parameter update holds Gaussian-bridge innovations fixed. Alternating
the two conditionals can break state/scale coupling. Here the density is in
physical coordinates: the bridge Jacobian must NOT appear in this conditional.
The five-dimensional affine reference has a constant Jacobian, which cancels.
"""
import torch
from pacnerf_elliptical import sweep


@torch.no_grad()
def centered_parameter_step(target, white, core, rng):
    guide=target.model; model=guide.model
    indices=torch.tensor([0,1,*range(core,white.shape[-1])])
    if bool((target.transform[indices,2:core]!=0).any()):
        raise ValueError('Physical parameters depend on state reference coordinates')
    physical=target.decode(white)
    affine=target.transform[indices[:,None],indices[None,:]]
    mean=target.mean[indices]
    def evaluate(v):
        candidate=physical.clone();candidate[:,indices]=mean+v@affine.T
        try:
            value=-model.objective(candidate)+.5*v.square().sum(-1)
            return value,~torch.isfinite(value)
        except (ValueError,torch.linalg.LinAlgError):
            values=[];bad=[]
            for row,coordinates in zip(candidate,v):
                try: value=-model.objective(row[None])[0]+.5*coordinates.square().sum()
                except (ValueError,torch.linalg.LinAlgError):value=row.new_tensor(-torch.inf)
                values.append(value);bad.append(~torch.isfinite(value))
            return torch.stack(values),torch.stack(bad)
    v=white[:,indices].clone();ratio,bad=evaluate(v)
    if bool(bad.any()):raise ValueError('Invalid centered conditional initial density')
    v,_,used,attempts,invalid=sweep(evaluate,v,ratio,rng,[{'indices':list(range(len(indices)))}])
    physical[:,indices]=mean+v@affine.T
    native,_=guide.transform(physical,inverse=True)
    result=torch.linalg.solve(target.transform,(native-target.mean).T).T
    value,bad=target.log_ratio_no_grad(result)
    if bool(bad.any()):raise ValueError('Invalid target after centered parameter coordinate change')
    return result,value,used+2,attempts,invalid
