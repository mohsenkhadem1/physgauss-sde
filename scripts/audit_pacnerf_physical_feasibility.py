"""Checkpoint-only floor/speed diagnostic; never reads future particle truth."""
import argparse,hashlib,json
from pathlib import Path
import torch
from pacnerf_discrepancy import make_model
from pacnerf_conditioned_transport import CachedPhysicalModel,make_target
from pacnerf_floor import make_floor_model
from pacnerf_comparison import write


@torch.no_grad()
def audit(inputs,checkpoint,output,index=8,legacy=False):
    if legacy:
        base,_=make_model(inputs,index);model=CachedPhysicalModel(base)
    else:model,_=make_floor_model(inputs,index)
    checkpoint=Path(checkpoint);reference=torch.load(checkpoint/'reference.pt',weights_only=True)
    chains=torch.load(checkpoint/'chains.pt',weights_only=True);target=make_target(model,reference)
    z=target.decode(chains['white'].flatten(0,1));q,v=model.decode(z)
    heights=model.data['rest'][None,None,:,1]+torch.einsum('btm,nm->btn',q,model.basis[:,1])
    gaussian=model.data['gaussians'][None,None,:,1]+torch.einsum('btm,gm->btg',q,model.renderer.gaussian_basis[:,1])
    minimum=heights.amin((1,2));gapminimum=heights[:,4:10].amin((1,2))
    velocity=torch.einsum('btm,ncm->btnc',v,model.basis)
    result={'future_geometry_used':False,'draws_per_chain':chains['white'].shape[1],
        'case_index':index,'physical_target':'hierarchical-v1' if legacy else 'floor-conditioned-v2',
        'reference_sha256':hashlib.sha256((checkpoint/'reference.pt').read_bytes()).hexdigest(),
        'chains_sha256':hashlib.sha256((checkpoint/'chains.pt').read_bytes()).hexdigest(),
        'minimum_initial_handle_y_m':float(model.data['rest'][:,1].min()),
        'minimum_handle_height_quantiles_m':torch.quantile(minimum,minimum.new_tensor([0,.01,.05,.5,.95,1])).tolist(),
        'gap_minimum_handle_height_quantiles_m':torch.quantile(gapminimum,gapminimum.new_tensor([0,.01,.05,.5,.95,1])).tolist(),
        'point_time_below_floor_fraction':{str(x):float((heights<x).double().mean()) for x in [0.,-.02,-.05,-.1]},
        'gaussian_center_point_time_below_floor_fraction':{str(x):float((gaussian<x).double().mean()) for x in [0.,-.02,-.05,-.1]},
        'maximum_handle_speed_quantiles_m_s':torch.quantile(velocity.norm(dim=-1).amax((1,2)),minimum.new_tensor([0,.5,.95,1])).tolist(),
        'qualification':'Descriptive finite-checkpoint diagnostic only; convergence must be assessed separately. No evaluation ground-truth access.'}
    write(output,result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['inputs','checkpoint','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--case-index',type=int,default=8);p.add_argument('--legacy-v1',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    audit(a.inputs,a.checkpoint,a.output,a.case_index,a.legacy_v1)
