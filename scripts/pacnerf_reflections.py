"""Gaussian Hamiltonian reflections and exact logistic-event slice augmentation.

Harmonic reflections follow Pakman & Paninski, arXiv:1208.4118. The extra
logistic slice variables retain the existing SOFT event after integration;
these are sampling walls, not a change to the object's physical contact model.
"""
import math
import torch


def logistic_slice_threshold(height,scale,rng):
    if scale<=0:raise ValueError('Positive logistic event scale required')
    log_u=torch.log(torch.rand(height.shape,dtype=height.dtype,device=height.device,generator=rng).clamp_min(torch.finfo(height.dtype).tiny))
    log_u=log_u-torch.nn.functional.softplus(-height/scale)
    return scale*(log_u-torch.log(-torch.expm1(log_u)))


def exit_times(q,p,normals,offset):
    a=torch.einsum('bkd,bd->bk',normals,q)
    b=torch.einsum('bkd,bd->bk',normals,p)
    radius=torch.hypot(a,b)
    reachable=radius>offset.abs()
    angle=torch.atan2(b,a)+torch.acos((-offset/radius.clamp_min(torch.finfo(q.dtype).tiny)).clamp(-1,1))
    result=torch.remainder(angle,2*math.pi)
    # An inward/grazing velocity cannot immediately hit the same wall again.
    result=torch.where((result<1e-13)&(b>=0),2*math.pi,result)
    return torch.where(reachable,result,torch.inf)


@torch.no_grad()
def harmonic_reflections(position,momentum,normals,offset,duration,max_reflections=4096):
    q=position.clone();p=momentum.clone();batch=len(q)
    remaining=torch.full((batch,),float(duration),dtype=q.dtype,device=q.device)
    counts=torch.zeros(batch,dtype=torch.long,device=q.device)
    bad=(torch.einsum('bkd,bd->bk',normals,q)+offset < -1e-10).any(-1)
    remaining[bad]=0
    rows=torch.arange(batch,device=q.device)
    while bool((remaining>1e-13).any()):
        active=remaining>1e-13
        hit_time,index=exit_times(q,p,normals,offset).min(-1)
        dt=torch.where(active,torch.minimum(hit_time,remaining),0.)
        cosine=dt.cos()[:,None];sine=dt.sin()[:,None]
        q,p=cosine*q+sine*p,cosine*p-sine*q
        hit=active&(hit_time<=remaining+1e-13)
        normal=normals[rows,index]
        projection=(normal*p).sum(-1)/normal.square().sum(-1).clamp_min(torch.finfo(q.dtype).tiny)
        p=torch.where(hit[:,None],p-2*projection[:,None]*normal,p)
        remaining=(remaining-dt).clamp_min(0)
        counts+=hit.long()
        bad|=counts>max_reflections
        remaining[bad]=0
    bad|=~torch.isfinite(q).all(-1)|~torch.isfinite(p).all(-1)
    bad|=(torch.einsum('bkd,bd->bk',normals,q)+offset < -1e-9).any(-1)
    return q,p,bad,counts


@torch.no_grad()
def reflected_trajectory(q,p,gradient,evaluate,normals,offset,epsilon,steps):
    q=q.clone();p=p.clone();g=gradient
    bad=torch.zeros(len(q),dtype=torch.bool,device=q.device)
    reflections=torch.zeros(len(q),dtype=torch.long,device=q.device)
    for _ in range(steps):
        p+=.5*epsilon*g
        q,p,invalid,count=harmonic_reflections(q,p,normals,offset,epsilon)
        bad|=invalid;reflections+=count
        value,g,invalid=evaluate(q)
        bad|=invalid
        p+=.5*epsilon*g
    return q,-p,value,g,bad,reflections


@torch.no_grad()
def state_transition(q,evaluate,normals,height_offset,scale,epsilon,steps,rng):
    height=torch.einsum('bkd,bd->bk',normals,q)+height_offset
    threshold=logistic_slice_threshold(height,scale,rng)
    offset=height_offset-threshold
    current,gradient,bad=evaluate(q)
    if bool(bad.any()):raise ValueError('Invalid current reflected state')
    p=torch.randn(q.shape,dtype=q.dtype,device=q.device,generator=rng)
    candidate,final_p,value,_,invalid,reflections=reflected_trajectory(q,p,gradient,evaluate,normals,offset,epsilon,steps)
    delta=-value+.5*(candidate.square().sum(-1)+final_p.square().sum(-1))+current-.5*(q.square().sum(-1)+p.square().sum(-1))
    invalid|=~torch.isfinite(delta)
    probability=torch.where(invalid,0.,torch.exp(-delta.clamp_min(0)))
    accepted=torch.rand(len(q),dtype=q.dtype,device=q.device,generator=rng)<probability
    return torch.where(accepted[:,None],candidate,q),{'accepted':accepted,'invalid':invalid,
        'divergent':invalid|(delta.abs()>100),'energy_error':delta,'probability':probability,'reflections':reflections}
