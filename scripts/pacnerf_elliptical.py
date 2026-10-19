"""Recoverable blocked elliptical slice sampling in a fixed Gaussian reference.

Murray, Adams & MacKay (2010), https://arxiv.org/abs/1001.0175.
Target is N(w;0,I)*exp(log_ratio(w)); no likelihood approximation or gradients.
Each block update uses the exact nonlinear posterior/reference ratio. A bracket
cap raises an error; it never silently accepts or retains a failed proposal.
"""
import math
from pathlib import Path
import torch
from pacnerf_comparison import write


@torch.no_grad()
def sweep(evaluate,position,ratio,rng,blocks,max_bracket=1000):
    q=position.clone();current=ratio.clone();evaluations=0
    attempts=torch.zeros(len(q),dtype=torch.long);invalid_proposals=torch.zeros_like(attempts)
    for block in blocks:
        index=block['indices'] if isinstance(block,dict) else slice(*block)
        original=q[:,index].clone()
        auxiliary=torch.randn(original.shape,dtype=q.dtype,generator=rng)
        threshold=current+torch.log(torch.rand(len(q),dtype=q.dtype,generator=rng))
        angle=2*math.pi*torch.rand(len(q),dtype=q.dtype,generator=rng)
        lower=angle-2*math.pi;upper=angle.clone()
        pending=torch.ones(len(q),dtype=torch.bool)
        for _ in range(max_bracket):
            candidate=q.clone()
            proposed=original*torch.cos(angle)[:,None]+auxiliary*torch.sin(angle)[:,None]
            candidate[:,index]=torch.where(pending[:,None],proposed,q[:,index])
            value,bad=evaluate(candidate);evaluations+=1
            attempts+=pending.long();invalid_proposals+=(pending&bad).long()
            accepted=pending&(~bad)&(value>=threshold)
            q[accepted]=candidate[accepted];current[accepted]=value[accepted]
            pending&=~accepted
            if not bool(pending.any()):break
            lower=torch.where(pending&(angle<0),angle,lower)
            upper=torch.where(pending&(angle>=0),angle,upper)
            angle=lower+(upper-lower)*torch.rand(len(q),dtype=q.dtype,generator=rng)
        else:raise ValueError('Elliptical bracket did not close; sampling stopped without accepting a failed proposal')
    return q,current,evaluations,attempts,invalid_proposals


def atomic_save(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp.pt');torch.save(value,temporary);temporary.replace(path)


@torch.no_grad()
def sample(evaluate,dimension,checkpoint,total_draws,warmup=500,seed=240001,chunk=50,progress=None,blocks=None,initial_scale=1.):
    if total_draws<16 or warmup<0:raise ValueError('Invalid sampling budget')
    if not math.isfinite(initial_scale) or initial_scale<=0:raise ValueError('Invalid initial reference scale')
    checkpoint=Path(checkpoint);checkpoint.mkdir(parents=True,exist_ok=True)
    blocks=[(0,dimension-3),(dimension-3,dimension)] if blocks is None else blocks
    flattened=[i for block in blocks for i in (block['indices'] if isinstance(block,dict) else range(*block))]
    if sorted(flattened)!=list(range(dimension)):
        raise ValueError('Fixed slice blocks must cover each coordinate exactly once')
    protocol={'sampler':'blocked-elliptical-reference-v1','dimension':dimension,'chains':4,'warmup':warmup,'seed':seed,'blocks':blocks,'initial_scale':initial_scale}
    protocol={**protocol,'blocks':[{'indices':list(b['indices'])} if isinstance(b,dict) else list(b) for b in blocks]}
    state_path=checkpoint/'chains.pt';warm_path=checkpoint/'elliptical_warmup.pt'
    previous=torch.load(state_path,weights_only=True) if state_path.exists() else torch.load(warm_path,weights_only=True) if warm_path.exists() else None
    rng=torch.Generator().manual_seed(seed)
    white=torch.empty(4,total_draws,dimension,dtype=torch.float64)
    ratios=torch.empty(4,total_draws,dtype=torch.float64)
    counts={p:{k:[0]*4 for k in ['accepted','invalid','divergent']} for p in ['warmup','retained']}
    if previous is None:
        q=initial_scale*torch.randn(4,dimension,dtype=torch.float64,generator=rng)
        current,bad=evaluate(q);evaluations=1;warm=0;retained=0
        proposals=[0]*4;invalid=[0]*4
    else:
        if previous['protocol']!=protocol:raise ValueError('Frozen elliptical sampling protocol changed')
        q=previous['position'].clone();current,bad=evaluate(q)
        if float((current-previous['position_log_ratio']).abs().max())>1e-6:raise ValueError('Saved slice target no longer reproduces')
        evaluations=previous['target_evaluations']+1;warm=previous['completed_warmup'];retained=previous.get('retained',0)
        counts=previous['counts'];proposals=previous['proposal_attempts'];invalid=previous['invalid_proposals']
        if retained>total_draws:raise ValueError('Cannot shrink saved chain')
        if retained:white[:,:retained]=previous['white'];ratios[:,:retained]=previous['log_ratio']
        rng.set_state(previous['rng_state'])
    if bool(bad.any()):raise ValueError('Invalid elliptical initial or resumed state')
    def state(include_draws):
        value={'protocol':protocol,'position':q.clone(),'position_log_ratio':current.clone(),
            'completed_warmup':warm,'retained':retained,'rng_state':rng.get_state(),
            'target_evaluations':evaluations,'counts':counts,'proposal_attempts':proposals,'invalid_proposals':invalid}
        if include_draws:value.update({'white':white[:,:retained].clone(),'log_ratio':ratios[:,:retained].clone()})
        return value
    while warm<warmup or retained<total_draws:
        q,current,used,tries,bad_proposals=sweep(evaluate,q,current,rng,blocks)
        evaluations+=used;proposals=[a+int(b) for a,b in zip(proposals,tries)]
        invalid=[a+int(b) for a,b in zip(invalid,bad_proposals)]
        phase='warmup' if warm<warmup else 'retained'
        counts[phase]['accepted']=[v+1 for v in counts[phase]['accepted']]
        if phase=='warmup':
            warm+=1
            if warm%25==0 or warm==warmup:atomic_save(warm_path,state(False))
        else:
            white[:,retained]=q;ratios[:,retained]=current;retained+=1
            if retained==16 or retained%chunk==0 or retained==total_draws:
                atomic_save(state_path,state(True));warm_path.unlink(missing_ok=True)
                write(checkpoint/'checkpoint_progress.json',{'sampler':protocol['sampler'],
                    'retained_draws_per_chain':retained,'requested_current_stage_total':total_draws,'target_evaluations':evaluations})
        if progress and (warm+retained)%25==0:
            progress({'warmup':warm,'retained':retained,'requested':total_draws,'target_evaluations':evaluations,'proposal_attempts':proposals,'invalid_proposals':invalid})
    return state(True)
