"""Recoverable split-HMC warmup and retained prefixes with exact RNG continuity."""
from dataclasses import asdict,replace
import json
import math
from pathlib import Path
import torch
from .split_hmc import sample_split_hmc,split_trajectory


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    temporary.replace(path)


def sample_with_warmup_checkpoints(evaluate,dimension,config,state_path,initial=None,progress=None,checkpoint_hook=None):
    """Original split-HMC outer loop with persisted dual-averaging state.

    No retained samples are published until this call finishes. The final
    warmup state remains resumable if interruption occurs during first draws.
    """
    config.validate();state_path=Path(state_path)
    rng=torch.Generator().manual_seed(config.seed)
    epsilon=config.step_size;log_epsilon=math.log(epsilon)
    mu,averaged,hbar=math.log(10*epsilon),log_epsilon,0.
    counts={phase:{k:[0]*config.chains for k in ['accepted','invalid','divergent']} for phase in ['warmup','retained']}
    previous=torch.load(state_path,weights_only=True) if state_path.exists() else None
    if previous is not None:
        if previous['config']!=asdict(config) or previous['dimension']!=dimension:
            raise ValueError('Warmup checkpoint cannot change the algorithm or requested first block')
        q=previous['position'].clone();start=previous['next_iteration']
        if not 0<=start<=config.warmup:raise ValueError('Invalid warmup checkpoint boundary')
        epsilon,log_epsilon=previous['epsilon'],previous['log_epsilon']
        mu,averaged,hbar=previous['mu'],previous['averaged'],previous['hbar']
        counts=previous['counts'];rng.set_state(previous['rng_state'])
        evaluations=previous['gradient_evaluations']+1
    else:
        q=torch.randn(config.chains,dimension,dtype=torch.float64,generator=rng) if initial is None else initial.detach().clone()
        start=0;evaluations=1
    if q.shape!=(config.chains,dimension):raise ValueError('Initial chains have wrong dimensions')
    ratio,gradient,invalid=evaluate(q)
    if hasattr(evaluate,'full_log_ratio'):
        ratio,full_invalid=evaluate.full_log_ratio(q);invalid|=full_invalid
    if bool(invalid.any()):raise ValueError('Invalid warmup chain state')
    if previous is not None and float((ratio-previous['log_ratio']).abs().max())>1e-6:
        raise ValueError('Warmup exact target no longer reproduces')
    saved=torch.empty(config.chains,config.draws,dimension,dtype=q.dtype)
    ratios=torch.empty(config.chains,config.draws,dtype=q.dtype);energy_errors=torch.empty_like(ratios)
    for iteration in range(start,config.warmup+config.draws):
        duration=config.min_time+(config.max_time-config.min_time)*float(torch.rand((),generator=rng))
        steps=max(1,round(duration/epsilon));p=torch.randn(q.shape,dtype=q.dtype,generator=rng)
        candidate,final_p,new_ratio,new_gradient,bad=split_trajectory(q,p,ratio,gradient,evaluate,epsilon,steps)
        evaluations+=steps
        if hasattr(evaluate,'full_log_ratio'):
            new_ratio,full_invalid=evaluate.full_log_ratio(candidate);bad|=full_invalid
        delta=(-new_ratio+.5*(candidate.square().sum(1)+final_p.square().sum(1))
            +ratio-.5*(q.square().sum(1)+p.square().sum(1)))
        bad|=~torch.isfinite(delta)
        probability=torch.where(bad,0.,torch.exp(-delta.clamp_min(0)))
        accept=torch.rand(config.chains,dtype=q.dtype,generator=rng)<probability
        q[accept],ratio[accept],gradient[accept]=candidate[accept],new_ratio[accept],new_gradient[accept]
        phase='warmup' if iteration<config.warmup else 'retained'
        for key,mask in [('accepted',accept),('invalid',bad),('divergent',bad|(delta.abs()>100))]:
            counts[phase][key]=[a+int(b) for a,b in zip(counts[phase][key],mask)]
        if iteration<config.warmup:
            t=iteration+1;rate=float(probability.mean())
            hbar=(1-1/(t+10))*hbar+(config.target_accept-rate)/(t+10)
            log_epsilon=mu-math.sqrt(t)/.05*hbar
            log_epsilon=min(math.log(.8),max(math.log(.015),log_epsilon))
            averaged=t**(-.75)*log_epsilon+(1-t**(-.75))*averaged
            epsilon=math.exp(averaged if t==config.warmup else log_epsilon)
            if t%25==0 or t==config.warmup:
                state={'config':asdict(config),'dimension':dimension,'position':q.clone(),
                    'log_ratio':ratio.clone(),'next_iteration':t,'epsilon':epsilon,'log_epsilon':log_epsilon,
                    'mu':mu,'averaged':averaged,'hbar':hbar,'counts':counts,
                    'gradient_evaluations':evaluations,'rng_state':rng.get_state()}
                temporary=state_path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(state_path)
                if checkpoint_hook is not None:checkpoint_hook(t)
        else:
            i=iteration-config.warmup
            saved[:,i],ratios[:,i],energy_errors[:,i]=q,ratio,torch.where(bad,torch.inf,delta)
        if progress and ((iteration+1)%25==0 or iteration+1==config.warmup+config.draws):
            progress({'transition':iteration+1,'total':config.warmup+config.draws,
                'step_size':epsilon,'steps':steps,'counts':counts[phase]})
    return {'white':saved,'log_ratio':ratios,'energy_errors':energy_errors,
        'config':asdict(config),'step_size':epsilon,'counts':counts,
        'gradient_evaluations':evaluations,'rng_state':rng.get_state()}


def sample_in_chunks(evaluate,dimension,config,checkpoint,max_chunk=100,initial=None,progress=None,resume=None):
    if max_chunk<32:raise ValueError('Checkpoint chunks must allow at least 32 retained draws')
    if config.draws<16:raise ValueError('Need at least sixteen new draws')
    checkpoint=Path(checkpoint);checkpoint.mkdir(parents=True,exist_ok=True)
    remaining=config.draws;current=resume
    requested_total=(0 if resume is None else resume['white'].shape[1])+remaining
    while remaining:
        count=min(16 if current is None else max_chunk,remaining)
        if 0<remaining-count<16:
            count=remaining if remaining<=max_chunk else remaining-16
        call=replace(config,draws=count,warmup=config.warmup if current is None else 0,
            step_size=config.step_size if current is None else current['step_size'])
        if current is None:
            current=sample_with_warmup_checkpoints(evaluate,dimension,call,checkpoint/'warmup_state.pt',initial,progress)
        else:
            current=sample_split_hmc(evaluate,dimension,call,progress=progress,resume=current)
        temporary=checkpoint/'chains.tmp.pt'
        torch.save(current,temporary);temporary.replace(checkpoint/'chains.pt')
        (checkpoint/'warmup_state.pt').unlink(missing_ok=True)
        _write(checkpoint/'checkpoint_progress.json',{
            'retained_draws_per_chain':current['white'].shape[1],
            'requested_current_stage_total':requested_total,'frozen_step_size':current['step_size'],
            'gradient_batches':current['gradient_evaluations'],
            'qualification':'same frozen kernel/RNG; original diagnostic budgets and stopping gates unchanged'})
        remaining-=count
    return current


