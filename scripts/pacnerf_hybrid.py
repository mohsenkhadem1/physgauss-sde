"""Exact conditional state HMC and five scalar parameter elliptical slices.

Only material/noise parameters are frozen DURING the conditional state step.
Their cached operators are constants in that partial derivative. State-dependent
neural covariance, contact, full images and feasibility remain exact. Every HMC
acceptance uses the full no-gradient posterior, not a surrogate target.
"""
import math
import hashlib
from pathlib import Path
import torch
from pacnerf_elliptical import sweep,atomic_save
from physgauss.split_hmc import split_trajectory
from pacnerf_comparison import write


def conditional_reference_transform(precision,core):
    d=len(precision);state=torch.arange(2,core);psi=torch.tensor([0,1,*range(core,d)])
    pss=precision[state[:,None],state[None,:]]
    cross=precision[state[:,None],psi[None,:]]
    ls=torch.linalg.cholesky(pss);regression=-torch.cholesky_solve(cross,ls)
    schur=precision[psi[:,None],psi[None,:]]+cross.T@regression
    lp=torch.linalg.cholesky((schur+schur.T)/2)
    ap=torch.linalg.solve_triangular(lp.T,torch.eye(len(psi),dtype=precision.dtype),upper=True)
    transform=torch.zeros_like(precision)
    transform[psi[:,None],psi[None,:]]=ap
    transform[state[:,None],psi[None,:]]=regression@ap
    transform[state[:,None],state[None,:]]=torch.linalg.solve_triangular(ls.T,torch.eye(len(state),dtype=precision.dtype),upper=True)
    if float((transform.T@precision@transform-torch.eye(d,dtype=precision.dtype)).abs().max())>1e-7:
        raise ValueError('Conditional Gaussian reference does not whiten its precision')
    return transform


class ConditionalStateEvaluator:
    def __init__(self,target,fixed,core):
        self.target=target;self.fixed=fixed.detach().clone();self.core=core
        self.model=target.model.model
        psi=[0,1,*range(core,fixed.shape[1])]
        if bool((target.transform[psi,2:core]!=0).any()):raise ValueError('State reference coordinates must preserve physical parameters')
        with torch.no_grad():
            native=target.coordinates(fixed)
            self.context={'theta':native[:,:2].clone(),'hyper':native[:,-3:].clone(),
                'transition':self.model.transition(native,self.model.dt),
                'integrals':self.model.integration_operators(native),'scales':self.model.scales(native),
                'gaussian':target.model.gaussian(native)}

    @torch.enable_grad()
    def __call__(self,state):
        variable=state.detach().requires_grad_();full=self.fixed.clone();full[:,2:self.core]=variable
        try:
            if self.model._conditional_context is not None:raise ValueError('Nested parameter freezing is not supported')
            self.model._conditional_context=self.context
            try:
                ratio=-self.target.model.objective(self.target.coordinates(full))+self.target.center+.5*full.square().sum(-1)
                gradient,=torch.autograd.grad(ratio.sum(),variable)
            finally:self.model._conditional_context=None
            bad=~torch.isfinite(ratio)|~torch.isfinite(gradient).all(-1)
            return ratio.detach(),torch.nan_to_num(gradient.detach()),bad
        except torch.linalg.LinAlgError:
            # Evaluate rows separately without freezing; a numerically invalid
            # row cannot invalidate another chain's otherwise valid proposal.
            value,gradient,bad=self.target(full.detach())
            return value,gradient[:,2:self.core],bad


class GenericStateEvaluator:
    def __init__(self,target,fixed,core):self.target=target;self.fixed=fixed.clone();self.core=core
    @torch.enable_grad()
    def __call__(self,state):
        full=self.fixed.clone();full[:,2:self.core]=state
        value,gradient,bad=self.target(full)
        return value,gradient[:,2:self.core],bad


@torch.no_grad()
def sample(target,dimension,core,checkpoint,total_draws,warmup=500,seed=240001,chunk=50,
        min_time=.6,max_time=1.8,target_accept=.9,progress=None,checkpoint_hook=None,initial=None):
    if total_draws<16 or not 2<core<=dimension or warmup<0:raise ValueError('Invalid hybrid dimensions or sampling budget')
    checkpoint=Path(checkpoint);checkpoint.mkdir(parents=True,exist_ok=True)
    path=checkpoint/'chains.pt';warm_path=checkpoint/'hybrid_warmup.pt'
    blocks=[(0,1),(1,2)]+[(i,i+1) for i in range(core,dimension)]
    protocol={'sampler':'conditional-state-hmc-parameter-ess-v1','dimension':dimension,'core':core,
        'chains':4,'warmup':warmup,'seed':seed,'initial_reference_scale':1.,'min_time':min_time,
        'max_time':max_time,'target_accept':target_accept,'parameter_blocks':[list(b) for b in blocks]}
    if initial is not None and initial.shape!=(4,dimension):raise ValueError('Hybrid initial states have wrong shape')
    protocol['initial_state_sha256']=None if initial is None else hashlib.sha256(initial.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    previous=torch.load(path,weights_only=True) if path.exists() else torch.load(warm_path,weights_only=True) if warm_path.exists() else None
    rng=torch.Generator().manual_seed(seed)
    white=torch.empty(4,total_draws,dimension,dtype=torch.float64);ratios=torch.empty(4,total_draws,dtype=torch.float64)
    errors=torch.empty_like(ratios)
    if previous is None:
        q=torch.randn(4,dimension,dtype=torch.float64,generator=rng) if initial is None else initial.detach().clone()
        warm=0;retained=0;epsilon=.3
        log_epsilon=math.log(epsilon);mu=math.log(10*epsilon);averaged=log_epsilon;hbar=0.
        counts={p:{k:[0]*4 for k in ['accepted','invalid','divergent']} for p in ['warmup','retained']}
        evaluations=0;gradients=0;slice_attempts=[0]*4;slice_invalid=[0]*4
    else:
        if previous['protocol']!=protocol:raise ValueError('Frozen hybrid sampling protocol changed')
        q=previous['position'].clone();warm=previous['completed_warmup'];retained=previous['retained']
        epsilon=previous['step_size'];log_epsilon=previous['log_epsilon'];mu=previous['mu'];averaged=previous['averaged'];hbar=previous['hbar']
        counts=previous['counts'];evaluations=previous['target_evaluations'];gradients=previous['gradient_evaluations']
        slice_attempts=previous['parameter_slice_attempts'];slice_invalid=previous['parameter_slice_invalid_proposals']
        if retained>total_draws:raise ValueError('Cannot shrink saved hybrid chains')
        if retained:
            white[:,:retained]=previous['white'];ratios[:,:retained]=previous['log_ratio'];errors[:,:retained]=previous['energy_errors']
        rng.set_state(previous['rng_state'])
    current,bad=target.log_ratio_no_grad(q);evaluations+=1
    if bool(bad.any()):raise ValueError('Invalid initial/resumed hybrid state')
    if previous is not None and float((current-previous['position_log_ratio']).abs().max())>1e-6:raise ValueError('Hybrid cached posterior no longer reproduces')
    def state(include):
        result={'protocol':protocol,'position':q.clone(),'position_log_ratio':current.clone(),
            'completed_warmup':warm,'retained':retained,'rng_state':rng.get_state(),'step_size':epsilon,
            'log_epsilon':log_epsilon,'mu':mu,'averaged':averaged,'hbar':hbar,'counts':counts,
            'target_evaluations':evaluations,'gradient_evaluations':gradients,
            'parameter_slice_attempts':slice_attempts,'parameter_slice_invalid_proposals':slice_invalid}
        if include:result.update({'white':white[:,:retained].clone(),'log_ratio':ratios[:,:retained].clone(),'energy_errors':errors[:,:retained].clone()})
        return result
    while warm<warmup or retained<total_draws:
        evaluate=target.state_target(q) if hasattr(target,'state_target') else GenericStateEvaluator(target,q,core)
        _,gradient,bad=evaluate(q[:,2:core]);gradients+=1;evaluations+=1
        if bool(bad.any()):raise ValueError('Invalid conditional state gradient')
        duration=min_time+(max_time-min_time)*float(torch.rand((),generator=rng));steps=max(1,round(duration/epsilon))
        momentum=torch.randn(gradient.shape,dtype=q.dtype,generator=rng)
        proposed,final_p,_,_,invalid=split_trajectory(q[:,2:core],momentum,current,gradient,evaluate,epsilon,steps)
        gradients+=steps;evaluations+=steps
        candidate=q.clone();candidate[:,2:core]=proposed
        proposed_ratio,invalid_full=target.log_ratio_no_grad(candidate);evaluations+=1;invalid|=invalid_full
        delta=-proposed_ratio+.5*(proposed.square().sum(-1)+final_p.square().sum(-1))+current-.5*(q[:,2:core].square().sum(-1)+momentum.square().sum(-1))
        invalid|=~torch.isfinite(delta);probability=torch.where(invalid,0.,torch.exp(-delta.clamp_min(0)))
        accept=torch.rand(4,dtype=q.dtype,generator=rng)<probability
        q[accept]=candidate[accept];current[accept]=proposed_ratio[accept]
        phase='warmup' if warm<warmup else 'retained'
        for key,mask in [('accepted',accept),('invalid',invalid),('divergent',invalid|(delta.abs()>100))]:
            counts[phase][key]=[a+int(b) for a,b in zip(counts[phase][key],mask)]
        q,current,used,attempts,invalid_slice=sweep(target.log_ratio_no_grad,q,current,rng,blocks)
        evaluations+=used;slice_attempts=[a+int(b) for a,b in zip(slice_attempts,attempts)]
        slice_invalid=[a+int(b) for a,b in zip(slice_invalid,invalid_slice)]
        if phase=='warmup':
            warm+=1;rate=float(probability.mean())
            hbar=(1-1/(warm+10))*hbar+(target_accept-rate)/(warm+10)
            log_epsilon=min(math.log(.8),max(math.log(.015),mu-math.sqrt(warm)/.05*hbar))
            averaged=warm**(-.75)*log_epsilon+(1-warm**(-.75))*averaged
            epsilon=math.exp(averaged if warm==warmup else log_epsilon)
            if warm%25==0 or warm==warmup:
                atomic_save(warm_path,state(False))
                if checkpoint_hook is not None:checkpoint_hook(warm)
        else:
            white[:,retained]=q;ratios[:,retained]=current;errors[:,retained]=torch.where(invalid,torch.inf,delta);retained+=1
            if retained==16 or retained%chunk==0 or retained==total_draws:
                atomic_save(path,state(True));warm_path.unlink(missing_ok=True)
                write(checkpoint/'checkpoint_progress.json',{'sampler':protocol['sampler'],'retained_draws_per_chain':retained,
                    'requested_current_stage_total':total_draws,'target_evaluations':evaluations,'gradient_evaluations':gradients})
        if progress and (warm+retained)%25==0:
            progress({'warmup':warm,'retained':retained,'requested':total_draws,'step_size':epsilon,
                'state_steps':steps,'state_counts':counts[phase],'parameter_slice_attempts':slice_attempts})
    return state(True)
