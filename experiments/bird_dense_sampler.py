"""Dense-reference reflected HMC with joint and scalar parameter slice sweeps.

The floor auxiliaries are discarded BEFORE updating parameters. The parameter
step and all saved log densities use the original soft-floor marginal target.
"""
import hashlib,math
from pathlib import Path
import torch
from pacnerf_reflected_hybrid import ReflectedStateContext
from pacnerf_reflections import state_transition
from pacnerf_elliptical import sweep,atomic_save
from pacnerf_comparison import write


@torch.no_grad()
def sample(target,dimension,core,checkpoint,total_draws,warmup=500,seed=940001,chunk=50,
        min_time=1.,max_time=4.,target_accept=.9,initial=None,floor_scale=.002,progress=None,checkpoint_hook=None,chains_count=4,interweave=False,scalar_parameters=True):
    if total_draws<16 or not 2<core<=dimension or warmup<0 or chains_count<1:raise ValueError('Invalid reflected sampling budget')
    root=Path(checkpoint);root.mkdir(parents=True,exist_ok=True)
    path=root/'chains.pt';warm_path=root/'reflected_warmup.pt'
    parameter_indices=[0,1,*range(core,dimension)]
    blocks=[{'indices':parameter_indices}]
    if scalar_parameters:blocks += [{'indices':[i]} for i in parameter_indices]
    if initial is not None and initial.shape!=(chains_count,dimension):raise ValueError('Invalid reflected initial states')
    protocol={'sampler':'dense-reflected-hmc-joint-and-scalar-parameter-ess-v2','dimension':dimension,'core':core,
        'chains':chains_count,'warmup':warmup,'seed':seed,'min_time':min_time,'max_time':max_time,'floor_scale':floor_scale,
        'target_accept':target_accept,'parameter_blocks':blocks,
        'initial_state_sha256':None if initial is None else hashlib.sha256(initial.contiguous().numpy().tobytes()).hexdigest()}
    if interweave:protocol['centered_parameter_interweaving']='exact-physical-density-v1'
    previous=torch.load(path,weights_only=True) if path.exists() else torch.load(warm_path,weights_only=True) if warm_path.exists() else None
    rng=torch.Generator().manual_seed(seed)
    white=torch.empty(chains_count,total_draws,dimension,dtype=torch.float64)
    ratios=torch.empty(chains_count,total_draws,dtype=torch.float64);errors=torch.empty_like(ratios)
    if previous is None:
        q=torch.randn(chains_count,dimension,dtype=torch.float64,generator=rng) if initial is None else initial.clone()
        warm=0;retained=0;epsilon=.3;log_epsilon=math.log(epsilon)
        mu=math.log(10*epsilon);averaged=log_epsilon;hbar=0.
        counts={phase:{key:[0]*chains_count for key in ['accepted','invalid','divergent','reflections']} for phase in ['warmup','retained']}
        evaluations=0;gradients=0;attempts=[0]*chains_count;invalid_slices=[0]*chains_count
    else:
        if previous['protocol']!=protocol:raise ValueError('Reflected sampling protocol changed')
        q=previous['position'].clone();warm=previous['completed_warmup'];retained=previous['retained']
        epsilon=previous['step_size'];log_epsilon=previous['log_epsilon'];mu=previous['mu'];averaged=previous['averaged'];hbar=previous['hbar']
        counts=previous['counts'];evaluations=previous['target_evaluations'];gradients=previous['gradient_evaluations']
        attempts=previous['parameter_slice_attempts'];invalid_slices=previous['parameter_slice_invalid_proposals']
        if retained>total_draws:raise ValueError('Cannot shrink reflected chains')
        if retained:
            white[:,:retained]=previous['white'];ratios[:,:retained]=previous['log_ratio'];errors[:,:retained]=previous['energy_errors']
        rng.set_state(previous['rng_state'])
    current,bad=target.log_ratio_no_grad(q);evaluations+=1
    if bool(bad.any()):raise ValueError('Invalid initial reflected target')
    if previous is not None and float((current-previous['position_log_ratio']).abs().max())>1e-6:raise ValueError('Reflected saved marginal target differs')
    def state(include):
        data={'protocol':protocol,'position':q.clone(),'position_log_ratio':current.clone(),'completed_warmup':warm,
            'retained':retained,'rng_state':rng.get_state(),'step_size':epsilon,'log_epsilon':log_epsilon,'mu':mu,
            'averaged':averaged,'hbar':hbar,'counts':counts,'target_evaluations':evaluations,'gradient_evaluations':gradients,
            'parameter_slice_attempts':attempts,'parameter_slice_invalid_proposals':invalid_slices}
        if include:data.update({'white':white[:,:retained].clone(),'log_ratio':ratios[:,:retained].clone(),'energy_errors':errors[:,:retained].clone()})
        return data
    while warm<warmup or retained<total_draws:
        context=target.reflection_context(q) if hasattr(target,'reflection_context') else ReflectedStateContext(target,q,core)
        normals,offset=context.geometry()
        duration=min_time+(max_time-min_time)*float(torch.rand((),generator=rng))
        steps=max(1,round(duration/epsilon))
        new_state,info=state_transition(q[:,2:core],context,normals,offset,floor_scale,epsilon,steps,rng)
        gradients+=steps+1;evaluations+=steps+1
        q[:,2:core]=new_state
        phase='warmup' if warm<warmup else 'retained'
        for key in ['accepted','invalid','divergent','reflections']:
            counts[phase][key]=[a+int(b) for a,b in zip(counts[phase][key],info[key])]
        # Auxiliary walls have been discarded. All following values and the
        # parameter update refer to the original marginal soft-floor target.
        current,bad=target.log_ratio_no_grad(q);evaluations+=1
        if bool(bad.any()):raise ValueError('Invalid marginal target after reflected state step')
        q,current,used,tried,invalid=sweep(target.log_ratio_no_grad,q,current,rng,blocks)
        evaluations+=used;attempts=[a+int(b) for a,b in zip(attempts,tried)];invalid_slices=[a+int(b) for a,b in zip(invalid_slices,invalid)]
        if interweave:
            q,current,used,tried,invalid=centered_parameter_step(target,q,core,rng,scalar_parameters)
            evaluations+=used;attempts=[a+int(b) for a,b in zip(attempts,tried)];invalid_slices=[a+int(b) for a,b in zip(invalid_slices,invalid)]
        if phase=='warmup':
            warm+=1;rate=float(info['probability'].mean())
            hbar=(1-1/(warm+10))*hbar+(target_accept-rate)/(warm+10)
            log_epsilon=min(math.log(.8),max(math.log(.015),mu-math.sqrt(warm)/.05*hbar))
            averaged=warm**(-.75)*log_epsilon+(1-warm**(-.75))*averaged
            epsilon=math.exp(averaged if warm==warmup else log_epsilon)
            if warm%25==0 or warm==warmup:
                atomic_save(warm_path,state(False))
                if checkpoint_hook is not None:checkpoint_hook(warm)
        else:
            white[:,retained]=q;ratios[:,retained]=current
            errors[:,retained]=torch.where(info['invalid'],torch.inf,info['energy_error']);retained+=1
            if retained==16 or retained%chunk==0 or retained==total_draws:
                atomic_save(path,state(True));warm_path.unlink(missing_ok=True)
                write(root/'checkpoint_progress.json',{'sampler':protocol['sampler'],'retained_draws_per_chain':retained,
                    'requested_current_stage_total':total_draws,'gradient_evaluations':gradients,'target_evaluations':evaluations})
        if progress and (warm+retained)%25==0:
            progress({'warmup':warm,'retained':retained,'requested':total_draws,'step_size':epsilon,'steps':steps,
                'counts':counts[phase],'parameter_slice_attempts':attempts})
    return state(True)


@torch.no_grad()
def centered_parameter_step(target,white,core,rng,scalar_parameters=True):
    """Compose joint and scalar physical-parameter conditionals at fixed path."""
    guide=target.model;model=guide.model
    indices=torch.tensor([0,1,*range(core,white.shape[-1])])
    if bool((target.transform[indices,2:core]!=0).any()):
        raise ValueError('Physical parameters depend on state coordinates')
    physical=target.decode(white)
    affine=target.transform[indices[:,None],indices[None,:]]
    mean=target.mean[indices]
    def evaluate(v):
        candidate=physical.clone();candidate[:,indices]=mean+v@affine.T
        try:
            value=-model.objective(candidate)+.5*v.square().sum(-1)
            return value,~torch.isfinite(value)
        except (ValueError,torch.linalg.LinAlgError):
            rows=[]
            for row,coordinates in zip(candidate,v):
                try:value=-model.objective(row[None])[0]+.5*coordinates.square().sum()
                except (ValueError,torch.linalg.LinAlgError):value=row.new_tensor(-torch.inf)
                rows.append(value)
            values=torch.stack(rows)
            return values,~torch.isfinite(values)
    v=white[:,indices].clone();ratio,bad=evaluate(v)
    if bool(bad.any()):raise ValueError('Invalid centered parameter conditional')
    blocks=[{'indices':list(range(len(indices)))}]
    if scalar_parameters:blocks += [{'indices':[i]} for i in range(len(indices))]
    v,_,used,attempts,invalid=sweep(evaluate,v,ratio,rng,blocks)
    physical[:,indices]=mean+v@affine.T
    native,_=guide.transform(physical,inverse=True)
    result=torch.linalg.solve(target.transform,(native-target.mean).T).T
    value,bad=target.log_ratio_no_grad(result)
    if bool(bad.any()):raise ValueError('Invalid target after centered parameter step')
    return result,value,used+2,attempts,invalid
