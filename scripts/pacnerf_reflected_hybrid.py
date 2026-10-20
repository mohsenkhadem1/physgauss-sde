"""Slice-augmented reflective state HMC, followed by marginal parameter ESS.

The floor auxiliaries are discarded BEFORE updating parameters. The parameter
step and all saved log densities use the original soft-floor marginal target.
"""
import hashlib,math
from pathlib import Path
import torch
from pacnerf_hybrid import ConditionalStateEvaluator
from pacnerf_reflections import state_transition
from pacnerf_elliptical import sweep,atomic_save
from pacnerf_comparison import write


class ReflectedStateContext(ConditionalStateEvaluator):
    @torch.no_grad()
    def geometry(self):
        model=self.model;m=model.modes;core=self.core
        root=self.context['gaussian'][1]
        state_map=torch.linalg.solve_triangular(root.transpose(-1,-2),
            self.target.transform[2:core,2:core].expand(len(self.fixed),-1,-1),upper=True)
        q_index=torch.arange(m,model.base_dimension-2).reshape(13,2*m)[:,:m].flatten()
        q_map=(state_map[:,q_index]*model.scale[2+q_index][None,:,None]).reshape(len(self.fixed),13,m,core-2)
        normals=torch.einsum('gm,btms->btgs',model.floor_basis,q_map).flatten(1,2)
        zero=self.fixed.clone();zero[:,2:core]=0
        physical=self.target.model.to_physical(self.target.coordinates(zero))
        offset=model.floor_heights(physical).flatten(1)
        return normals,offset

    @torch.enable_grad()
    def _evaluate(self,state,fixed,context):
        variable=state.detach().requires_grad_();full=fixed.clone();full[:,2:self.core]=variable
        try:
            if self.model._conditional_context is not None:raise ValueError('Nested conditional model context')
            self.model._conditional_context=context
            physical,jac=self.target.model.transform(self.target.coordinates(full))
            # The sigmoid floor event is represented by sampled linear walls.
            # Every other term, including nonlinear contact/neural covariance
            # and complete observed images, remains in this exact residual.
            loss=self.model.dynamics(physical)+.5*self.model.image_residual(physical).square().sum(-1)+self.model.image_normalizer(physical)
            value=-loss+jac+self.target.center+.5*full.square().sum(-1)
            gradient,=torch.autograd.grad(value.sum(),variable)
        finally:self.model._conditional_context=None
        bad=~torch.isfinite(value)|~torch.isfinite(gradient).all(-1)
        return value.detach(),torch.nan_to_num(gradient.detach()),bad

    def __call__(self,state):
        try:return self._evaluate(state,self.fixed,self.context)
        except torch.linalg.LinAlgError:
            rows=[]
            for i in range(len(state)):
                try:rows.append(self._evaluate(state[i:i+1],self.fixed[i:i+1],None))
                except torch.linalg.LinAlgError:
                    rows.append((state.new_tensor([-torch.inf]),torch.zeros_like(state[i:i+1]),torch.tensor([True])))
            return tuple(torch.cat([row[i] for row in rows]) for i in range(3))


@torch.no_grad()
def sample(target,dimension,core,checkpoint,total_draws,warmup=500,seed=940001,chunk=50,
        min_time=1.,max_time=4.,target_accept=.9,initial=None,floor_scale=.002,progress=None,checkpoint_hook=None):
    if total_draws<16 or not 2<core<=dimension or warmup<0:raise ValueError('Invalid reflected sampling budget')
    root=Path(checkpoint);root.mkdir(parents=True,exist_ok=True)
    path=root/'chains.pt';warm_path=root/'reflected_warmup.pt'
    blocks=[{'indices':[0,1,*range(core,dimension)]}]
    if initial is not None and initial.shape!=(4,dimension):raise ValueError('Invalid reflected initial states')
    protocol={'sampler':'logistic-slice-reflected-hmc-parameter-ess-v1','dimension':dimension,'core':core,
        'chains':4,'warmup':warmup,'seed':seed,'min_time':min_time,'max_time':max_time,'floor_scale':floor_scale,
        'target_accept':target_accept,'parameter_blocks':blocks,
        'initial_state_sha256':None if initial is None else hashlib.sha256(initial.contiguous().numpy().tobytes()).hexdigest()}
    previous=torch.load(path,weights_only=True) if path.exists() else torch.load(warm_path,weights_only=True) if warm_path.exists() else None
    rng=torch.Generator().manual_seed(seed)
    white=torch.empty(4,total_draws,dimension,dtype=torch.float64)
    ratios=torch.empty(4,total_draws,dtype=torch.float64);errors=torch.empty_like(ratios)
    if previous is None:
        q=torch.randn(4,dimension,dtype=torch.float64,generator=rng) if initial is None else initial.clone()
        warm=0;retained=0;epsilon=.3;log_epsilon=math.log(epsilon)
        mu=math.log(10*epsilon);averaged=log_epsilon;hbar=0.
        counts={phase:{key:[0]*4 for key in ['accepted','invalid','divergent','reflections']} for phase in ['warmup','retained']}
        evaluations=0;gradients=0;attempts=[0]*4;invalid_slices=[0]*4
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
