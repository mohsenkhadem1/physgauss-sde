"""FP32 linear corrections with FP64 original-operator refinement and fallback.

Only the preconditioning/linear-solve arithmetic changes. Every accepted
direction meets the same original FP64 residual gate. Energy, force, line
search, mass, constraints and physical-time equations remain FP64.
"""
import torch
from experiments.campaign_p2_torch import TorchP2,LinearFailure
from experiments.campaign_p2_block_preconditioner import BlockJacobiP2


class MixedCorrectionP2(BlockJacobiP2):
    def __init__(self,model,device='cpu',maximum_cg_iterations=4000,maximum_low_iterations=4000):
        if type(maximum_low_iterations) is not int or maximum_low_iterations<=0:raise ValueError('Positive total FP32 iteration budget required')
        super().__init__(model,device,maximum_cg_iterations)
        self.maximum_low_iterations=maximum_low_iterations
        self.low=object.__new__(TorchP2)
        self.low.__dict__={k:(v.to(torch.float32) if isinstance(v,torch.Tensor) and v.is_floating_point() else v) for k,v in self.__dict__.items() if k!='linear_records'}
        self.low.dtype=torch.float32;self.mixed_records=[]

    def cg(self,cache,rhs,mass_factor,rtol=1e-8,atol=1e-12,maximum_iterations=None):
        cap=self.maximum_cg_iterations if maximum_iterations is None else maximum_iterations
        def op(v):
            v=v.masked_fill(self.mask,0.)
            out=self.tangent(cache,v)
            if mass_factor:out=out+mass_factor*self.mass(v)
            return out.masked_fill(self.mask,0.)
        low_cache={k:v.to(torch.float32) for k,v in cache.items() if k in ['F','cofactor','s']}
        b=rhs.masked_fill(self.mask,0.);x=torch.zeros_like(b);r=b.clone();norm=float(torch.linalg.vector_norm(r));tolerance=max(atol,rtol*norm);steps=0;rounds=[];reason=None;failed_low_cap=0
        for _ in range(8):
            if norm<=tolerance:break
            remaining=self.maximum_low_iterations-steps
            if remaining<=0:reason='Total FP32 iteration cap';break
            try:delta,k=self.low.cg(low_cache,r.to(torch.float32),mass_factor,rtol=.01,atol=1e-25,maximum_iterations=min(cap,remaining))
            except (LinearFailure,RuntimeError) as error:
                reason=type(error).__name__+': '+str(error);failed_low_cap=min(cap,remaining);break
            trial=x+delta.to(self.dtype);new_r=b-op(trial);new_norm=float(torch.linalg.vector_norm(new_r));steps+=k
            rounds.append(dict(inner_iterations=k,original_FP64_residual_N=new_norm))
            if not torch.isfinite(new_r).all() or new_norm>=.7*norm:reason='Insufficient original-residual contraction';break
            x,r,norm=trial,new_r,new_norm
        if norm>tolerance:
            delta,k=super().cg(cache,r,mass_factor,rtol=min(rtol,.1*tolerance/max(norm,1e-300)),atol=.1*tolerance,maximum_iterations=cap)
            x=x+delta;steps+=k;reason=reason or 'FP32 refinement round cap'
        actual=b-op(x);error=float(torch.linalg.vector_norm(actual))
        if error>max(2*tolerance,1e-11):raise LinearFailure('Mixed correction failed original FP64 residual gate')
        self.mixed_records.append(dict(rounds=rounds,fallback=reason is not None,fallback_reason=reason,original_FP64_residual_N=error,requested_tolerance_N=tolerance,completed_reported_CG_iterations=steps,failed_FP32_iterations_upper_bound=failed_low_cap,total_CG_iterations_upper_bound=steps+failed_low_cap,maximum_total_FP32_iterations=self.maximum_low_iterations,maximum_FP64_fallback_iterations=cap,inner_FP32_curvature_checks_are_not_FP64_curvature_certificates=True))
        return x,steps+failed_low_cap
