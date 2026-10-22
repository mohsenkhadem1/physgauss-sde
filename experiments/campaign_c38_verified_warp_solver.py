"""NEW Warp-HVP directions checked against the unchanged original Torch HVP.

CG iterations/tolerances/acceptance and final static residual are untouched.
One original Torch operator application checks each returned linear direction;
this is not an absolute final-static tolerance imposed on early Newton steps.
"""
import numpy as np
import torch
from experiments.campaign_p2_torch_warp_hvp import TorchWarpBulkHVPMixin
from experiments.campaign_p2_streaming import StreamingTwoLevelP2
from experiments.campaign_p2_torch import LinearFailure


def checked_linear_residual(rhs,applied,mask,requested_tolerance_N):
    if rhs.shape!=applied.shape or mask.shape!=rhs.shape or mask.dtype!=torch.bool or not bool(torch.isfinite(rhs).all()) or not bool(torch.isfinite(applied).all()) or not np.isfinite(requested_tolerance_N) or requested_tolerance_N<=0:raise ValueError('Finite original linear system and requested tolerance required')
    b=rhs.masked_fill(mask,0.);a=applied.masked_fill(mask,0.)
    norm=float(torch.linalg.vector_norm(b-a));scale=max(float(torch.linalg.vector_norm(b)),float(torch.linalg.vector_norm(a)),1e-30)
    bound=max(2*float(requested_tolerance_N),1e-11);margin=64*np.finfo(np.float64).eps*scale
    row=dict(independent_original_Torch_HVP_true_R_N=norm,original_requested_CG_tolerance_N=float(requested_tolerance_N),original_CG_acceptance_bound_N=bound,float64_evaluation_margin_N=margin,passed=bool(norm<=bound+margin),interpretation='Original inexact-CG acceptance bound, NOT final static1e-10N criterion')
    if not row['passed']:raise LinearFailure('Independent original Torch linear residual exceeds original CG acceptance bound: '+str(row))
    return row


class IndependentlyCheckedWarpCGMixin(TorchWarpBulkHVPMixin):
    def cg(self,cache,rhs,mass_factor,*args,**kwargs):
        direction,iterations=super().cg(cache,rhs,mass_factor,*args,**kwargs)
        accepted=self.linear_records[-1]
        applied=StreamingTwoLevelP2.tangent(self,cache,direction.masked_fill(self.mask,0.))
        if mass_factor:applied=applied+mass_factor*self.mass(direction.masked_fill(self.mask,0.))
        row=checked_linear_residual(rhs,applied,self.mask,accepted['requested_tolerance_N'])
        row['iterations']=iterations
        if not hasattr(self,'independent_linear_checks'):self.independent_linear_checks=[]
        self.independent_linear_checks.append(row)
        return direction,iterations
