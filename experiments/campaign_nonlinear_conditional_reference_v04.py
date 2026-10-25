"""1024-panel extension; original likelihood/prior/qualification/oracle unchanged."""
import numpy as np
from scipy.special import logsumexp
from experiments.campaign_surface_conditional_amplitude_v01 import gauss_axis,BOUNDS,LOG_WIDTH
from experiments.campaign_nonlinear_conditional_reference_v01 import NonlinearConditionalReference
from experiments.campaign_nonlinear_mis_pwl_v02 import NonlinearMISPWLV02
from dataclasses import dataclass
from experiments.campaign_surface_conditional_amplitude_v02 import AmplitudePWLPlan


@dataclass(frozen=True)
class ExtendedAmplitudePWLPlan(AmplitudePWLPlan):
    """Additional inner resolution, with unchanged outer rule and gates."""
    inner:tuple=(128,256,512,1024)

    def __post_init__(self):
        inner=tuple(self.inner);outer=tuple(self.outer)
        if len(inner)<2 or any(type(n)is not int or n not in(128,256,512,1024)for n in inner)or any(a>=b for a,b in zip(inner,inner[1:])):
            raise ValueError('Strictly increasing at leasttwo declared PWLpanel orders128/256/512/1024')
        # Delegate every other check and canonicalization to the frozen plan.
        base=AmplitudePWLPlan(inner=(128,256),outer=outer,moment_chunk=self.moment_chunk)
        object.__setattr__(self,'inner',inner);object.__setattr__(self,'outer',base.outer)


class NonlinearConditionalReferenceV04(NonlinearConditionalReference):
    def _state(self,outer,inner):
        if self._state_cache is not None and self._state_cache['key']==(outer,inner):return self._state_cache
        t,w=gauss_axis(outer);breaks=self._breaks()
        z=np.concatenate([a+(b-a)*t for a,b in zip(breaks,breaks[1:])]);zw=np.concatenate([(b-a)*w for a,b in zip(breaks,breaks[1:])])
        r,mu,SD,peak=self._statistics(z);lo=np.maximum(1/BOUNDS[0,1],r/BOUNDS[1,1]);hi=np.minimum(1/BOUNDS[0,0],r/BOUNDS[1,0])
        primitive=NonlinearMISPWLV02(mu,SD,lo,hi,log_ratio=z,linear_peak=peak,full_loglik=self.full_loglik,full_loglik_az=self._full_loglik_az,panels=inner,chunk=self.evaluation_chunk)
        lm=logsumexp(primitive.log_weights,axis=1);rowlog=lm+peak+np.log(zw)-np.log(LOG_WIDTH.prod());logZ=float(logsumexp(rowlog))
        rho=np.exp(rowlog-logZ);rho/=rho.sum()
        conditional=np.exp(primitive.log_weights-lm[:,None]);conditional/=conditional.sum(axis=1)[:,None]
        if not np.isfinite(logZ)or not np.isfinite(rho).all()or not np.isfinite(conditional).all():raise FloatingPointError('Invalid nonlinear posterior mass')
        self._state_cache=dict(key=(outer,inner),z=z,r=r,primitive=primitive,rho=rho,conditional=conditional,log_evidence=logZ)
        return self._state_cache


    def qualify(self,*,plan=ExtendedAmplitudePWLPlan()):
        answer=super().qualify(plan=plan)
        answer['diagnostics'].update(schema='nonlinear-conditional-reference-v04',proposal='Fixed .5 Gaussian + .5 log-a MIS',normalized_proposals=True,linear_peak_added_once=True,no_probability_clipping=True)
        return answer
