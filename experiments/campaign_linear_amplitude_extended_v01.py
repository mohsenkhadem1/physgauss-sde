"""Separate linear-law inner1024 extension, no frozen-engine mutation.

Only the positive PWL primitive admits1024 panels. Every old128/256/512
operation is unchanged. Frozen integration, continuous CDF, qualification and
independent original-a oracle are inherited; no likelihood/prior/gate change.
"""
import numpy as np
from scipy.special import logsumexp
from experiments.campaign_surface_conditional_amplitude_v01 import gauss_axis,BOUNDS,LOG_WIDTH
from experiments.campaign_surface_conditional_amplitude_v02 import ConditionalAmplitudeReferenceV02
from experiments.campaign_positive_amplitude_pwl_v02 import PositiveAmplitudePWLV02
from experiments.campaign_nonlinear_conditional_reference_v04 import ExtendedAmplitudePWLPlan


class LinearAmplitudeExtendedReferenceV01(ConditionalAmplitudeReferenceV02):
    def _state(self,outer,inner):
        if self._state_cache is not None and self._state_cache['key']==(outer,inner):return self._state_cache
        t,w=gauss_axis(outer);breaks=self._breaks();z=np.concatenate([a+(b-a)*t for a,b in zip(breaks,breaks[1:])]);zw=np.concatenate([(b-a)*w for a,b in zip(breaks,breaks[1:])])
        r,mu,SD,peak=self._statistics(z);lo=np.maximum(1/BOUNDS[0,1],r/BOUNDS[1,1]);hi=np.minimum(1/BOUNDS[0,0],r/BOUNDS[1,0])
        primitive=PositiveAmplitudePWLV02(mu,SD,lo,hi,panels=inner)
        logmass=logsumexp(primitive.log_weights,axis=1);rowlog=logmass+peak+np.log(zw)-np.log(LOG_WIDTH.prod());logZ=float(logsumexp(rowlog))
        rho=np.exp(rowlog-logZ);rho/=rho.sum();conditional=np.exp(primitive.log_weights-logmass[:,None]);conditional/=conditional.sum(axis=1)[:,None]
        a=primitive.amplitude;abar=np.sum(conditional*a,axis=1);avariance=np.sum(conditional*(a-abar[:,None])**2,axis=1)
        loga=np.log(a);logmean=np.sum(conditional*loga,axis=1);logvariance=np.sum(conditional*(loga-logmean[:,None])**2,axis=1)
        if not np.isfinite(logZ)or not np.isfinite(rho).all()or not np.isfinite(conditional).all():raise FloatingPointError('Invalid positive conditional density mass')
        state=dict(key=(outer,inner),z=z,r=r,mu=mu,SD=SD,peak=peak,primitive=primitive,conditional=conditional,rho=rho,log_evidence=logZ,amplitude_mean=abar,amplitude_variance=avariance,log_amplitude_mean=logmean,log_amplitude_variance=logvariance)
        self._state_cache=state;return state

    def qualify(self,*,plan=ExtendedAmplitudePWLPlan()):
        answer=super().qualify(plan=plan)
        answer['diagnostics'].update(schema='conditional-amplitude-linear-extended-v01',original_likelihood_prior_oracle_gates_unchanged=True)
        return answer
