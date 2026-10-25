"""Positive continuous nonlinear-likelihood correction to a Gaussian proposal.

The linear response defines an integration change of variables only. Full
nonlinear likelihood/Jacobian values are sampled at fixed proposal-CDF knots;
the same nonnegative PWL density defines hypotheses and continuous CDFs.
Returned log weights exclude linear_peak: the caller adds that proposal
peak exactly once. This also holds for the zero-linear-signal branch.
No observations, material truth, atlas, model evaluation, or I/O is embedded.
"""
import numpy as np
from experiments.campaign_positive_amplitude_pwl_v01 import PositiveAmplitudePWL
from experiments.campaign_surface_conditional_amplitude_v01 import BOUNDS,LOG_WIDTH,log_gaussian_interval


def intrinsic_coordinates(amplitude,log_ratio):
    a=np.asarray(amplitude,float);z=np.asarray(log_ratio,float)
    la=np.log(a)
    return np.stack(((-la-np.log(BOUNDS[0,0]))/LOG_WIDTH[0],
                     (z-la-np.log(BOUNDS[1,0]))/LOG_WIDTH[1]),axis=-1)


class NonlinearTiltedPWL(PositiveAmplitudePWL):
    def __init__(self,mu,SD,lower,upper,*,log_ratio,linear_peak,full_loglik,panels=128,chunk=2048,full_loglik_az=None):
        super().__init__(mu,SD,lower,upper,panels=panels)
        z=np.asarray(log_ratio,float);peak=np.asarray(linear_peak,float)
        if z.shape!=self.mu.shape or peak.shape!=z.shape or not np.isfinite(z).all()or not np.isfinite(peak).all()or type(chunk)is not int or chunk<1:raise ValueError('Finite proposal log-ratio/peak and positive chunk')
        # Reconstruct proposal knots; original helper already validated PPFs.
        knots_a=np.empty_like(self.h)
        knots_a[~self.zero]=1/self.h[~self.zero]
        if np.any(self.zero):
            knots_a[self.zero]=np.exp(np.log(self.lower[self.zero,None])+np.log(self.upper[self.zero]/self.lower[self.zero])[:,None]*self.knots)
        unit=intrinsic_coordinates(knots_a,z[:,None]).reshape(-1,2)
        flat_a=knots_a.ravel();flat_z=np.broadcast_to(z[:,None],knots_a.shape).ravel()
        ll=np.concatenate([np.asarray(full_loglik(unit[i:i+chunk]) if full_loglik_az is None else full_loglik_az(flat_a[i:i+chunk],flat_z[i:i+chunk]),float) for i in range(0,len(unit),chunk)])
        if ll.shape!=(len(unit),)or not np.isfinite(ll).all():raise FloatingPointError('Finite full nonlinear knot likelihoods required')
        ll=ll.reshape(knots_a.shape)
        linear=peak[:,None]-.5*((knots_a-self.mu[:,None])/self.SD[:,None])**2
        logh=ll-linear
        logh[~self.zero]-=np.log(knots_a[~self.zero])
        # The log-a proposal already absorbs 1/a for zero linear response.
        self.row_shift=np.max(logh,axis=1)
        self.h=np.exp(logh-self.row_shift[:,None])
        xi=np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
        width=np.diff(self.knots)
        hquad=(1-xi)[None,None,:]*self.h[:,:-1,None]+xi[None,None,:]*self.h[:,1:,None]
        self.density_weights=(hquad*(width/2)[None,:,None]).reshape(len(z),-1)
        self.panel_mass=self.density_weights.reshape(len(z),panels,2).sum(axis=2)
        self.mass0=self.density_weights.sum(axis=1)
        self.log_kernel_mass=self.log_kernel_mass+self.row_shift
        with np.errstate(divide='ignore'):self.log_weights=self.log_kernel_mass[:,None]+np.log(self.density_weights)
        if np.any(self.mass0<=0)or not np.isfinite(self.mass0).all()or np.isnan(self.log_weights).any()or np.isposinf(self.log_weights).any():raise FloatingPointError('Unresolved tilted positive density')

    def _probability(self,threshold,*,tail):
        q=np.asarray(threshold,float)
        if q.shape!=self.lower.shape or not np.isfinite(q).all():raise ValueError('Finite amplitude threshold per row')
        out=np.empty_like(q);below=q<=self.lower;above=q>=self.upper;inside=~(below|above)
        out[below]=1. if tail else 0.;out[above]=0. if tail else 1.
        normal=inside&~self.zero
        # Row order matters when combining Gaussian and zero-response proposals.
        t=np.empty_like(q)
        if np.any(normal):
            m=self.mu[normal];s=self.SD[normal];alpha=(self.lower[normal]-m)/s;beta=(self.upper[normal]-m)/s;cut=(q[normal]-m)/s
            left=log_gaussian_interval(alpha,cut);right=log_gaussian_interval(cut,beta)
            t[normal]=np.exp((right if tail else left)-np.logaddexp(left,right))
        zero=inside&self.zero
        if np.any(zero):
            t[zero]=(np.log(self.upper[zero]/q[zero])if tail else np.log(q[zero]/self.lower[zero]))/np.log(self.upper[zero]/self.lower[zero])
        if np.any(inside):
            if tail:knots=1-self.knots[::-1];h=self.h[inside,::-1];mass=self.panel_mass[inside,::-1]
            else:knots=self.knots;h=self.h[inside];mass=self.panel_mass[inside]
            out[inside]=self._primitive(knots,h,mass,t[inside])/self.mass0[inside]
        if not np.isfinite(out).all()or np.any(out<0)or np.any(out>1+32*np.finfo(float).eps):raise FloatingPointError('Invalid tilted CDF; no clipping')
        return out
