"""Positive continuous conditional-amplitude quadrature, source-only.

Only the Gaussian-CDF-coordinate density h(t)=1/a(t) is approximated by a
positive piecewise-linear function. No FE response or material coordinate is
interpolated. Two-point positive Gauss integration exactly integrates each
linear density panel; the same density defines the continuous CDF. This is
finite quadrature, requiring panel/outer refinement and independent checks.
"""
import numpy as np
from scipy.stats import truncnorm
from experiments.campaign_surface_conditional_amplitude_v01 import log_gaussian_interval


class PositiveAmplitudePWL:
    def __init__(self,mu,SD,lower,upper,*,panels=128):
        if type(panels)is not int or panels not in(128,256,512):
            raise ValueError('Declared Chebyshev-Lobatto panel count128/256/512')
        self.mu,self.SD,self.lower,self.upper=np.broadcast_arrays(*[np.asarray(v,float)for v in(mu,SD,lower,upper)])
        if self.mu.ndim!=1 or not len(self.mu) or not all(np.isfinite(v).all()for v in(self.mu,self.lower,self.upper)) or np.isnan(self.SD).any() or np.any(self.SD<=0) or np.any(self.lower<=0) or np.any(self.upper<=self.lower):
            raise ValueError('Finite rowwise amplitudes, positive SD or+inf, positive ordered bounds')
        self.panels=panels;self.zero=np.isposinf(self.SD)
        self.knots=np.sin(np.pi*np.arange(panels+1)/(2*panels))**2
        self.knots[0]=0.;self.knots[-1]=1.
        width=np.diff(self.knots)
        if np.any(width<=0):raise FloatingPointError('Strict positive panel widths')
        xi=np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
        self.t=(self.knots[:-1,None]+width[:,None]*xi).ravel()
        self.h=np.ones((len(self.mu),panels+1));self.amplitude=np.empty((len(self.mu),2*panels))
        self.log_kernel_mass=np.empty(len(self.mu));normal=~self.zero
        if np.any(normal):
            m=self.mu[normal];s=self.SD[normal];lo=self.lower[normal];hi=self.upper[normal]
            alpha=(lo-m)/s;beta=(hi-m)/s
            self.log_kernel_mass[normal]=np.log(s*np.sqrt(2*np.pi))+log_gaussian_interval(alpha,beta)
            # Endpoints are exact physical bounds, never a PPF endpoint call.
            interior=m[:,None]+s[:,None]*truncnorm.ppf(self.knots[None,1:-1],alpha[:,None],beta[:,None])
            endpoints=np.column_stack((lo,interior,hi))
            if np.any(endpoints<=0)or not np.isfinite(endpoints).all()or np.any(endpoints<lo[:,None])or np.any(endpoints>hi[:,None]):raise FloatingPointError('Invalid conditional PPF density nodes')
            self.h[normal]=1/endpoints
            self.amplitude[normal]=m[:,None]+s[:,None]*truncnorm.ppf(self.t[None,:],alpha[:,None],beta[:,None])
        if np.any(self.zero):
            lo=self.lower[self.zero];hi=self.upper[self.zero];span=np.log(hi/lo)
            self.log_kernel_mass[self.zero]=np.log(span)
            self.amplitude[self.zero]=np.exp(np.log(lo[:,None])+span[:,None]*self.t)
        # Positive convex endpoint combinations: no clipping or mass repair.
        hquad=((1-xi)[None,None,:]*self.h[:,:-1,None]+xi[None,None,:]*self.h[:,1:,None])
        self.density_weights=(hquad*(width/2)[None,:,None]).reshape(len(self.mu),-1)
        self.panel_mass=self.density_weights.reshape(len(self.mu),panels,2).sum(axis=2)
        self.mass0=self.density_weights.sum(axis=1)
        self.log_weights=self.log_kernel_mass[:,None]+np.log(self.density_weights)
        if not np.isfinite(self.amplitude).all()or np.any(self.amplitude<self.lower[:,None])or np.any(self.amplitude>self.upper[:,None])or not np.isfinite(self.log_weights).all()or np.any(self.density_weights<=0):raise FloatingPointError('Invalid positive amplitude hypotheses')

    @staticmethod
    def _primitive(knots,h,panel_mass,x):
        """Positive panel primitive; no subtraction of nearly equal CDFs."""
        x=np.asarray(x,float);N=len(knots)-1
        if x.shape!=(len(h),)or not np.isfinite(x).all()or np.any((x<0)|(x>1)):raise ValueError('Unit primitive coordinate per row')
        idx=np.searchsorted(knots,x,side='right')-1
        idx=np.minimum(idx,N-1) # The defined endpoint x=1 uses the last panel.
        rows=np.arange(len(h));delta=x-knots[idx];width=knots[idx+1]-knots[idx];fraction=delta/width
        local=delta*((1-.5*fraction)*h[rows,idx]+.5*fraction*h[rows,idx+1])
        cumulative=np.concatenate((np.zeros((len(h),1)),np.cumsum(panel_mass,axis=1)),axis=1)
        return cumulative[rows,idx]+local

    def _probability(self,threshold,*,tail):
        q=np.asarray(threshold,float)
        if q.shape!=self.lower.shape or not np.isfinite(q).all():raise ValueError('Finite physical threshold per row')
        answer=np.empty_like(q);below=q<=self.lower;above=q>=self.upper;inside=~(below|above)
        answer[below]=1. if tail else 0.;answer[above]=0. if tail else 1.
        zero=inside&self.zero
        if np.any(zero):
            numerator=np.log(self.upper[zero]/q[zero])if tail else np.log(q[zero]/self.lower[zero])
            answer[zero]=numerator/np.log(self.upper[zero]/self.lower[zero])
        normal=inside&~self.zero
        if np.any(normal):
            m=self.mu[normal];s=self.SD[normal];alpha=(self.lower[normal]-m)/s;beta=(self.upper[normal]-m)/s;cut=(q[normal]-m)/s
            left=log_gaussian_interval(alpha,cut);right=log_gaussian_interval(cut,beta)
            # Ratio computed from its own side: never 1 minus a tiny tail.
            coordinate=np.exp((right if tail else left)-np.logaddexp(left,right))
            if tail:
                knots=1-self.knots[::-1];h=self.h[normal,::-1];mass=self.panel_mass[normal,::-1]
            else:knots=self.knots;h=self.h[normal];mass=self.panel_mass[normal]
            answer[normal]=self._primitive(knots,h,mass,coordinate)/self.mass0[normal]
        if not np.isfinite(answer).all()or np.any(answer<0)or np.any(answer>1+32*np.finfo(float).eps):raise FloatingPointError('Invalid continuous density probability; no clipping')
        return answer

    def cdf(self,threshold):return self._probability(threshold,tail=False)
    def tail(self,threshold):return self._probability(threshold,tail=True)

    def diagnostics(self):
        return dict(panels=self.panels,hypotheses_per_row=2*self.panels,density='Positive PWL h(t)=1/a(t); gzero exactloga',continuous_CDF=True,atomic_CDF=False,probability_clipping=False,FE_or_parameter_interpolation=False,requires_panel_and_outer_refinement=True)
