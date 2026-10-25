"""Positive two-proposal quadrature with a fixed log-a prior floor.

Each normalized proposal contributes half the full L/(a*qMix) integral.
Returned weights EXCLUDE linear_peak; caller adds it exactly once. There is
no Gaussian kernel-mass multiplier. This changes quadrature, never the prior
or likelihood. Mandatory independent refinement/oracle checks remain needed.
"""
import numpy as np
from experiments.campaign_positive_amplitude_pwl_v01 import PositiveAmplitudePWL
from experiments.campaign_nonlinear_tilted_pwl_v01 import intrinsic_coordinates
from experiments.campaign_nonlinear_tilted_pwl_v02 import positive_panel_probability
from experiments.campaign_surface_conditional_amplitude_v01 import log_gaussian_interval


class NonlinearMISPWL:
    def __init__(self,mu,SD,lower,upper,*,log_ratio,linear_peak,full_loglik,panels=128,chunk=2048,full_loglik_az=None):
        base=PositiveAmplitudePWL(mu,SD,lower,upper,panels=panels)
        for name in ('mu','SD','lower','upper','zero','knots','t','panels'):setattr(self,name,getattr(base,name))
        z=np.asarray(log_ratio,float);peak=np.asarray(linear_peak,float);R=len(self.mu)
        if z.shape!=(R,)or peak.shape!=(R,)or not np.isfinite(z).all()or not np.isfinite(peak).all()or type(chunk)is not int or chunk<1:raise ValueError('Finite rowwise z/peak and positive chunk')
        self.logspan=np.log(self.upper/self.lower)
        prior_knots=np.exp(np.log(self.lower[:,None])+self.logspan[:,None]*self.knots)
        prior_knots[:,0]=self.lower;prior_knots[:,-1]=self.upper
        gaussian_knots=prior_knots.copy();gaussian_knots[~self.zero]=1/base.h[~self.zero]
        # Copy exact physical endpoints rather than reciprocal roundtrips.
        gaussian_knots[:,0]=self.lower;gaussian_knots[:,-1]=self.upper
        knots_a=np.stack((gaussian_knots,prior_knots),axis=1)
        prior_atoms=np.exp(np.log(self.lower[:,None])+self.logspan[:,None]*self.t)
        self.amplitude=np.concatenate((base.amplitude,prior_atoms),axis=1)
        logqP=-np.log(knots_a)-np.log(self.logspan)[:,None,None]
        logqG=logqP.copy();normal=~self.zero
        if np.any(normal):
            m=self.mu[normal];s=self.SD[normal]
            logmass=log_gaussian_interval((self.lower[normal]-m)/s,(self.upper[normal]-m)/s)
            logqG[normal]=-.5*((knots_a[normal]-m[:,None,None])/s[:,None,None])**2-np.log(s*np.sqrt(2*np.pi))[:,None,None]-logmass[:,None,None]
        logmix=np.logaddexp(logqG,logqP)-np.log(2.)
        flat_a=knots_a.ravel();flat_z=np.broadcast_to(z[:,None,None],knots_a.shape).ravel();parts=[]
        for start in range(0,len(flat_a),chunk):
            a=flat_a[start:start+chunk];zz=flat_z[start:start+chunk]
            value=full_loglik(intrinsic_coordinates(a,zz))if full_loglik_az is None else full_loglik_az(a,zz)
            value=np.asarray(value,float)
            if value.shape!=a.shape or not np.isfinite(value).all():raise FloatingPointError('Finite bounded full likelihood callback required')
            parts.append(value)
        ll=np.concatenate(parts).reshape(knots_a.shape)
        logh=ll-peak[:,None,None]-np.log(knots_a)-logmix-np.log(2.)
        self.row_shift=np.max(logh,axis=(1,2));self.h=np.exp(logh-self.row_shift[:,None,None])
        xi=np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)]);width=np.diff(self.knots)
        hquad=(1-xi)[None,None,None,:]*self.h[:,:,:-1,None]+xi[None,None,None,:]*self.h[:,:,1:,None]
        self.stream_density_weights=(hquad*(width/2)[None,None,:,None]).reshape(R,2,2*panels)
        self.panel_mass=self.stream_density_weights.reshape(R,2,panels,2).sum(axis=3)
        self.density_weights=self.stream_density_weights.reshape(R,4*panels)
        self.mass0=self.density_weights.sum(axis=1)
        self.log_kernel_mass=self.row_shift.copy()
        with np.errstate(divide='ignore'):self.log_weights=self.row_shift[:,None]+np.log(self.density_weights)
        if np.any(self.mass0<=0)or not np.isfinite(self.mass0).all()or np.isnan(self.log_weights).any()or np.isposinf(self.log_weights).any():raise FloatingPointError('Unresolved positive MIS mass')

    def _probability(self,threshold,*,tail):
        q=np.asarray(threshold,float)
        if q.shape!=self.lower.shape or not np.isfinite(q).all():raise ValueError('Finite threshold per row')
        out=np.empty_like(q);below=q<=self.lower;above=q>=self.upper;inside=~(below|above)
        out[below]=1. if tail else 0.;out[above]=0. if tail else 1.
        if np.any(inside):
            qp=q[inside];lo=self.lower[inside];hi=self.upper[inside]
            tp=(np.log(hi/qp)if tail else np.log(qp/lo))/np.log(hi/lo)
            tg=tp.copy();normal=~self.zero[inside]
            if np.any(normal):
                m=self.mu[inside][normal];s=self.SD[inside][normal];cut=(qp[normal]-m)/s
                left=log_gaussian_interval((lo[normal]-m)/s,cut);right=log_gaussian_interval(cut,(hi[normal]-m)/s)
                tg[normal]=np.exp((right if tail else left)-np.logaddexp(left,right))
            mass=self.panel_mass[inside];streammass=np.sum(mass.astype(np.longdouble),axis=2)
            probs=np.zeros((len(qp),2))
            for j,t in enumerate((tg,tp)):
                positive=streammass[:,j]>0
                # Exact zero numerical mass contributes zero; no positive mass is dropped.
                if np.any(positive):probs[positive,j]=positive_panel_probability(self.knots,self.h[inside,j][positive],mass[:,j][positive],t[positive],reverse=tail)
            out[inside]=np.asarray(np.sum(probs.astype(np.longdouble)*streammass,axis=1)/np.sum(streammass,axis=1),float)
        if not np.isfinite(out).all()or np.any((out<0)|(out>1)):raise FloatingPointError('Invalid positive MIS probability; no clipping')
        return out

    def cdf(self,threshold):return self._probability(threshold,tail=False)
    def tail(self,threshold):return self._probability(threshold,tail=True)
    def diagnostics(self):return dict(panels=self.panels,hypotheses_per_row=4*self.panels,proposal='Fixed .5 truncated Gaussian + .5 log-a',continuous_CDF=True,probability_clipping=False,requires_panel_and_outer_refinement=True)
