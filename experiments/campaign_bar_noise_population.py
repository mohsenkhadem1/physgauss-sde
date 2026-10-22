"""Continuous conditional integration for the immutable nonlinear bar likelihood.

No observation generation or data reads at import. c=1/K has density 1/c;
the Gaussian conditional is integrated with a bounded inverse-power series.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from experiments import campaign_nonlinear_bar_inverse as bar

ORDER=20
Z_LIMIT=10.
C_LO,C_HI=1/400.,1/100.
S=bar.SIGMA/np.sqrt(20.)
F=np.array([1.,1.,3.,3.])
ROOT2PI=np.sqrt(2*np.pi)

def truncated_moments(upper):
    z=np.clip(np.asarray(upper,dtype=float),-Z_LIMIT,Z_LIMIT)
    a=-Z_LIMIT;pa=np.exp(-a*a/2)/ROOT2PI;pz=np.exp(-z*z/2)/ROOT2PI
    result=[ndtr(z)-ndtr(a),pa-pz]
    for n in range(2,ORDER+1):result.append(a**(n-1)*pa-z**(n-1)*pz+(n-1)*result[n-2])
    return np.stack(result)

def inverse_integral(m,upper=C_HI,power=1):
    """Integral exp(-z²/2)/c**power dc, with physical limits and ±10σ.

    Caller must have proved the complete ±10σ interval lies inside the prior.
    Omitted tails and geometric remainder are separately bounded in qualification.
    """
    m=np.asarray(m,dtype=float);z=np.minimum((np.asarray(upper)-m)/S,Z_LIMIT)
    moments=truncated_moments(z);alpha=-S/m
    total=np.zeros_like(m)
    for n in range(ORDER+1):total+=math.comb(power+n-1,n)*alpha**n*moments[n]
    value=S*ROOT2PI*total/m**power
    return np.where(z<=-Z_LIMIT,0.,value)

class PopulationPosterior:
    def __init__(self,observations,rtol=1e-6,atol=1e-12):
        self.observations=np.asarray(observations,dtype=float)
        if self.observations.ndim!=2 or self.observations.shape[1]!=4 or not np.isfinite(self.observations).all():raise ValueError('Finite [cases,4] raw observations required')
        self.n=len(self.observations);self.rtol=rtol;self.atol=atol;self.errors=[]
        endpoint=np.stack([self.parts(np.full(self.n,x))[1] for x in (0.,1.)])
        lo=endpoint.min(axis=0);hi=endpoint.max(axis=0)
        ratio=Z_LIMIT*S/lo
        # Bound all inverse powers used (1/c, 1/c², 1/c³), not only density.
        remainder=((ORDER+3)**2/2)*ratio**(ORDER+1)/(1-ratio)**3
        tail=(C_HI/C_LO)**3*math.erfc(Z_LIMIT/np.sqrt(2.))
        self.series_bound=float(np.max(remainder)+tail)
        if np.any(lo-Z_LIMIT*S<=C_LO) or np.any(hi+Z_LIMIT*S>=C_HI) or np.any(ratio>=1) or self.series_bound>=1e-14:raise ValueError('Entire-prior conditional series/tail qualification failed; no case may be dropped')
        self.shift=np.empty(self.n)
        for i in range(self.n):
            def objective(u):
                E=np.exp(bar.LOWER[0]+u*(bar.UPPER[0]-bar.LOWER[0]));a=bar.SIZE[0]*(bar.stretch(np.array([E]),bar.FORCES)[0][0]-1)
                r=self.observations[i]-np.array([0,a[0],0,a[1]]);m=r@F/20;r=r-F*m
                return r@r/(2*bar.SIGMA**2)
            self.shift[i]=minimize_scalar(objective,bounds=(0,1),method='bounded',options={'xatol':1e-11}).fun
        self.normalizer=self.integral(lambda u:self.weight(u))
        if np.any(self.normalizer<=0):raise RuntimeError('Nonpositive conditional normalizer')

    def parts(self,u):
        u=np.broadcast_to(u,(self.n,));E=np.exp(bar.LOWER[0]+u*(bar.UPPER[0]-bar.LOWER[0]))
        a=bar.SIZE[0]*(bar.stretch(E,bar.FORCES)[0]-1)
        g=np.stack([np.zeros(self.n),a[:,0],np.zeros(self.n),a[:,1]],axis=1)
        residual=self.observations-g;m=residual@F/20
        residual=residual-m[:,None]*F
        d=np.sum(residual**2,axis=1)/(2*bar.SIGMA**2)
        return E,m,d

    def weight(self,u,upper=C_HI,power=1):
        _,m,d=self.parts(u)
        return np.exp(self.shift-d)*inverse_integral(m,upper,power)

    def integral(self,function):
        value,error,info=quad_vec(function,0.,1.,epsrel=self.rtol,epsabs=self.atol,norm='max',limit=256,full_output=True)
        self.errors.append(float(error))
        if not info.success:raise RuntimeError(f'Adaptive outer integral failed: {info.message}')
        return value

    def cdf(self,kind,value):
        v=np.broadcast_to(value,(self.n,))
        if kind=='E':
            upper=np.clip((np.log(v)-bar.LOWER[0])/(bar.UPPER[0]-bar.LOWER[0]),0,1)
            return self.integral(lambda t:upper*self.weight(t*upper))/self.normalizer
        if kind=='K':return 1-self.integral(lambda u:self.weight(u,1/v))/self.normalizer
        if kind=='left':return self.integral(lambda u:self.weight(u,v/2))/self.normalizer
        if kind=='right':
            def integrand(u):
                E,_,_=self.parts(u);extension=bar.SIZE[0]*(bar.stretch(E,[2.])[0][:,0]-1)
                return self.weight(u,(v-extension)/2)
            return self.integral(integrand)/self.normalizer
        raise ValueError(kind)

    def quantile(self,kind,probability):
        bounds={'E':(4500.,9000.),'K':(100.,400.),'left':(.005,.02),'right':(.005,.05)}
        lo=np.full(self.n,bounds[kind][0]);hi=np.full(self.n,bounds[kind][1])
        for _ in range(23):
            middle=(lo+hi)/2;below=self.cdf(kind,middle)<probability
            lo=np.where(below,middle,lo);hi=np.where(below,hi,middle)
        return (lo+hi)/2

    def moments(self):
        center=np.array([6300.,200.,.01,.02]);scale=np.array([1000.,50.,.001,.001])
        def integrand(u):
            E,m,d=self.parts(u);i0=inverse_integral(m);iK=inverse_integral(m,power=2);iKK=inverse_integral(m,power=3)
            # Complete symmetric ±10σ normal moments; omitted tails are bounded.
            normal=S*ROOT2PI*(ndtr(Z_LIMIT)-ndtr(-Z_LIMIT))
            ec=normal/i0;ec2=m*normal/i0
            ek=iK/i0;vk=iKK/i0-ek**2;vc=ec2-ec**2
            if np.min(vk)<-1e-8 or np.min(vc)<-1e-18:raise RuntimeError('Materially negative conditional variance')
            extension=bar.SIZE[0]*(bar.stretch(E,[2.])[0][:,0]-1)
            mean=np.stack([E,ek,2*ec,2*ec+extension],axis=1)
            variance=np.stack([np.zeros(self.n),np.maximum(vk,0),4*np.maximum(vc,0),4*np.maximum(vc,0)],axis=1)
            delta=(mean-center)/scale
            weight=np.exp(self.shift-d)*i0
            return np.concatenate([weight[:,None]*delta,weight[:,None]*(delta**2+variance/scale**2)],axis=1)
        value=self.integral(integrand)/self.normalizer[:,None]
        variance=value[:,4:]-value[:,:4]**2
        if np.min(variance)<-1e-12:raise RuntimeError('Negative integrated centered variance')
        return center+scale*value[:,:4],scale*np.sqrt(np.maximum(variance,0))

    def summarize(self):
        kinds=('E','K','left','right');truth=np.r_[bar.TRUTH,bar.predict(bar.TRUTH[None],[2.])[0]]
        means,stds=self.moments();out={}
        for j,kind in enumerate(kinds):
            if kind=='left':
                low=2/out['K']['q95'];high=2/out['K']['q05']
            else:low=self.quantile(kind,.05);high=self.quantile(kind,.95)
            out[kind]={'q05':low,'q95':high,'cdf_truth':self.cdf(kind,truth[j]),'mean':means[:,j],'std':stds[:,j]}
        return out
