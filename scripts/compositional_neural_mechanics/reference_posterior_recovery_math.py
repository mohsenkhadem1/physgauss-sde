"""Recovery: positive adaptive conditional CDF, unchanged Simpson moments.
The failed cumulative-Simpson implementation is preserved in the original file.
"""
import math
import numpy as np
from scipy.special import logsumexp,ndtr
from scipy.integrate import quad
from scipy.optimize import brentq
from scripts.compositional_neural_mechanics.reference_posterior_math import scale_bounds,LOGK_MEAN,LOGK_BOUNDS,compare


class ConditionalCDF:
    def __init__(self,ratio,abc,track):
        self.lo,self.hi=scale_bounds(ratio);self.left=ratio[0]-LOGK_MEAN;self.right=ratio[1]-LOGK_MEAN
        self.A,self.B,self.C=map(float,abc);self.track=track
        self.use_wrench=track=='VF' and self.A>0
        if self.hi<=self.lo:raise ValueError('Empty exact prior support')
        turns=[self.lo,self.hi]
        if self.use_wrench and self.B>0 and self.B**2>24*self.A:
            root=math.sqrt(self.B**2-24*self.A)
            for y in ((self.B-root)/(4*self.A),(self.B+root)/(4*self.A)):
                if y>0 and self.lo<math.log(y)<self.hi:turns.append(math.log(y))
        turns=sorted(turns);stationary=[]
        for a,b in zip(turns[:-1],turns[1:]):
            da,db=self.derivative(a),self.derivative(b)
            if da==0:stationary.append(a)
            if da*db<0:stationary.append(brentq(self.derivative,a,b,xtol=1e-14))
        candidates=turns+stationary;self.mode=max(candidates,key=self.logdensity);self.peak=self.logdensity(self.mode)
        splits=turns+stationary
        for mode in stationary+[self.mode]:
            curvature=3.
            if self.use_wrench:curvature+=2*self.A*math.exp(2*mode)-self.B*math.exp(mode)
            if curvature>0:
                width=1/math.sqrt(curvature)
                splits.extend(mode+k*width for k in (-12,-8,-4,-2,-1,1,2,4,8,12))
        self.splits=sorted(set(t for t in splits if self.lo<t<self.hi));self.Z,self.integration_error=self.integral(self.hi)
        if not np.isfinite(self.Z) or self.Z<=0 or self.integration_error>1e-9*self.Z:raise ValueError('Adaptive conditional normalizer failed')
        self.logZ=self.peak+math.log(self.Z);self.cache={}

    def logdensity(self,s):
        v=-.5*(s*s+(s+self.left)**2+(s+self.right)**2)
        if self.use_wrench:v-=.5*self.A*(math.exp(s)-self.B/self.A)**2
        return v

    def derivative(self,s):
        v=-3*s-self.left-self.right
        if self.use_wrench:v+=-self.A*math.exp(2*s)+self.B*math.exp(s)
        return v

    def integral(self,x):
        return quad(lambda s:math.exp(self.logdensity(s)-self.peak),self.lo,x,points=[p for p in self.splits if p<x],epsabs=1e-12,epsrel=1e-11,limit=200)

    def __call__(self,x):
        if x<=self.lo:return 0.
        if x>=self.hi:return 1.
        if not self.use_wrench:
            mean=-(self.left+self.right)/3;std=1/math.sqrt(3);a=ndtr((self.lo-mean)/std);b=ndtr((self.hi-mean)/std)
            return float((ndtr((x-mean)/std)-a)/(b-a))
        if x not in self.cache:
            value,error=self.integral(x)
            if error>1e-9*self.Z:raise ValueError('Adaptive conditional CDF error exceeded tolerance')
            value/=self.Z
            if value < -1e-12 or value>1+1e-12:raise ValueError('Adaptive CDF outside range')
            self.cache[x]=float(value)
        return self.cache[x]


def integrate(nodes,log_quadrature_over_proposal,position_loglikelihood,wrench_abc,wrench_logconstant,track,points):
    if points<3 or points%2!=1 or track not in ('V','VF'):raise ValueError('Odd uniform Simpson grid and explicit track required')
    r=np.asarray(nodes);bounds=np.array([scale_bounds(v)for v in r])
    if not (bounds[:,1]>bounds[:,0]).all():raise ValueError('Outside-support nodes must be explicitly excluded')
    s=bounds[:,0,None]+(bounds[:,1]-bounds[:,0])[:,None]*np.linspace(0,1,points)
    shifts=np.stack([np.zeros(len(r)),r[:,0],r[:,1]],1);x=s[:,:,None]+shifts[:,None,:]
    prior_normalizer=(ndtr(1)-ndtr(-1))*(ndtr(LOGK_BOUNDS[1]-LOGK_MEAN)-ndtr(LOGK_BOUNDS[0]-LOGK_MEAN))**2
    logp=-.5*np.sum((x-np.array([0.,LOGK_MEAN,LOGK_MEAN]))**2,axis=-1)-1.5*np.log(2*np.pi)-np.log(prior_normalizer)
    logp+=np.asarray(position_loglikelihood)[:,None]
    if track=='VF':
        a,b,c=np.asarray(wrench_abc).T;e=np.exp(s)
        logp+=-.5*(a[:,None]*e**2-2*b[:,None]*e+c[:,None])+wrench_logconstant
    coeff=np.ones(points);coeff[1:-1:2]=4;coeff[2:-1:2]=2
    log_simpson=np.log(coeff[None,:]*(bounds[:,1]-bounds[:,0])[:,None]/(points-1)/3)
    logweight=np.asarray(log_quadrature_over_proposal)[:,None]+logp+log_simpson
    evidence=logsumexp(logweight);w=np.exp(logweight-evidence);mean=np.einsum('ij,ijk->k',w,x);difference=x-mean
    cov=np.einsum('ij,ijk,ijl->kl',w,difference,difference);sd=np.sqrt(np.diag(cov))
    conditionals=[ConditionalCDF(v,abc,track)for v,abc in zip(r,wrench_abc)];mass=w.sum(1);intervals=[]
    logquad=np.array([d.logZ+logp[i,0]-d.logdensity(bounds[i,0])for i,d in enumerate(conditionals)])
    logsimpson=logsumexp(logp+log_simpson,axis=1);mass_error=float(np.max(np.abs(np.expm1(logsimpson-logquad))))
    for j in range(3):
        def probability(value):return sum(mass[k]*cdf(float(value-shifts[k,j]))for k,cdf in enumerate(conditionals))
        intervals.append([brentq(lambda t:probability(t)-q,float(x[:,:,j].min())-1e-12,float(x[:,:,j].max())+1e-12,xtol=1e-12)for q in (.05,.95)])
    return dict(log_evidence=float(evidence),mean=mean.tolist(),covariance=cov.tolist(),sd=sd.tolist(),interval90=intervals,conditional_cdf_total=float(mass.sum()),quadrature_effective_nodes=float(1/np.sum(mass**2)),
                adaptive_normalizer_max_relative_difference=mass_error,adaptive_normalizer_gate=bool(mass_error<1e-6),
                cdf_scheme='Positive adaptive conditional density integration with stationary-point/peak-width splits; exact truncated-normal CDF forV; Simpson mixture masses unchanged')
