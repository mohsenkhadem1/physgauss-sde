"""Continuous two-latent bar posterior with full finite-prior boundaries.

For supplied independent endpoint measurements, c=1/K has conditional Gaussian
likelihood and prior density 1/c. Gaussian probability-coordinate quadrature
handles both truncated tails without clamping/redrawing a physical hypothesis.
Outer intrinsic-log-E integration is adaptive. No observations are generated.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr,ndtri_exp
from numpy.polynomial.legendre import leggauss
from experiments.campaign_c38_bar_reference import LOWER,UPPER,SIGMA_M,SIZE,TARGET_FORCE,stretch

C_LO=1/400.;C_HI=1/100.

@lru_cache(maxsize=8)
def _rule(order):
    v,w=leggauss(order)
    return (v+1)/2,w/2


def conditional_nodes(mean,std,upper=C_HI,order=24):
    """Nodes and log unnormalized Gaussian mass in [C_LO,min(upper,C_HI)]."""
    m,upper=np.broadcast_arrays(np.asarray(mean,dtype=float),np.asarray(upper,dtype=float));hi=np.minimum(upper,C_HI);valid=hi>C_LO
    # Invalid intervals get harmless nodes but exactly zero integral mass.
    hi=np.where(valid,hi,C_HI);a=(C_LO-m)/std;b=(hi-m)/std
    flip=a>0;lo=np.where(flip,-b,a);high=np.where(flip,-a,b)
    A=log_ndtr(lo);B=log_ndtr(high)
    with np.errstate(divide='ignore',invalid='ignore'):
        mass=B+np.log(-np.expm1(A-B))
    v,w=_rule(order)
    logp=np.logaddexp(A[...,None],mass[...,None]+np.log(v))
    z=ndtri_exp(logp);z=np.where(flip[...,None],-z,z);c=m[...,None]+std*z
    # Tight tails can lose an ulp in m+std*z. No material prior clipping.
    if not np.isfinite(c).all() or np.any(c<=0):raise FloatingPointError('Invalid truncated Gaussian quadrature nodes')
    return c,w,np.where(valid,mass,-np.inf)


class BarPosteriorBatch:
    """Batch shares declared forces/camera sides, never future target observations."""
    def __init__(self,observed,forces,sides,*,rtol=1e-7,atol=1e-12,conditional_order=24):
        self.y=np.atleast_2d(np.asarray(observed,dtype=float));self.F=np.asarray(forces,dtype=float);self.sides=np.asarray(sides,dtype=int)
        if self.y.ndim!=2 or self.F.ndim!=1 or self.sides.shape!=self.F.shape or self.y.shape[1]!=len(self.F) or not len(self.F) or not np.isfinite(self.y).all() or not np.isfinite(self.F).all() or (self.F<=0).any() or not np.isin(self.sides,[0,1]).all():raise ValueError('Supplied independent endpoint observations, positive forces and left/right side indicators required')
        self.n=len(self.y);self.rtol=rtol;self.atol=atol;self.order=conditional_order;self.std=SIGMA_M/np.sqrt(self.F@self.F);self.errors=[]
        self.shift=np.empty(self.n)
        for i in range(self.n):
            def f(u):
                E=np.exp(LOWER[0]+u*(UPPER[0]-LOWER[0]));s,_=stretch(np.array([E]),self.F);g=SIZE[0]*(s[0]-1)*self.sides;r=self.y[i]-g;m=r@self.F/(self.F@self.F);d=np.sum((r-m*self.F)**2)/(2*SIGMA_M**2)
                c,w,lmass=conditional_nodes(m,self.std,order=self.order)
                return float(d-lmass-np.log(np.sum(w/c)))
            result=minimize_scalar(f,bounds=(0,1),method='bounded',options={'xatol':1e-9})
            self.shift[i]=min(f(0.),f(1.),float(result.fun))
        self.Z,self.base_intervals=self.integral(self.weight,return_intervals=True)
        if not np.isfinite(self.Z).all() or (self.Z<=0).any():raise RuntimeError('Unresolved posterior normalizer')

    def parts(self,u):
        u=np.broadcast_to(u,(self.n,));E=np.exp(LOWER[0]+u*(UPPER[0]-LOWER[0]));s,_=stretch(E,self.F);g=SIZE[0]*(s-1)*self.sides[None];r=self.y-g;m=r@self.F/(self.F@self.F);d=np.sum((r-m[:,None]*self.F)**2,axis=1)/(2*SIGMA_M**2)
        return E,m,d

    def weight(self,u,upper=C_HI):
        _,m,d=self.parts(u);c,w,lmass=conditional_nodes(m,self.std,upper,self.order)
        return np.exp(self.shift-d+lmass)*np.sum(w/c,axis=-1)

    def integral(self,function,*,return_intervals=False):
        value,error,info=quad_vec(function,0.,1.,epsrel=self.rtol,epsabs=self.atol,norm='max',limit=256,full_output=True)
        self.errors.append(float(error))
        if not info.success or not np.isfinite(value).all():raise RuntimeError('Adaptive log-E integration failed; preserve case, no dropping')
        return (value,info.intervals.copy()) if return_intervals else value

    def cdf(self,kind,value):
        v=np.broadcast_to(value,(self.n,))
        if kind=='E':
            upper=np.clip((np.log(v)-LOWER[0])/(UPPER[0]-LOWER[0]),0,1)
            return self.integral(lambda t:upper*self.weight(t*upper))/self.Z
        if kind=='K':return 1-self.integral(lambda t:self.weight(t,1/v))/self.Z
        if kind=='left':return self.integral(lambda t:self.weight(t,v/TARGET_FORCE))/self.Z
        if kind=='right':
            def f(t):
                E,_,_=self.parts(t);s,_=stretch(E,np.array([TARGET_FORCE]));extension=SIZE[0]*(s[:,0]-1)
                return self.weight(t,(v-extension)/TARGET_FORCE)
            return self.integral(f)/self.Z
        raise ValueError('Unknown marginal')

    def quantile(self,kind,probability):
        if not 0<probability<1:raise ValueError('Interior probability required')
        if kind=='left':return TARGET_FORCE/self.quantile('K',1-probability)
        upper_extension=SIZE[0]*(stretch(np.array([4500.]),np.array([TARGET_FORCE]))[0][0,0]-1)
        bounds={'E':(4500.,9000.),'K':(100.,400.),'right':(TARGET_FORCE*C_LO,TARGET_FORCE*C_HI+upper_extension)}
        lo=np.full(self.n,bounds[kind][0]);hi=np.full(self.n,bounds[kind][1])
        for _ in range(23):
            mid=(lo+hi)/2;below=self.cdf(kind,mid)<probability;lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
        return (lo+hi)/2

    def moments(self):
        center=np.array([6750.,250.,.008,.014]);scale=np.array([2000.,100.,.002,.002])
        def f(u):
            E,m,d=self.parts(u);c,w,logmass=conditional_nodes(m,self.std,order=self.order);s,_=stretch(E,np.array([TARGET_FORCE]));extension=SIZE[0]*(s[:,0]-1)
            values=np.stack([np.broadcast_to(E[:,None],c.shape),1/c,TARGET_FORCE*c,TARGET_FORCE*c+extension[:,None]],axis=-1)
            delta=(values-center)/scale;mass=np.exp(self.shift-d+logmass);cw=w/c
            return np.concatenate([self.weight(u)[:,None],mass[:,None]*np.einsum('nq,nqk->nk',cw,delta),mass[:,None]*np.einsum('nq,nqk->nk',cw,delta**2)],axis=1)
        value=self.integral(f);Z=value[:,0];first=value[:,1:5]/Z[:,None];second=value[:,5:]/Z[:,None];variance=second-first**2
        bound=128*np.finfo(float).eps*(np.abs(second)+first**2+np.finfo(float).tiny)
        if np.any(variance < -bound):raise FloatingPointError('Materially negative centered posterior variance')
        return center+scale*first,scale*np.sqrt(np.maximum(variance,0))

    def field_moments(self,points):
        """Stable displacement moments for the common canonical 3D target."""
        X=np.asarray(points,dtype=float)
        if X.ndim!=2 or X.shape[1]!=3 or not np.isfinite(X).all():raise ValueError('Fixed finite material target points required')
        def f(u):
            E,m,d=self.parts(u);c,w,lmass=conditional_nodes(m,self.std,order=self.order);cw=w/c;s,y=stretch(E,np.array([TARGET_FORCE]));a=np.stack([s[:,0]-1,np.sqrt(y[:,0])-1,np.sqrt(y[:,0])-1],axis=1)
            displacement=X[None,None]*a[:,None,None];displacement=np.broadcast_to(displacement,(self.n,c.shape[1],len(X),3)).copy();displacement[:,:,:,0]+=TARGET_FORCE*c[:,:,None]
            displacement=displacement.reshape(self.n,c.shape[1],-1)/.02;mass=np.exp(self.shift-d+lmass)
            return mass[:,None]*np.concatenate([cw.sum(1)[:,None],np.einsum('nq,nqk->nk',cw,displacement),np.einsum('nq,nqk->nk',cw,displacement**2)],axis=1)
        value=self.integral(f);k=X.size;first=value[:,1:1+k]/value[:,:1];second=value[:,1+k:]/value[:,:1];variance=second-first**2
        bound=128*np.finfo(float).eps*(abs(second)+first**2+np.finfo(float).tiny)
        if np.any(variance < -bound):raise FloatingPointError('Negative canonical-field centered variance')
        return X[None]+.02*first.reshape(self.n,*X.shape),.02*np.sqrt(np.maximum(variance,0)).reshape(self.n,*X.shape)

    def quadrature_cloud(self,index,*,outer_order=8,conditional_order=16):
        """Positive quadrature atoms for design, not a claim of posterior samples.

        Mesh comes from the normalizer's adaptive outer partition. Verify its
        moments/action scores under doubled orders before physical reporting.
        """
        if type(index) is not int or not 0<=index<self.n:raise ValueError('Exact object index required')
        z,w=leggauss(outer_order);interval=self.base_intervals
        u=((interval[:,0,None]+interval[:,1,None])/2+(interval[:,1,None]-interval[:,0,None])*z/2).ravel();ow=((interval[:,1,None]-interval[:,0,None])*w/2).ravel()
        E=np.exp(LOWER[0]+u*(UPPER[0]-LOWER[0]));s,_=stretch(E,self.F);g=SIZE[0]*(s-1)*self.sides;res=self.y[index]-g;m=res@self.F/(self.F@self.F);d=np.sum((res-m[:,None]*self.F)**2,axis=1)/(2*SIGMA_M**2)
        c,cw,lmass=conditional_nodes(m,self.std,order=conditional_order);weights=ow[:,None]*np.exp(self.shift[index]-d+lmass)[:,None]*cw/c
        mass=weights.sum();theta=np.stack([np.broadcast_to(E[:,None],c.shape),1/c],axis=-1).reshape(-1,2)
        return theta,weights.ravel()/mass,{'normalizer_relative_error':float(abs(mass/self.Z[index]-1)),'positive_quadrature_atoms':len(theta),'not_MCMC_samples':True}
