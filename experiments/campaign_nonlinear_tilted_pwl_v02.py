"""Stable CDF arithmetic for the UNCHANGED positive tilted PWL measure.

Knot/hypothesis/likelihood/weight construction is inherited byte-for-byte.
CDFs reuse original positive panel masses; reflection never reconstructs
panel mass from rounded 1-knots widths. Proposal coordinates0/1 have their
exact mathematical probability0/1. No clipping, mass repair or relaxed gate.
"""
import numpy as np
from experiments.campaign_nonlinear_tilted_pwl_v01 import NonlinearTiltedPWL
from experiments.campaign_surface_conditional_amplitude_v01 import log_gaussian_interval


def positive_panel_probability(knots,h,panel_mass,coordinate,*,reverse=False):
    """Normalized original PWL measure using positive partial-panel fractions.

    For panel fraction s the original integral fraction is
    s*((2-s)*h0+s*h1)/(h0+h1). The authority for the full panel is its
    stored positive quadrature mass, not a recomputed reflected width.
    Extended prefix sums reduce arithmetic error only; no density is fitted.
    """
    k=np.asarray(knots,float);h=np.asarray(h,float);mass=np.asarray(panel_mass,float);x=np.asarray(coordinate,float)
    if k.ndim!=1 or len(k)<2 or k[0]!=0 or k[-1]!=1 or np.any(np.diff(k)<=0)or h.ndim!=2 or h.shape[1]!=len(k)or mass.shape!=(len(h),len(k)-1)or x.shape!=(len(h),):raise ValueError('Matching ordered knots/rows/panel masses/coordinates')
    if not all(np.isfinite(v).all()for v in(k,h,mass,x))or np.any(h<0)or np.any(mass<0)or np.any((x<0)|(x>1)):raise ValueError('Nonnegative finite original density, masses and unit coordinates')
    if np.any((mass>0)&(h[:,:-1]==0)&(h[:,1:]==0)):raise ValueError('Positive panel mass requires a nonzero density endpoint')
    dtype=np.longdouble
    if np.finfo(dtype).eps>=np.finfo(float).eps:raise RuntimeError('Extended precision required for stable accumulated panel arithmetic')
    density=h.astype(dtype);weights=mass.astype(dtype);axis=k.astype(dtype)
    if reverse:axis=1-axis[::-1];density=density[:,::-1];weights=weights[:,::-1]
    total=np.sum(weights,axis=1,dtype=dtype)
    if np.any(total<=0):raise FloatingPointError('Positive original row mass required')
    answer=np.empty_like(x);answer[x==0]=0.;answer[x==1]=1.;inside=(x>0)&(x<1)
    if np.any(inside):
        q=x[inside].astype(dtype);H=density[inside];M=weights[inside];rows=np.arange(len(q))
        index=np.searchsorted(axis,q,side='right')-1
        s=(q-axis[index])/(axis[index+1]-axis[index])
        if np.any((s<0)|(s>1)):raise FloatingPointError('Partial coordinate outside its exact selected panel')
        left=H[rows,index];right=H[rows,index+1];scale=np.maximum(left,right);positive=scale>0
        a=np.zeros_like(left);b=np.zeros_like(right);a[positive]=left[positive]/scale[positive];b[positive]=right[positive]/scale[positive]
        fraction=s.copy();fraction[positive]=s[positive]*((2-s[positive])*a[positive]+s[positive]*b[positive])/(a[positive]+b[positive])
        cumulative=np.concatenate((np.zeros((len(q),1),dtype=dtype),np.cumsum(M,axis=1,dtype=dtype)),axis=1)
        numerator=cumulative[rows,index]+M[rows,index]*fraction
        answer[inside]=np.asarray(numerator/total[inside],float)
    if not np.isfinite(answer).all()or np.any((answer<0)|(answer>1)):raise FloatingPointError('Unresolved positive-panel CDF arithmetic; no clipping')
    return answer


class NonlinearTiltedPWLV02(NonlinearTiltedPWL):
    def _probability(self,threshold,*,tail):
        q=np.asarray(threshold,float)
        if q.shape!=self.lower.shape or not np.isfinite(q).all():raise ValueError('Finite amplitude threshold per row')
        out=np.empty_like(q);below=q<=self.lower;above=q>=self.upper;inside=~(below|above)
        out[below]=1. if tail else 0.;out[above]=0. if tail else 1.
        t=np.empty_like(q);normal=inside&~self.zero
        if np.any(normal):
            m=self.mu[normal];sd=self.SD[normal];alpha=(self.lower[normal]-m)/sd;beta=(self.upper[normal]-m)/sd;cut=(q[normal]-m)/sd
            left=log_gaussian_interval(alpha,cut);right=log_gaussian_interval(cut,beta)
            t[normal]=np.exp((right if tail else left)-np.logaddexp(left,right))
        zero=inside&self.zero
        if np.any(zero):t[zero]=(np.log(self.upper[zero]/q[zero])if tail else np.log(q[zero]/self.lower[zero]))/np.log(self.upper[zero]/self.lower[zero])
        if np.any(inside):out[inside]=positive_panel_probability(self.knots,self.h[inside],self.panel_mass[inside],t[inside],reverse=tail)
        if not np.isfinite(out).all()or np.any((out<0)|(out>1)):raise FloatingPointError('Invalid stable tilted CDF; no clipping')
        return out
