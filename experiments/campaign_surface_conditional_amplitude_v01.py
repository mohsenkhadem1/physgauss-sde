"""Prospective exact inverse-amplitude/log-ratio posterior quadrature.

No actual model, cohort, observations, execution or I/O. Every spectral mode
is retained. The Gaussian likelihood is integrated in a=1/E conditional on
z=log(K/E), using positive truncated-Gaussian-CDF nodes and the essential1/a
prior factor. Finite quadrature is qualified, never assumed exact inference.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.special import erf,log_ndtr,logsumexp
from scipy.stats import truncnorm
from scipy.optimize import brentq
from scipy.integrate import IntegrationWarning
from experiments.campaign_surface_inverse_reference_v01 import compare_grids

BOUNDS=np.array([[2000.,8000.],[5.,30.]])
LOG_WIDTH=np.log(BOUNDS[:,1]/BOUNDS[:,0])
SIGMA_M=20e-6


@dataclass(frozen=True)
class AmplitudePlan:
    inner:tuple=(32,64,128)
    outer:tuple=(128,256,512)
    moment_chunk:int=64
    def __post_init__(self):
        for values in(self.inner,self.outer):
            if len(values)<2 or any(type(n)is not int or n<4 for n in values)or any(a>=b for a,b in zip(values,values[1:])):
                raise ValueError('At least two increasing positive Gauss orders>=4')
        if type(self.moment_chunk)is not int or self.moment_chunk<1:raise ValueError('Positive moment chunk')


class RatioModal:
    """All-mode exact g(r)=sum(coeff/(1+r*lambda)); no out-of-prior decoder."""
    def __init__(self,modal,labels):
        self.modal=modal;self.labels=tuple(labels);q,r=modal.output_shape
        if r!=len(self.labels)or len(set(self.labels))!=r:raise ValueError('Unique labels per force')
        self.channels=q;self.coefficients=np.asarray(modal.coefficients).reshape(q,r,len(modal.eigenvalues));self.selected={}
    def __call__(self,ratio,labels):
        labels=tuple(labels);r=np.asarray(ratio,float)
        if r.ndim!=1 or not np.isfinite(r).all()or np.any(r<=0)or not labels:raise ValueError('Positive finite ratio vector and labels')
        if labels not in self.selected:
            ids=[self.labels.index(label)for label in labels]
            self.selected[labels]=self.coefficients[:,ids,:].transpose(1,0,2).reshape(len(labels)*self.channels,-1)
        denominator=1+r[:,None]*self.modal.eigenvalues
        if not np.isfinite(denominator).all()or np.any(denominator<=0):raise FloatingPointError('Invalid all-mode ratio denominator')
        return((1/denominator)@self.selected[labels].T).reshape(len(r),len(labels),self.channels)


def log_gaussian_interval(lower,upper):
    """log(Phi(upper)-Phi(lower)); reflected tails and cancellation-safe logdiff."""
    a,b=np.broadcast_arrays(np.asarray(lower,float),np.asarray(upper,float))
    if not np.isfinite(a).all()or not np.isfinite(b).all()or np.any(a>=b):raise ValueError('Finite strictly ordered standardized bounds')
    out=np.empty(a.shape);negative=b<=0;positive=a>=0;cross=~(negative|positive)
    for mask,left,right in((negative,a,b),(positive,-b,-a)):
        if not np.any(mask):continue
        large=log_ndtr(right[mask]);small=log_ndtr(left[mask]);delta=small-large
        if np.any(delta>=0):raise FloatingPointError('Unresolved Gaussian interval logmass')
        out[mask]=large+np.log(-np.expm1(delta))
    if np.any(cross):
        mass=.5*(erf(b[cross]/np.sqrt(2))-erf(a[cross]/np.sqrt(2)))
        if np.any(mass<=0):raise FloatingPointError('Unresolved crossing Gaussian mass')
        out[cross]=np.log(mass)
    if not np.isfinite(out).all():raise FloatingPointError('Nonfinite Gaussian interval logmass')
    return out


def gauss_axis(n):
    if type(n)is not int or n<4:raise ValueError('Declared positive Gauss order>=4')
    x,w=np.polynomial.legendre.leggauss(n);return(x+1)/2,w/2


def amplitude_nodes(mu,SD,lower,upper,order):
    """Positive physical a nodes and log kernel/a weights, without LLpeak.

    Kernel is exp(-(a-mu)^2/(2SD^2)); SD=+inf selects exact log-a prior
    substitution (g=0), whose kernel is constant1. No epsilon clipping.
    """
    mu,SD,lo,hi=np.broadcast_arrays(np.asarray(mu,float),np.asarray(SD,float),np.asarray(lower,float),np.asarray(upper,float))
    if mu.ndim!=1 or not np.isfinite(mu).all()or not np.isfinite(lo).all()or not np.isfinite(hi).all()or np.any(lo<=0)or np.any(hi<=lo)or np.any(SD<=0)or np.isnan(SD).any():raise ValueError('Valid positive amplitude intervals and SD')
    t,w=gauss_axis(order);a=np.empty((len(mu),order));lw=np.empty_like(a);zero=np.isposinf(SD);normal=~zero
    if np.any(zero):
        width=np.log(hi[zero]/lo[zero]);a[zero]=np.exp(np.log(lo[zero,None])+width[:,None]*t)
        lw[zero]=np.log(width[:,None])+np.log(w)
    if np.any(normal):
        alpha=(lo[normal]-mu[normal])/SD[normal];beta=(hi[normal]-mu[normal])/SD[normal]
        lm=log_gaussian_interval(alpha,beta)
        standardized=truncnorm.ppf(t[None,:],alpha[:,None],beta[:,None])
        a[normal]=mu[normal,None]+SD[normal,None]*standardized
        lw[normal]=np.log(SD[normal,None]*np.sqrt(2*np.pi))+lm[:,None]+np.log(w)-np.log(a[normal])
    if not np.isfinite(a).all()or not np.isfinite(lw).all()or np.any(a<=0)or np.any(a<lo[:,None])or np.any(a>hi[:,None]):raise FloatingPointError('Nonfinite/out-of-domain conditional amplitude nodes; no clipping')
    return a,lw


class ConditionalAmplitudeReference:
    def __init__(self,ratio_predict,observations,target_labels,*,sigma=SIGMA_M,log_ratio_predict=None):
        self.ratio_predict=ratio_predict;self.log_ratio_predict=log_ratio_predict;observations=list(observations);self.targets=tuple(target_labels)
        if not observations or not self.targets or sigma!=SIGMA_M:raise ValueError('Data, targets and fixed independent20um noise required')
        self.observed=tuple(dict.fromkeys(label for label,_ in observations));self.repeated=[self.observed.index(l)for l,_ in observations]
        data=np.asarray([y for _,y in observations],float)
        if data.ndim!=2 or not np.isfinite(data).all()or not data.shape[1]:raise ValueError('Finite consistent observation channels')
        self.channels=data.shape[1];self.y=data.ravel();self.sigma=sigma
        self.log_normalizer=-len(self.y)*np.log(sigma*np.sqrt(2*np.pi))

    def _predict(self,z,labels):
        return self.ratio_predict(np.exp(z),labels)if self.log_ratio_predict is None else self.log_ratio_predict(z,labels)

    def _observed_ratio(self,z):
        g=np.asarray(self._predict(np.asarray(z,float),self.observed),float)
        return g[:,self.repeated,:].reshape(len(z),-1)

    def _statistics(self,z):
        z=np.asarray(z,float);r=np.exp(z);g=np.asarray(self._predict(z,self.observed),float)
        if g.shape!=(len(z),len(self.observed),self.channels)or not np.isfinite(g).all():raise ValueError('Finite ratio predictions[ratio,label,channel]')
        g=g[:,self.repeated,:].reshape(len(z),-1);norm=np.linalg.norm(g,axis=1);zero=np.all(g==0,axis=1)
        if not np.isfinite(norm).all()or np.any((norm==0)&~zero):raise FloatingPointError('Unresolved nonzero observation response norm')
        mu=np.zeros(len(z));SD=np.full(len(z),np.inf)
        mu[~zero]=np.sum(g[~zero]*self.y,axis=1)/norm[~zero]**2;SD[~zero]=self.sigma/norm[~zero]
        residual=(self.y-mu[:,None]*g)/self.sigma
        peak=self.log_normalizer-.5*np.sum(residual**2,axis=1)
        if not np.isfinite(mu).all()or not np.isfinite(peak).all():raise FloatingPointError('Invalid conditional Gaussian statistics')
        return r,mu,SD,peak

    @staticmethod
    def _breaks(partial=None):
        Emin,Emax=BOUNDS[0];Kmin,Kmax=BOUNDS[1];low=Kmin/Emax;high=Kmax/Emin;kinks=[Kmin/Emin,Kmax/Emax]
        if partial is not None:
            dim,u=partial
            if dim not in(0,1)or not 0<u<1:raise ValueError('Interior unit threshold')
            q=np.exp(np.log(BOUNDS[dim,0])+u*LOG_WIDTH[dim])
            if dim==0:low=max(low,Kmin/q);kinks.append(Kmax/q)
            else:high=min(high,q/Emin);kinks.append(q/Emax)
        if low>=high:raise FloatingPointError('Empty partial domain')
        return np.log(np.array([low]+sorted(set(k for k in kinks if low<k<high))+[high]))

    def _nodes(self,outer,inner,partial=None):
        t,w=gauss_axis(outer);breaks=self._breaks(partial);z=np.concatenate([a+(b-a)*t for a,b in zip(breaks,breaks[1:])]);zw=np.concatenate([(b-a)*w for a,b in zip(breaks,breaks[1:])])
        r,mu,SD,peak=self._statistics(z);lo=np.maximum(1/BOUNDS[0,1],r/BOUNDS[1,1]);hi=np.minimum(1/BOUNDS[0,0],r/BOUNDS[1,0])
        if partial is not None:
            dim,u=partial;q=np.exp(np.log(BOUNDS[dim,0])+u*LOG_WIDTH[dim]);lo=np.maximum(lo,1/q if dim==0 else r/q)
        a,lw=amplitude_nodes(mu,SD,lo,hi,inner)
        lw=lw+peak[:,None]+np.log(zw[:,None])-np.log(LOG_WIDTH.prod())
        return z,r,a,lw

    def cdf(self,dimension,threshold,*,outer,inner,log_evidence):
        if dimension not in(0,1)or not np.isfinite(threshold)or not 0<=threshold<=1:raise ValueError('Unit CDF threshold')
        if threshold==0:return 0.
        if threshold==1:return 1.
        *_,lw=self._nodes(outer,inner,(dimension,float(threshold)))
        value=float(np.exp(logsumexp(lw)-log_evidence))
        if not np.isfinite(value)or not 0<=value<=1+1e-8:raise FloatingPointError('Unresolved partial integral/normalizer')
        return value

    def integrate(self,*,outer,inner,moment_chunk=64,keep=False,intrinsic=True):
        if type(moment_chunk)is not int or moment_chunk<1:raise ValueError('Positive bounded moment chunk')
        z,r,a,lw=self._nodes(outer,inner);logZ=float(logsumexp(lw));p=np.exp(lw-logZ)
        probability_sum=float(p.sum())
        if not np.isfinite(probability_sum)or probability_sum<=0:raise FloatingPointError('Invalid quadrature probability normalization')
        p/=probability_sum
        loga=np.log(a);unit=np.stack([(-loga-np.log(BOUNDS[0,0]))/LOG_WIDTH[0],(z[:,None]-loga-np.log(BOUNDS[1,0]))/LOG_WIDTH[1]],axis=-1)
        if np.any(unit< -64*np.finfo(float).eps)or np.any(unit>1+64*np.finfo(float).eps):raise FloatingPointError('Decoded amplitude outside original prior')
        unitflat=unit.reshape(-1,2);prob=p.ravel();mean=prob@unitflat;cov=(unitflat-mean).T@(prob[:,None]*(unitflat-mean))
        target=np.asarray(self._predict(z,self.targets),float).reshape(len(r),-1)
        if target.shape[1]!=len(self.targets)*self.channels or not np.isfinite(target).all():raise ValueError('Finite target ratio predictions')
        predictive_mean=(np.sum(p*a,axis=1))@target;variance=np.zeros_like(predictive_mean)
        for first in range(0,len(r),moment_chunk):
            delta=a[first:first+moment_chunk,:,None]*target[first:first+moment_chunk,None,:]-predictive_mean
            variance+=np.einsum('nm,nmd,nmd->d',p[first:first+moment_chunk],delta,delta)
        result=dict(n=outer,outer=outer,inner=inner,weighted_hypotheses=len(prob),z_segments=len(self._breaks())-1,
            log_evidence=logZ,probability_sum_roundoff=abs(probability_sum-1),unit_mean=mean,unit_covariance=cov,predictive_mean=predictive_mean,predictive_SD=np.sqrt(variance))
        if intrinsic:
            knots=np.linspace(0,1,33);CDF=np.array([[self.cdf(d,u,outer=outer,inner=inner,log_evidence=logZ)for u in knots]for d in range(2)])
            if np.any(np.diff(CDF,axis=1)<-1e-12):raise FloatingPointError('Nonmonotone partial-integral intrinsic CDF')
            quantiles=np.array([[brentq(lambda u:self.cdf(d,u,outer=outer,inner=inner,log_evidence=logZ)-q,0,1,xtol=2e-8,rtol=1e-12)for q in(.05,.5,.95)]for d in range(2)])
            result.update(unit_CDF=CDF,unit_quantiles=quantiles)
        if keep:
            _,mu,SD,peak=self._statistics(z)
            loglikelihood=peak[:,None]-.5*((a-mu[:,None])/SD[:,None])**2
            result.update(unit_grid=unitflat,quadrature_probability=prob,inverse_amplitude=a.ravel(),ratio=np.repeat(r,inner),
                log_ratio=np.repeat(z,inner),log_likelihood=loglikelihood.ravel())
        return result

    def qualify(self,*,plan=AmplitudePlan()):
        """Separate inner and outer refinements; preserve all coarse failures.

        Qualify only the latest successive pair on BOTH axes under the same
        intrinsic/prediction gates as the tensor reference. Independent adaptive original-a evidence and six quantile-CDF checks are
        also mandatory; physical/catalog admission remains a separate requirement.
        """
        requested=list(dict.fromkeys([(plan.outer[-1],n)for n in plan.inner]+[(n,plan.inner[-1])for n in plan.outer]));results={};levels=[]
        for outer,inner in requested:
            try:
                row=self.integrate(outer=outer,inner=inner,moment_chunk=plan.moment_chunk,keep=(outer,inner)==(plan.outer[-1],plan.inner[-1]));results[outer,inner]=row
                levels.append(dict(outer=outer,inner=inner,status='computed',log_evidence=row['log_evidence'],weighted_hypotheses=row['weighted_hypotheses']))
            except(FloatingPointError,ValueError,np.linalg.LinAlgError)as error:levels.append(dict(outer=outer,inner=inner,status='unresolved',reason=str(error)))
        comparisons={}
        for axis,orders in(('inner',plan.inner),('outer',plan.outer)):
            rows=[]
            for c,f in zip(orders,orders[1:]):
                ck=(plan.outer[-1],c)if axis=='inner'else(c,plan.inner[-1]);fk=(plan.outer[-1],f)if axis=='inner'else(f,plan.inner[-1])
                if ck not in results or fk not in results:check=dict(passed=False,reason='Missing unresolved level')
                else:
                    check=compare_grids(results[ck],results[fk]);difference=float(np.max(abs(results[ck]['unit_mean']-results[fk]['unit_mean'])))
                    check.update(maximum_unit_mean_difference=difference,passed=check['passed']and difference<=.001)
                rows.append(dict(coarse=c,fine=f,**check))
            comparisons[axis]=rows
        refined=all(rows[-1]['passed']for rows in comparisons.values());final=results.get((plan.outer[-1],plan.inner[-1]))
        adaptive=dict(passed=False,reason='Missing final intrinsic result')
        if final is not None:
            try:
                from experiments.campaign_amplitude_adaptive_oracle_v01 import adaptive_integral
                independent={};relative_errors={}
                with warnings.catch_warnings():
                    warnings.simplefilter('error',IntegrationWarning)
                    for eps in(1e-6,1e-8):
                        full=adaptive_integral(self._observed_ratio,self.y,sigma=self.sigma,epsrel=eps)
                        CDF=np.empty((2,3));relative=max(full['estimated_outer_relative_error'],full['maximum_estimated_inner_relative_error'])
                        for dim in(0,1):
                            for j,q in enumerate((.05,.5,.95)):
                                threshold=np.exp(np.log(BOUNDS[dim,0])+final['unit_quantiles'][dim,j]*LOG_WIDTH[dim])
                                partial=adaptive_integral(self._observed_ratio,self.y,sigma=self.sigma,CDF=('E'if dim==0 else'K',float(threshold)),epsrel=eps)
                                CDF[dim,j]=np.exp(partial['log_integral']-full['log_integral'])
                                relative=max(relative,partial['estimated_outer_relative_error'],partial['maximum_estimated_inner_relative_error'])
                        if not np.isfinite(full['log_integral'])or not np.isfinite(CDF).all()or not np.isfinite(relative)or np.any((CDF<0)|(CDF>1+1e-8)):
                            raise FloatingPointError('Nonfinite/out-of-domain independent adaptive result')
                        independent[eps]=(full['log_integral'],CDF);relative_errors[str(eps)]=float(relative)
                difference=abs(independent[1e-8][0]-final['log_evidence']);cdf_error=float(np.max(abs(independent[1e-8][1]-[.05,.5,.95])))
                log_refinement=abs(independent[1e-6][0]-independent[1e-8][0]);cdf_refinement=float(np.max(abs(independent[1e-6][1]-independent[1e-8][1])))
                adaptive=dict(passed=bool(difference<=.01 and cdf_error<=.005 and log_refinement<=.01 and cdf_refinement<=.005 and max(relative_errors.values())<=1e-5),
                    log_evidence_difference=float(difference),maximum_CDF_error=cdf_error,CDF_checks_per_epsrel=6,epsrel=[1e-6,1e-8],
                    adaptive_refinement_log_evidence_difference=float(log_refinement),adaptive_refinement_maximum_CDF_difference=cdf_refinement,
                    maximum_QUADPACK_relative_error_estimates=relative_errors,scope='Independent original-a adaptive quadrature and two-tolerance agreement; estimates are not rigorous error bounds or proof against missed peaks')
            except(FloatingPointError,ValueError,RuntimeError,IntegrationWarning)as error:adaptive=dict(passed=False,reason=type(error).__name__+': '+str(error))
        qualified=refined and adaptive['passed']
        return dict(qualified=bool(qualified),local_final=final,diagnostics=dict(schema='conditional-amplitude-v01',qualified=bool(qualified),levels=levels,successive=comparisons,independent_adaptive=adaptive,
            prior='da dz/(a ΔlogE ΔlogK), original independent loguniform E/K prior',
            all_modes_retained=True,no_parameter_snapping_or_interpolation=True,no_quadrature_ESS=True,
            conditional_on_finite_linear_model=True,independent_adaptive_per_posterior_required=True,physical_admission_separate=True,
            gates=dict(logZ=.01,CDF=.005,quantile=.002,unit_mean=.001,target_absolute_m=2e-6,target_relative_SD=.01)))


class ExactRatioMemo:
    """Optional bounded exact-byte LRU for a single immutable model fingerprint.

    No interpolation or parameter snapping. Scalar adaptive-oracle queries are
    excluded by default; their many tiny keys would evict useful fixed arrays.
    """
    def __init__(self,callback,*,model_fingerprint,maximum_bytes=256*1024**2,minimum_rows=16):
        from collections import OrderedDict
        if not isinstance(model_fingerprint,str)or len(model_fingerprint)!=64 or maximum_bytes<0 or minimum_rows<1:raise ValueError('Explicit model fingerprint and bounded memo budget')
        self.callback=callback;self.model_fingerprint=model_fingerprint;self.maximum_bytes=int(maximum_bytes);self.minimum_rows=minimum_rows
        self.entries=OrderedDict();self.allocated_bytes=0;self.hits=0;self.misses=0;self.evictions=0
    def __call__(self,log_ratio,labels):
        z=np.ascontiguousarray(log_ratio,dtype=np.float64);labels=tuple(labels)
        if z.ndim!=1 or not np.isfinite(z).all():raise ValueError('Finite log-ratio vector')
        key=(labels,z.shape,z.tobytes())
        if len(z)>=self.minimum_rows and key in self.entries:
            self.hits+=1;value,cost=self.entries.pop(key);self.entries[key]=(value,cost);return value
        self.misses+=1;value=np.asarray(self.callback(z,labels),float)
        if value.ndim!=3 or value.shape[:2]!=(len(z),len(labels))or not np.isfinite(value).all():raise ValueError('Finite ratio output layout')
        cost=value.nbytes+len(key[2])+4096 # Conservative Python/container overhead perentry.
        if len(z)>=self.minimum_rows and cost<=self.maximum_bytes:
            while self.entries and self.allocated_bytes+cost>self.maximum_bytes:
                _,(_,old)=self.entries.popitem(last=False);self.allocated_bytes-=old;self.evictions+=1
            value=np.ascontiguousarray(value).copy();value.flags.writeable=False
            self.entries[key]=(value,cost);self.allocated_bytes+=cost
        return value
    def diagnostics(self):
        return dict(model_fingerprint=self.model_fingerprint,maximum_bytes=self.maximum_bytes,accounted_bytes=self.allocated_bytes,
            entries=len(self.entries),hits=self.hits,misses=self.misses,evictions=self.evictions,minimum_rows=self.minimum_rows,interpolation=False)
