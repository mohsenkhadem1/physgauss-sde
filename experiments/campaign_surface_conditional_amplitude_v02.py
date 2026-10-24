"""Positive fixed-outer conditional-amplitude inference, new numerical version.

The PWL approximation concerns the quadrature Jacobian 1/a(t), never the FE
response or material coordinates. Continuous CDFs and moment hypotheses use
exactly the same positive density. Frozen v01 results/gates remain unchanged.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import brentq
from scipy.linalg import cho_factor,cho_solve
from experiments.campaign_surface_conditional_amplitude_v01 import ConditionalAmplitudeReference as FrozenReference,gauss_axis,BOUNDS,LOG_WIDTH
from experiments.campaign_positive_amplitude_pwl_v01 import PositiveAmplitudePWL

@dataclass(frozen=True)
class AmplitudePWLPlan:
    inner:tuple=(128,256,512)
    outer:tuple=(128,256,512)
    moment_chunk:int=64

    def __post_init__(self):
        inner=tuple(self.inner);outer=tuple(self.outer)
        if len(inner)<2 or any(type(n)is not int or n not in(128,256,512)for n in inner)or any(a>=b for a,b in zip(inner,inner[1:])):raise ValueError('Strictly increasing at leasttwo declared PWLpanel orders128/256/512')
        if len(outer)<2 or any(type(n)is not int or n<4 for n in outer)or any(a>=b for a,b in zip(outer,outer[1:])):raise ValueError('Strictly increasing at leasttwo positive Gaussouter orders>=4')
        if type(self.moment_chunk)is not int or self.moment_chunk<1:raise ValueError('Positive integer moment chunk')
        object.__setattr__(self,'inner',inner);object.__setattr__(self,'outer',outer)


class ConditionalAmplitudeReferenceV02(FrozenReference):
    def __init__(self,*args,retain_hypotheses=True,**kwargs):
        super().__init__(*args,**kwargs);self.retain_hypotheses=bool(retain_hypotheses);self._state_cache=None

    def _state(self,outer,inner):
        if self._state_cache is not None and self._state_cache['key']==(outer,inner):return self._state_cache
        t,w=gauss_axis(outer);breaks=self._breaks();z=np.concatenate([a+(b-a)*t for a,b in zip(breaks,breaks[1:])]);zw=np.concatenate([(b-a)*w for a,b in zip(breaks,breaks[1:])])
        r,mu,SD,peak=self._statistics(z);lo=np.maximum(1/BOUNDS[0,1],r/BOUNDS[1,1]);hi=np.minimum(1/BOUNDS[0,0],r/BOUNDS[1,0])
        primitive=PositiveAmplitudePWL(mu,SD,lo,hi,panels=inner)
        logmass=logsumexp(primitive.log_weights,axis=1);rowlog=logmass+peak+np.log(zw)-np.log(LOG_WIDTH.prod());logZ=float(logsumexp(rowlog))
        rho=np.exp(rowlog-logZ);rho/=rho.sum();conditional=np.exp(primitive.log_weights-logmass[:,None]);conditional/=conditional.sum(axis=1)[:,None]
        a=primitive.amplitude;abar=np.sum(conditional*a,axis=1);avariance=np.sum(conditional*(a-abar[:,None])**2,axis=1)
        loga=np.log(a);logmean=np.sum(conditional*loga,axis=1);logvariance=np.sum(conditional*(loga-logmean[:,None])**2,axis=1)
        if not np.isfinite(logZ)or not np.isfinite(rho).all()or not np.isfinite(conditional).all():raise FloatingPointError('Invalid positive conditional density mass')
        state=dict(key=(outer,inner),z=z,r=r,mu=mu,SD=SD,peak=peak,primitive=primitive,conditional=conditional,rho=rho,log_evidence=logZ,amplitude_mean=abar,amplitude_variance=avariance,log_amplitude_mean=logmean,log_amplitude_variance=logvariance)
        self._state_cache=state;return state

    def cdf(self,dimension,threshold,*,outer,inner,log_evidence):
        if dimension not in(0,1)or not np.isfinite(threshold)or not 0<=threshold<=1:raise ValueError('Unit intrinsic CDF threshold')
        if threshold==0:return 0.
        if threshold==1:return 1.
        state=self._state(outer,inner)
        if abs(state['log_evidence']-log_evidence)>32*np.finfo(float).eps*max(1.,abs(log_evidence)):raise ValueError('CDF requires same full positive-density normalizer')
        q=np.exp(np.log(BOUNDS[dimension,0])+threshold*LOG_WIDTH[dimension]);cut=np.full(len(state['z']),1/q)if dimension==0 else state['r']/q
        value=float(state['rho']@state['primitive'].tail(cut))
        if not np.isfinite(value)or not 0<=value<=1+32*np.finfo(float).eps:raise FloatingPointError('Invalid continuous fixed-outer CDF; no clipping')
        return value

    def integrate(self,*,outer,inner,moment_chunk=64,keep=False,intrinsic=True):
        if type(moment_chunk)is not int or moment_chunk<1:raise ValueError('Positive moment chunk')
        s=self._state(outer,inner);z=s['z'];rho=s['rho'];m=s['amplitude_mean'];v=s['amplitude_variance'];a=s['primitive'].amplitude
        rowunit=np.column_stack(((-s['log_amplitude_mean']-np.log(BOUNDS[0,0]))/LOG_WIDTH[0],(z-s['log_amplitude_mean']-np.log(BOUNDS[1,0]))/LOG_WIDTH[1]))
        unitdelta=rowunit-rowunit[0];unitmean=rowunit[0]+rho@unitdelta;center=unitdelta-rho@unitdelta;direction=-1/LOG_WIDTH
        unitcov=center.T@(rho[:,None]*center)+float(rho@s['log_amplitude_variance'])*np.outer(direction,direction)
        g=np.asarray(self._predict(z,self.targets),float).reshape(len(z),-1)
        if g.shape[1]!=len(self.targets)*self.channels or not np.isfinite(g).all():raise ValueError('Finite target ratio predictions')
        rowmean=m[:,None]*g;delta=rowmean-rowmean[0];meandelta=np.einsum('r,rq->q',rho,delta);mean=rowmean[0]+meandelta;variance=np.einsum('r,rq->q',rho,v[:,None]*g*g+(delta-meandelta)**2)
        if np.any(variance<0)or not np.isfinite(variance).all():raise FloatingPointError('Invalid centered conditional predictive variance')
        loglikelihood=s['peak'][:,None]-.5*((a-s['mu'][:,None])/s['SD'][:,None])**2
        index=np.unravel_index(int(np.argmax(loglikelihood)),a.shape);mapa=a[index];mapz=z[index[0]];mapunit=np.array([(-np.log(mapa)-np.log(BOUNDS[0,0]))/LOG_WIDTH[0],(mapz-np.log(mapa)-np.log(BOUNDS[1,0]))/LOG_WIDTH[1]])
        result=dict(n=outer,outer=outer,inner=inner,panels=inner,weighted_hypotheses=a.size,z_segments=len(self._breaks())-1,log_evidence=s['log_evidence'],probability_sum_roundoff=abs(float(rho.sum())-1),unit_mean=unitmean,unit_covariance=unitcov,predictive_mean=mean,predictive_SD=np.sqrt(variance),plugin_unit=mapunit,plugin_prediction=mapa*g[index[0]],plugin_log_likelihood=float(loglikelihood[index]),grouped=dict(log_ratio=z.copy(),probability=rho.copy(),amplitude_mean=m.copy(),amplitude_variance=v.copy()))
        if intrinsic:
            knots=np.linspace(0,1,33);CDF=np.array([[self.cdf(d,u,outer=outer,inner=inner,log_evidence=s['log_evidence'])for u in knots]for d in range(2)])
            if np.any(np.diff(CDF,axis=1)<-1e-12):raise FloatingPointError('Nonmonotone fixed-outer positive-density CDF')
            quantiles=np.array([[brentq(lambda u:self.cdf(d,u,outer=outer,inner=inner,log_evidence=s['log_evidence'])-q,0,1,xtol=2e-8,rtol=1e-12)for q in(.05,.5,.95)]for d in range(2)])
            result.update(unit_CDF=CDF,unit_quantiles=quantiles)
        if keep and self.retain_hypotheses:
            loga=np.log(a);unit=np.stack([(-loga-np.log(BOUNDS[0,0]))/LOG_WIDTH[0],(z[:,None]-loga-np.log(BOUNDS[1,0]))/LOG_WIDTH[1]],axis=-1)
            if np.any(unit< -64*np.finfo(float).eps)or np.any(unit>1+64*np.finfo(float).eps):raise FloatingPointError('Positive amplitude nodes outside original unit prior')
            result.update(unit_grid=unit.reshape(-1,2),quadrature_probability=(rho[:,None]*s['conditional']).ravel(),inverse_amplitude=a.ravel().copy(),log_ratio=np.repeat(z,a.shape[1]),log_likelihood=loglikelihood.ravel())
        return result

    def qualify(self,*,plan=AmplitudePWLPlan()):
        answer=super().qualify(plan=plan)
        answer['diagnostics'].update(schema='conditional-amplitude-v02',CDF_method='Samefixedouterpositiveweights andsamecontinuousPWLconditionaldensity asmomenthypotheses',panel_orders=list(plan.inner),inner_orders_are_PWL_panels=True,no_probability_clipping=True,no_empirical_node_CDF=True,no_moving_threshold_outer_nodes=True,retained_raw_hypotheses=self.retain_hypotheses)
        return answer


def grouped_utilities(grouped,ratio_predict_logz,action_labels,target_labels,*,sigma=20e-6):
    """Exact dense-hypothesis covariance via conditional mean/variance identity."""
    z=np.asarray(grouped['log_ratio'],float);w=np.asarray(grouped['probability'],float);m=np.asarray(grouped['amplitude_mean'],float);v=np.asarray(grouped['amplitude_variance'],float)
    if z.ndim!=1 or any(x.shape!=z.shape for x in(w,m,v))or not all(np.isfinite(x).all()for x in(z,w,m,v))or np.any(w<0)or np.any(v<0)or np.any(m<=0)or not np.isclose(w.sum(),1,rtol=0,atol=1e-12)or sigma!=20e-6:raise ValueError('Finite normalized grouped posterior moments and fixed20um noise')
    actions=tuple(action_labels);targets=tuple(target_labels)
    if not actions or not targets:raise ValueError('Nonempty action/target labels')
    g=np.asarray(ratio_predict_logz(z,actions+targets),float)
    if g.ndim!=3 or g.shape[:2]!=(len(z),len(actions)+len(targets))or g.shape[2]==0 or not np.isfinite(g).all():raise ValueError('Finite grouped allmode response ratios')
    A=g[:,:len(actions)];T=g[:,len(actions):].reshape(len(z),-1);channels=g.shape[2]
    AM=m[:,None,None]*A;TM=m[:,None]*T;AD=AM-AM[0];TD=TM-TM[0];AC=AD-np.einsum('r,rai->ai',w,AD);TC=TD-w@TD
    aa=np.einsum('r,rai,raj->aij',w,AC,AC)+np.einsum('r,rai,raj->aij',w*v,A,A)
    ta=np.einsum('r,ri,raj->aij',w,TC,AC)+np.einsum('r,ri,raj->aij',w*v,T,A)
    tt=float(np.sum(w[:,None]*(v[:,None]*T*T+TC*TC))/T.shape[1]);scores=[]
    if not np.isfinite(tt)or not np.isfinite(aa).all()or not np.isfinite(ta).all():raise FloatingPointError('Nonfinite grouped covariance')
    for C,B in zip(aa,ta):
        factor=cho_factor(C+sigma*sigma*np.eye(channels),lower=True);reduction=float(np.sum(B*cho_solve(factor,B.T).T)/T.shape[1]);sign,logdet=np.linalg.slogdet(np.eye(channels)+C/(sigma*sigma))
        if not np.isfinite(reduction)or not np.isfinite(logdet)or sign<=0 or reduction< -1e-20 or reduction>tt+max(1e-20,tt*1e-10):raise FloatingPointError('Invalid grouped covariance risk reduction')
        scores.append(dict(expected_linear_target_MSE_reduction_m2=reduction,prior_target_MSE_m2=tt,expected_linear_remaining_target_MSE_m2=tt-reduction,Gaussian_observation_information_nats=float(.5*logdet)))
    return scores
