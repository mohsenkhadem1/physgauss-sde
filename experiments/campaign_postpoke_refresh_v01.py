"""Exact finite likelihood and genuine prefix-to-joint posterior SMC bridge.

Cached particles approximate the prefix posterior. They are never assigned a
continuous proposal density or treated as exact independent posterior samples.
Resampling is followed by invariant full-target MH moves; independent fresh
joint-likelihood references and finite-chain diagnostics remain necessary.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy.special import logsumexp
from experiments import campaign_diverse_force_cohort_v01 as cohort
from experiments import campaign_tempered_reference as temper


class JointObservationForward(cohort.old.ArbitraryForceForward):
    """Original A and one reset poke, shared matrix, all24 original steps."""
    def __init__(self,poke_force,observed_A,observed_poke,*,chunk=1024):
        super().__init__();self.chunk=chunk
        force=np.asarray(poke_force,float);a=np.asarray(observed_A,float);b=np.asarray(observed_poke,float)
        if force.shape!=(60,5,3)or a.shape!=(12,)or b.shape!=(12,)or not all(np.isfinite(v).all()for v in(force,a,b)):raise ValueError('One known force and two finite12-observation vectors')
        self.held=torch.tensor(np.stack([cohort.old.original.known_force('A'),force]).reshape(2,60,15).transpose(1,2,0)[:24]);self.observed_A=a.copy();self.observed_poke=b.copy()

    def predictions(self,unit):
        theta=cohort.physical(unit);result=[];m=self.mass;h=cohort.old.original.H
        for first in range(0,len(theta),self.chunk):
            values=theta[first:first+self.chunk];n=len(values);center=np.column_stack([values[:,1],np.full(n,.02),np.zeros(n)])
            logw=np.log(self.areas)[None]-.5*np.sum((self.X[None]-center[:,None])**2,2)/values[:,2,None]**2
            weights=np.exp(logw-logsumexp(logw,axis=1)[:,None]);diagonal=torch.tensor(np.repeat(weights,3,axis=1));C=.05*diagonal
            K=torch.tensor(values[:,0,None,None])*self.Kunit+torch.diag_embed(diagonal*torch.tensor(values[:,3,None]));chol=torch.linalg.cholesky(K+torch.diag_embed(m[None]/h**2+C/h))
            u=torch.zeros((n,15,2),dtype=torch.float64);v=u.clone();saved=[]
            for step in range(24):
                rhs=self.held[step][None]+m[None,:,None]*(u/h**2+v/h)+C[:,:,None]*u/h
                new=torch.cholesky_solve(rhs,chol);v=(new-u)/h;u=new
                if step+1 in(12,24):saved.append(u.reshape(n,5,3,2)[:,cohort.old.original.OBS_NODES].numpy().transpose(0,3,1,2).copy())
            result.append(np.stack(saved,axis=2).reshape(n,2,12))
        return np.concatenate(result)

    def log_components(self,unit):
        y=self.predictions(unit);sigma=cohort.old.original.SIGMA
        ll=-.5*np.sum(((y-np.stack([self.observed_A,self.observed_poke])[None])/sigma)**2,axis=2)-12*np.log(sigma*np.sqrt(2*np.pi))
        return ll[:,0],ll[:,1]

    def joint_log_likelihood(self,unit):
        a,b=self.log_components(unit);return a+b


def proposal_logq(x,mean,inverse):
    delta=np.asarray(x)-mean
    return -.5*np.einsum('bi,ij,bj->b',delta,inverse,delta)


def mh_log_ratio(old,new,old_A,new_A,old_poke,new_poke,beta,kind,mean,inverse):
    """Uniform original prior; exact untruncated Gaussian independence ratio."""
    if not 0<=beta<=1:raise ValueError('Bridge beta within[0,1]')
    correction=np.where(np.asarray(kind)==0,proposal_logq(old,mean,inverse)-proposal_logq(new,mean,inverse),0.)
    result=np.asarray(new_A)-old_A+correction
    return result+beta*(np.asarray(new_poke)-old_poke)if beta else result


@dataclass(frozen=True)
class BridgeResult:
    unit:np.ndarray
    prefix_log_likelihood:np.ndarray
    poke_log_likelihood:np.ndarray
    log_conditional_evidence_estimate:float
    stages:list
    seed:int
    initialization_approximate:bool=True


def bridge_smc(initial,log_components,proposal_mean,proposal_covariance,*,seed,ess_fraction=.8,rejuvenation_steps=16,max_stages=64,rw_scale=.5):
    """SMC bridge p(A)*L(poke)^beta with fixed, valid full-target MH moves.

    The initial prefix cloud is approximate; subsequent kernels target exact
    logL(A)+beta*logL(poke) with the unchanged uniform intrinsic box prior.
    Local correlated Gaussian proposals use OUT-OF-BOX REJECTION. Independence
    Gaussian proposals retain q(old)/q(new), and uniform-prior proposals cancel.
    No covariance adaptation, jitter, coordinate reflection or pilot truth.
    """
    x=np.array(initial,float,copy=True);mean=np.array(proposal_mean,float,copy=True);cov=np.array(proposal_covariance,float,copy=True)
    if x.ndim!=2 or len(x)<2 or mean.shape!=(x.shape[1],)or cov.shape!=(x.shape[1],x.shape[1])or not np.isfinite(mean).all()or not np.isfinite(cov).all()or not np.allclose(cov,cov.T,atol=1e-15,rtol=1e-13)or not np.isfinite(x).all()or np.any((x<0)|(x>1))or not 0<ess_fraction<1 or not isinstance(rejuvenation_steps,int)or rejuvenation_steps<1 or not isinstance(max_stages,int)or max_stages<1 or not 0<rw_scale<=1:raise ValueError('Finite in-box initial particles, fixed SPD proposal, valid prospective settings')
    L=np.linalg.cholesky(cov);inverse=np.linalg.inv(cov);rng=np.random.default_rng(seed);n,d=x.shape
    def evaluate(unit):
        a,b=log_components(unit);a=np.asarray(a,float);b=np.asarray(b,float)
        if a.shape!=(len(unit),)or b.shape!=a.shape or not np.isfinite(a).all()or not np.isfinite(b).all():raise FloatingPointError('Both finite normalized likelihoods required')
        return a,b
    a,b=evaluate(x);beta=0.;logZ=0.;stages=[];ancestry=np.arange(n)
    while beta<1:
        if len(stages)>=max_stages:raise RuntimeError('Bridge max_stages reached without posterior')
        def at(delta):
            w,normalizer=temper.normalized_log_weights(delta*b);return w,normalizer-np.log(n),float(1/np.square(w).sum())
        remaining=1-beta;delta=remaining;weights,inc,ess=at(delta)
        if ess<ess_fraction*n:
            lo,hi=0.,remaining
            for _ in range(50):
                mid=(lo+hi)/2
                if at(mid)[2]<ess_fraction*n:hi=mid
                else:lo=mid
            delta=hi;weights,inc,ess=at(delta)
        if delta<1e-12:raise RuntimeError('Bridge temperature stalled')
        beta=min(1.,beta+delta);logZ+=inc;ids=temper.systematic_resample(weights,rng);x=x[ids];a=a[ids];b=b[ids];ancestry=ancestry[ids]
        proposed=np.zeros(3,int);accepted=proposed.copy();outside=proposed.copy()
        for _ in range(rejuvenation_steps):
            kind=rng.choice(3,size=n,p=[.5,.45,.05]);candidate=x+rw_scale*(rng.normal(size=x.shape)@L.T)
            independent=mean+rng.normal(size=x.shape)@L.T;candidate[kind==0]=independent[kind==0];candidate[kind==2]=rng.uniform(size=(int((kind==2).sum()),d))
            valid=((candidate>=0)&(candidate<=1)).all(1);na=np.full(n,-np.inf);nb=na.copy()
            if valid.any():na[valid],nb[valid]=evaluate(candidate[valid])
            ratio=mh_log_ratio(x,candidate,a,na,b,nb,beta,kind,mean,inverse)
            accept=valid&(np.log(rng.random(n))<np.minimum(0.,ratio));x[accept]=candidate[accept];a[accept]=na[accept];b[accept]=nb[accept]
            for k in range(3):proposed[k]+=int((kind==k).sum());accepted[k]+=int(((kind==k)&accept).sum());outside[k]+=int(((kind==k)&~valid).sum())
        stages.append(dict(beta=beta,delta_beta=delta,pre_resampling_ESS=ess,log_evidence_increment=float(inc),unique_initial_ancestors=int(len(np.unique(ancestry))),proposed=proposed.tolist(),accepted=accepted.tolist(),outside_rejected=outside.tolist(),proposal_fixed=True))
    return BridgeResult(x,a,b,float(logZ),stages,int(seed))


def prediction_summary(forward,unit,*,chain=False):
    """Stream all posterior fields into positive100-draw moments, no thinning.

    Preserve full18 original physical/fixture/Aend/Bend functions and their
    diagnostics while avoiding a512k×135 resident target matrix.
    """
    from experiments.campaign_force_prediction_precision_v01 import combine,delta_mcse
    from physgauss.chain_diagnostics import diagnose_quantities
    x=np.asarray(unit,float);shape=x.shape
    if chain and(shape!=(8,64000,4)):raise ValueError('Exactly8 full64000-draw chains')
    flat=x.reshape(-1,4);means=[];variances=[];counts=[];functions=[];width=1600 if chain else 512
    for first in range(0,len(flat),width):
        values=flat[first:first+width];target,pokes,f=cohort.joint_features(forward,cohort.physical(values),with_functions=True);del pokes
        functions.append(f)
        if chain:
            t=target.reshape(-1,100,135);mu=t.mean(1);var=((t-mu[:,None])**2).mean(1)
            means.extend(mu);variances.extend(var);counts.extend([100]*len(mu))
        else:
            mu=target.mean(0);means.append(mu);variances.append(((target-mu)**2).mean(0));counts.append(len(target))
    mass=np.asarray(counts,float)[:,None]/len(flat);mean=np.asarray(means)[:,None];var=np.asarray(variances)[:,None]
    _,pooled,pv=combine(mass,mean,var,[list(range(len(mass)))]);f=np.concatenate(functions)
    result=dict(mean=pooled[0,0],variance=pv[0,0],function_mean=f.mean(0),function_std=f.std(0))
    if chain:
        estimates=[];blocks_per_chain=640
        for k in cohort.BATCHES:
            group=k//100;groups=[list(range(i,i+group))for i in range(0,len(mass),group)]
            estimates.append(delta_mcse(*combine(mass,mean,var,groups)))
        estimates.append(delta_mcse(*combine(mass,mean,var,[list(range(i*blocks_per_chain,(i+1)*blocks_per_chain))for i in range(8)])))
        result.update(mean_MCSE=np.maximum.reduce([r['mean_MCSE'][0]for r in estimates]),SD_MCSE=np.maximum.reduce([r['SD_MCSE'][0]for r in estimates]))
        result['field_MCSE']=cohort.moment_gate(result);functions=f.reshape(8,64000,18)
        diagnostics=diagnose_quantities(torch.tensor(functions),['function_'+str(i)for i in range(18)])
        se=[]
        for k in cohort.BATCHES:
            b=functions.reshape(8,64000//k,k,18).mean(2).reshape(-1,18);se.append(b.std(0,ddof=1)/np.sqrt(len(b)))
        error=np.max(se,axis=0);sd=result['function_std'];ratio=np.divide(error,sd,out=np.where(error==0,0.,np.inf),where=sd>0)
        result.update(function_diagnostics=diagnostics,function_maximum_relative_MCSE=float(ratio.max()),function_MCSE_passed=bool(diagnostics['all_monitored_quantities_pass']and ratio.max()<=.05))
    return result
