"""Exact target moments of a positive hierarchical LINEAR posterior bank.

Every amplitude hypothesis is retained. Gaussian scalar outcome likelihood is
applied at each amplitude node, not at a conditional mean or via a Gaussian
posterior approximation. Only target-moment algebra is grouped. Physical and
continuous-posterior admission remain separate caller responsibilities.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp,ndtr
from experiments.campaign_layout_mixture_decision_v01 import _positive_total

@dataclass
class GroupedLinearBank:
    row_probability:np.ndarray
    category:np.ndarray
    amplitude:np.ndarray
    conditional:np.ndarray
    candidate_ratio:np.ndarray
    target_ratio:np.ndarray
    def __post_init__(self):
        self.row_probability=np.asarray(self.row_probability,float);self.category=np.asarray(self.category);self.amplitude=np.asarray(self.amplitude,float);self.conditional=np.asarray(self.conditional,float);self.candidate_ratio=np.asarray(self.candidate_ratio,float);self.target_ratio=np.asarray(self.target_ratio,float);R=len(self.row_probability)
        if self.row_probability.shape!=(R,)or not R or self.category.shape!=(R,)or self.amplitude.ndim!=2 or self.amplitude.shape[0]!=R or not self.amplitude.shape[1]or self.conditional.shape!=self.amplitude.shape or any(v.ndim!=2 or v.shape[0]!=R or not v.shape[1]for v in(self.candidate_ratio,self.target_ratio)):raise ValueError('Finite grouped posterior dimensions')
        if any(not np.isfinite(v).all()for v in(self.row_probability,self.amplitude,self.conditional,self.candidate_ratio,self.target_ratio))or np.any(self.row_probability<0)or np.any(self.amplitude<=0)or np.any(self.conditional<0)or self.category.dtype.kind not in'iu' or np.any(self.category<0):raise ValueError('Positive finite grouped posterior arrays')
        self.row_probability=self.row_probability/_positive_total(self.row_probability)
        with np.errstate(over='ignore'):s=self.conditional.sum(axis=1)
        if not np.isfinite(s).all()or np.any(s<=0):raise ValueError('Every conditional amplitude row has finite positive mass')
        self.conditional=self.conditional/s[:,None];self.logconditional=np.full(self.conditional.shape,-np.inf);np.log(self.conditional,out=self.logconditional,where=self.conditional>0)
    @property
    def hypotheses(self):return self.amplitude.size
    def prior_moments(self):
        m=np.sum(self.conditional*self.amplitude,axis=1);v=np.sum(self.conditional*(self.amplitude-m[:,None])**2,axis=1);T=m[:,None]*self.target_ratio;offset=T[0];delta=T-offset;mean=offset+self.row_probability@delta;variance=np.sum(self.row_probability[:,None]*(v[:,None]*self.target_ratio**2+(delta-self.row_probability@delta)**2),axis=0)
        between=0.
        for k in np.unique(self.category):
            take=self.category==k;mass=self.row_probability[take].sum()
            if mass>0:mu=(self.row_probability[take]/mass)@T[take];between+=mass*float(np.mean((mu-mean)**2))
        return dict(mean=mean,SD=np.sqrt(variance),total_variance=float(np.mean(variance)),between_layout_variance=between)
    def response_bounds(self,ratio):
        lo=self.amplitude.min(axis=1)[:,None];hi=self.amplitude.max(axis=1)[:,None];a=lo*ratio;b=hi*ratio
        return np.minimum(a,b).min(axis=0),np.maximum(a,b).max(axis=0)


def outcome_risk(bank,action,y,sigma,*,row_chunk=64):
    y=np.asarray(y,float)
    if y.ndim!=1 or not np.isfinite(y).all()or not np.isfinite(sigma)or sigma<=0 or type(row_chunk)is not int or row_chunk<1 or not 0<=action<bank.candidate_ratio.shape[1]:raise ValueError('Finite scalar noise/outcomes and bounded grouped chunks')
    R=len(bank.row_probability);Q=len(y);rowlog=np.empty((R,Q));m=np.empty((R,Q));v=np.empty((R,Q))
    for begin in range(0,R,row_chunk):
        end=min(begin+row_chunk,R);a=bank.amplitude[begin:end];ll=bank.logconditional[begin:end,:,None]-.5*((a[:,:,None]*bank.candidate_ratio[begin:end,action,None,None]-y)/sigma)**2-np.log(sigma*np.sqrt(2*np.pi));peak=np.max(ll,axis=1);w=np.exp(ll-peak[:,None,:]);mass=w.sum(axis=1)
        if not np.isfinite(peak).all()or np.any(mass<=0):raise FloatingPointError('Unresolved positive conditional outcome mass')
        anchor=a[:,0,None];delta=a[:,:,None]-anchor[:,None,:];mean=anchor+np.sum(w*delta,axis=1)/mass;variance=np.sum(w*(a[:,:,None]-mean[:,None,:])**2,axis=1)/mass
        rowlog[begin:end]=peak+np.log(mass);m[begin:end]=mean;v[begin:end]=variance
    logrow=np.full(R,-np.inf);np.log(bank.row_probability,out=logrow,where=bank.row_probability>0);joint=logrow[:,None]+rowlog;lp=logsumexp(joint,axis=0);rho=np.exp(joint-lp);rho/=rho.sum(axis=0)[None,:];answer=np.empty(Q)
    for j in range(Q):
        T=m[:,j,None]*bank.target_ratio;offset=T[0];delta=T-offset;dmean=rho[:,j]@delta;answer[j]=np.sum(rho[:,j,None]*(v[:,j,None]*bank.target_ratio**2+(delta-dmean)**2))/bank.target_ratio.shape[1]
    if not np.isfinite(answer).all()or np.any(answer<0):raise FloatingPointError('Invalid centered full predictive variance')
    return lp,answer


def expected_risk(bank,action,sigma,*,order=16,row_chunk=64,outcome_chunk=8,maximum_outcome_nodes=200000):
    if type(order)is not int or order<2 or type(outcome_chunk)is not int or outcome_chunk<1 or type(maximum_outcome_nodes)is not int or maximum_outcome_nodes<1 or not np.isfinite(sigma)or sigma<=0:raise ValueError('Positive bounded grouped outcome plan')
    lower,upper=bank.response_bounds(bank.candidate_ratio[:,action,None]);a=float(lower[0]-8*sigma);b=float(upper[0]+8*sigma);panels=max(1,int(np.ceil((b-a)/(.5*sigma))))
    if panels*order>maximum_outcome_nodes:raise MemoryError('Fixed outcome node budget exceeded')
    x,w=np.polynomial.legendre.leggauss(order);edges=np.linspace(a,b,panels+1);half=np.diff(edges)/2;nodes=((edges[:-1]+edges[1:])[:,None]/2+half[:,None]*x).ravel();weights=(half[:,None]*w).ravel();risk=mass=0.
    for start in range(0,len(nodes),outcome_chunk):
        stop=start+outcome_chunk;lp,v=outcome_risk(bank,action,nodes[start:stop],sigma,row_chunk=row_chunk);p=weights[start:stop]*np.exp(lp);risk+=float(p@v);mass+=float(p.sum())
    lo,hi=bank.response_bounds(bank.target_ratio);tail=float(2*ndtr(-8));prior=bank.prior_moments()['total_variance']
    if not np.isfinite(risk)or risk<0 or not 0<mass<=1+1e-10:raise FloatingPointError('Unresolved grouped outcome integration')
    return dict(expected_target_variance=risk,prior_target_variance=prior,variance_reduction=prior-risk,predictive_mass=mass,omitted_mass_bound=tail,omitted_risk_bound=float(tail*np.mean((hi-lo)**2)/4),outcome_nodes=len(nodes),order=order)


def decision_bank(references,action_labels,target_labels,*,outer,amplitude_order):
    """NEW decision quadrature, not a claim of identical frozen PWL atoms.

    Positive truncated-Gaussian-CDF Gauss nodes integrate the original1/a
    conditional prior. References remain the independently qualified continuous
    posterior witnesses. Every declared category must be present.
    """
    from experiments.campaign_surface_conditional_amplitude_v01 import amplitude_nodes,gauss_axis,BOUNDS,LOG_WIDTH
    from experiments.campaign_layout_mixture_decision_v01 import layout_probabilities
    if len(references)!=4 or amplitude_order not in(8,16,32,64)or outer not in(256,512):raise ValueError('Four categories and predeclared decision quadrature')
    arrays={key:[]for key in('amplitude','conditional','candidate_ratio','target_ratio')};rowlogs=[];categories=[];logZ=[];moments=[]
    for k,reference in enumerate(references):
        t,w=gauss_axis(outer);breaks=reference._breaks();z=np.concatenate([a+(b-a)*t for a,b in zip(breaks,breaks[1:])]);zw=np.concatenate([(b-a)*w for a,b in zip(breaks,breaks[1:])])
        ratio,mu,SD,peak=reference._statistics(z);
        lower=np.maximum(1/BOUNDS[0,1],ratio/BOUNDS[1,1]);upper=np.minimum(1/BOUNDS[0,0],ratio/BOUNDS[1,0]);a,lw=amplitude_nodes(mu,SD,lower,upper,amplitude_order);lm=logsumexp(lw,axis=1);rl=lm+peak+np.log(zw)-np.log(LOG_WIDTH.prod());lz=float(logsumexp(rl));rp=np.exp(rl-lz);cp=np.exp(lw-lm[:,None]);g=np.asarray(reference._predict(z,tuple(action_labels)+tuple(target_labels)),float)
        if g.shape!=(len(z),len(action_labels)+len(target_labels),1):raise ValueError('Scalar action/target ratio channels')
        cr=g[:,:len(action_labels),0];tr=g[:,len(action_labels):,0];single=GroupedLinearBank(rp,np.full(len(z),k,dtype=int),a,cp,cr,tr);moments.append(single.prior_moments());rowlogs.append(rl);logZ.append(lz);categories.append(np.full(len(z),k,dtype=int))
        for key,value in [('amplitude',a),('conditional',cp),('candidate_ratio',cr),('target_ratio',tr)]:arrays[key].append(value)
    rho=layout_probabilities(logZ,[.25]*4);row_probability=np.concatenate([rho[k]*np.exp(rowlogs[k]-logZ[k])for k in range(4)]);bank=GroupedLinearBank(row_probability,np.concatenate(categories),**{key:np.concatenate(value)for key,value in arrays.items()})
    return bank,dict(outer=outer,amplitude_order=amplitude_order,log_evidence=logZ,layout_probability=rho.tolist(),category_moments=moments,weighted_hypotheses=bank.hypotheses,measure='New positive original-prior decision quadrature; separately validated, not frozen PWL hypothesis equivalence')


def reference_moment_checks(metadata,finals):
    if len(finals)!=4 or len(metadata['category_moments'])!=4:raise ValueError('All four reference categories retained')
    errors=[]
    for z,m,f in zip(metadata['log_evidence'],metadata['category_moments'],finals):
        evidence=abs(z-f['log_evidence']);mean=float(np.max(abs(m['mean']-f['predictive_mean'])));sd=float(np.max(abs(m['SD']-f['predictive_SD'])));errors.append(dict(log_evidence=evidence,mean_m=mean,SD_m=sd,passed=bool(evidence<=.01 and mean<=2e-6 and sd<=2e-6)))
    from experiments.campaign_layout_mixture_decision_v01 import layout_probabilities
    newp=layout_probabilities(metadata['log_evidence'],[.25]*4);oldp=layout_probabilities([f['log_evidence']for f in finals],[.25]*4)
    def mixture(probability,means,SD):
        mean=probability@means;variance=probability@(SD**2+(means-mean)**2);return mean,np.sqrt(variance)
    newmean,newsd=mixture(newp,np.array([m['mean']for m in metadata['category_moments']]),np.array([m['SD']for m in metadata['category_moments']]))
    oldmean,oldsd=mixture(oldp,np.array([f['predictive_mean']for f in finals]),np.array([f['predictive_SD']for f in finals]))
    mixmean=float(np.max(abs(newmean-oldmean)));mixsd=float(np.max(abs(newsd-oldsd)));tv=float(np.sum(abs(newp-oldp))/2)
    return dict(passed=all(e['passed']for e in errors)and mixmean<=2e-6 and mixsd<=2e-6 and tv<=.01,per_category=errors,mixture_mean_error_m=mixmean,mixture_SD_error_m=mixsd,layout_TV=tv,additional_to_full_reference_qualification=True)
