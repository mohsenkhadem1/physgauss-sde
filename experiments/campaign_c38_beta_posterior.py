"""Fixed small bounded posterior density for the C38 idealized log-NH family.

16 product-Beta components represent joint correlation through a mixture.
No RGBD/general-layout/dynamic inference claim. Marginal probabilities are
analytic mixtures of betainc, not quantiles of resampled finite particles.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from scipy.special import betainc,logsumexp
from scipy.linalg import eigh_tridiagonal
from experiments.campaign_c38_bar_reference import prior_physical,oracle_positions,material_quadrature,TARGET_FORCE


def canonical_inputs(forces,sides,observed,mask):
    """One/two acquired readings, order invariant, no parameter/role arguments.

    Invalid unacquired slots are ignored before checking values. Sorting only
    acquired (force,side,measurement) handles repeated identical sensor types.
    """
    f=np.asarray(forces);s=np.asarray(sides);y=np.asarray(observed);m=np.asarray(mask)
    if f.ndim!=2 or f.shape[1]!=2 or s.shape!=f.shape or y.shape!=f.shape or m.shape!=f.shape or m.dtype!=np.bool_ or not np.isin(m.sum(1),[1,2]).all():raise ValueError('One/two explicit acquired measurement slots required')
    result=np.zeros((len(f),10))
    for i in range(len(f)):
        ids=np.flatnonzero(m[i]);ff=f[i,ids].astype(float);ss=s[i,ids].astype(float);yy=y[i,ids].astype(float)
        if not np.isfinite(ff).all() or not np.isfinite(ss).all() or not np.isfinite(yy).all() or (ff<=0).any() or not np.isin(ss,[0,1]).all():raise ValueError('Finite acquired observations, positive force and left/right indicator required')
        order=np.lexsort((yy,ss,ff));n=len(ids);active=np.zeros(2);active[:n]=1;F=np.zeros(2);F[:n]=ff[order]/2;side=np.zeros((2,2));side[np.arange(n),ss[order].astype(int)]=1;Y=np.zeros(2);Y[:n]=yy[order]/.02
        result[i]=np.r_[active,F,side.ravel(),Y]
    return result


class BetaPosterior(nn.Module):
    def __init__(self):
        super().__init__();self.components=16
        self.body=nn.Sequential(nn.Linear(10,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU());self.head=nn.Linear(64,16*5)
        nn.init.normal_(self.head.weight,std=.02)
        with torch.no_grad():
            b=self.head.bias.reshape(16,5);b[:,0]=0;b[:,1:3]=0;b[:,3:]=np.log(2.)
    def forward(self,features):
        if features.ndim!=2 or features.shape[1]!=10 or not bool(torch.isfinite(features).all()):raise ValueError('Finite canonical acquired-observation features required')
        raw=self.head(self.body(features)).reshape(-1,16,5)
        # Mean/log-concentration parameterization learns narrow conditionals
        # without requiring raw network outputs of order1000. Fixed numerical
        # log-concentration range[-6,12], not a changed physical prior.
        mean=torch.sigmoid(raw[:,:,1:3].double());concentration=torch.exp(raw[:,:,3:5].double().clamp(-6,12))
        return dict(logits=raw[:,:,0],alpha=.001+mean*concentration,beta=.001+(1-mean)*concentration)


def joint_log_prob(unit,parameters):
    x=unit.double();a=parameters['alpha'].double();b=parameters['beta'].double();logits=parameters['logits'].double()
    if x.ndim!=2 or x.shape!=(len(a),2) or a.shape!=b.shape or a.ndim!=3 or a.shape[2]!=2 or logits.shape!=a.shape[:2] or not bool(torch.isfinite(x).all()) or bool(((x<=0)|(x>=1)).any()):raise ValueError('Strictly interior unit-prior supervised labels required; no clipping')
    log_beta=torch.lgamma(a)+torch.lgamma(b)-torch.lgamma(a+b)
    component=((a-1)*torch.log(x[:,None])+(b-1)*torch.log1p(-x[:,None])-log_beta).sum(-1)
    value=torch.logsumexp(torch.log_softmax(logits,dim=-1)+component,dim=-1)
    if not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite proper mixture density')
    return value


def numpy_parameters(parameters):
    p={k:(v.detach().cpu().double().numpy() if isinstance(v,torch.Tensor) else np.asarray(v,dtype=float)) for k,v in parameters.items()};a=p['alpha'];b=p['beta'];logits=p['logits']
    if a.ndim!=3 or a.shape[2]!=2 or b.shape!=a.shape or logits.shape!=a.shape[:2] or not all(np.isfinite(x).all() for x in p.values()) or (a<=0).any() or (b<=0).any():raise ValueError('Finite proper mixture parameters required')
    return dict(weights=np.exp(logits-logsumexp(logits,axis=1,keepdims=True)),alpha=a,beta=b)


def marginal_cdf(parameters,unit):
    p=numpy_parameters(parameters);x=np.asarray(unit,dtype=float)
    if x.shape!=(len(p['weights']),2) or not np.isfinite(x).all():raise ValueError('Finite [objects,2] marginal query required')
    return np.sum(p['weights'][:,:,None]*betainc(p['alpha'],p['beta'],np.clip(x,0,1)[:,None]),axis=1)


def marginal_quantiles(parameters,probabilities=(.05,.5,.95)):
    p=numpy_parameters(parameters);B=len(p['weights']);out=[]
    # Parameters are unchanged; analytic CDF root, never particle percentiles.
    for prob in probabilities:
        if not 0<prob<1:raise ValueError('Interior marginal probability required')
        lo=np.zeros((B,2));hi=np.ones((B,2))
        for _ in range(40):
            mid=(lo+hi)/2;cdf=np.sum(p['weights'][:,:,None]*betainc(p['alpha'],p['beta'],mid[:,None]),axis=1);below=cdf<prob;lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
        out.append((lo+hi)/2)
    return np.stack(out,axis=2)


def beta_quadrature(alpha,beta,order):
    """Normalized Gauss-Beta via Jacobi recurrence, avoiding beta_fn overflow."""
    if not np.isfinite([alpha,beta]).all() or min(alpha,beta)<=0 or type(order) is not int or not 2<=order<=64:raise ValueError('Positive Beta parameters and fixed order2..64 required')
    total=alpha+beta;n=np.arange(order,dtype=float);diagonal=np.empty(order);diagonal[0]=alpha/total
    diagonal[1:]=.5+(alpha-beta)*(total-2)/(2*(2*n[1:]+total-2)*(2*n[1:]+total))
    k=np.arange(1,order,dtype=float);square=np.empty(order-1);square[0]=alpha*beta/(total**2*(total+1))
    if order>2:
        j=k[1:];square[1:]=j*(j+alpha-1)*(j+beta-1)*(j+total-2)/((2*j+total-2)**2*(2*j+total-1)*(2*j+total-3))
    nodes,vectors=eigh_tridiagonal(diagonal,np.sqrt(square));weights=vectors[0]**2;weights/=weights.sum()
    if not np.isfinite(nodes).all() or (nodes<=0).any() or (nodes>=1).any():raise FloatingPointError('Quadrature lost interior Beta support; no clipping')
    return nodes,weights


def mixture_quadrature(parameters,index,*,order=12):
    p=numpy_parameters(parameters);theta=[];weight=[]
    for k,pi in enumerate(p['weights'][index]):
        e,we=beta_quadrature(p['alpha'][index,k,0],p['beta'][index,k,0],order)
        z,wz=beta_quadrature(p['alpha'][index,k,1],p['beta'][index,k,1],order)
        theta.append(np.stack(np.meshgrid(e,z,indexing='ij'),axis=-1).reshape(-1,2));weight.append(pi*np.outer(we,wz).ravel())
    return np.concatenate(theta),np.concatenate(weight)


def field_moments(parameters,*,order=12):
    p=numpy_parameters(parameters);X=material_quadrature();mean=[];std=[]
    for i in range(len(p['weights'])):
        unit,w=mixture_quadrature(parameters,i,order=order);q=oracle_positions(prior_physical(unit),TARGET_FORCE,X)-X;m=w@q.reshape(len(w),-1);variance=w@((q.reshape(len(w),-1)-m)**2)
        mean.append(X+m.reshape(X.shape));std.append(np.sqrt(variance).reshape(X.shape))
    return np.asarray(mean),np.asarray(std)
