"""Two-parameter static inference primitives; no data, execution or I/O.

Every generalized elastic/foundation mode is retained. This is algebraically
the supplied linear finite element model, not a learned surrogate. Physical
admission and linear/nonlinear discrepancy must be checked by the caller.
"""
import numpy as np
from scipy import linalg, sparse
from scipy.integrate import cumulative_simpson
from scipy.special import logsumexp


class ModalStaticReference:
    def __init__(self, elastic, foundation, forces, queries):
        A=elastic.toarray() if sparse.issparse(elastic) else np.asarray(elastic,float)
        B=foundation.toarray() if sparse.issparse(foundation) else np.asarray(foundation,float)
        f=np.asarray(forces,float);P=queries.toarray() if sparse.issparse(queries) else np.asarray(queries,float)
        if A.ndim!=2 or A.shape[0]!=A.shape[1] or B.shape!=A.shape or f.ndim!=2 or f.shape[0]!=len(A) or P.ndim!=2 or P.shape[1]!=len(A):
            raise ValueError('Square common stiffnesses, force[dof,R], query[Q,dof]')
        if not all(np.isfinite(x).all() for x in (A,B,f,P)) or not np.allclose(A,A.T,atol=1e-13,rtol=1e-12) or not np.allclose(B,B.T,atol=1e-13,rtol=1e-12):
            raise ValueError('Finite symmetric matrices required')
        eigenvalues,V=linalg.eigh(B,A,check_finite=True,driver='gvd')
        self.eigenvalues=eigenvalues
        self.coefficients=np.einsum('qm,mr->qrm',P@V,V.T@f).reshape(-1,len(A))
        self.output_shape=(len(P),f.shape[1]);self.modes=len(A)
        residual=B@V-(A@V)*eigenvalues
        scale=max(np.linalg.norm(B)*np.linalg.norm(V),np.finfo(float).tiny)
        self.diagnostics=dict(modes=len(A),all_modes_retained=True,eigenvalues_clipped=False,
            generalized_relative_residual=float(np.linalg.norm(residual)/scale),
            maximum_metric_orthogonality_error=float(np.max(abs(V.T@A@V-np.eye(len(A))))),
            minimum_generalized_eigenvalue=float(eigenvalues.min()))
        if self.diagnostics['generalized_relative_residual']>1e-10 or self.diagnostics['maximum_metric_orthogonality_error']>1e-8:
            raise FloatingPointError('Generalized spectral identity controls failed')

    def __call__(self,physical):
        theta=np.asarray(physical,float)
        if theta.ndim!=2 or theta.shape[1]!=2 or not np.isfinite(theta).all() or np.any(theta<=0):raise ValueError('Positive physical[E,K]')
        denominator=theta[:,0,None]+theta[:,1,None]*self.eigenvalues
        if np.any(denominator<=0):raise FloatingPointError('Nonpositive generalized stiffness denominator')
        return ((1/denominator)@self.coefficients.T).reshape(len(theta),*self.output_shape)


def point_query_matrix(forward,points):
    from experiments.campaign_surface_traction_static_v01 import query_map
    ids,bary=query_map(forward.n,points);Q=len(ids)
    rows=np.broadcast_to((3*np.arange(Q)[:,None,None]+np.arange(3)[None,None,:]),(Q,4,3)).ravel()
    cols=(3*ids[:,:,None]+np.arange(3)).ravel();data=np.broadcast_to(bary[:,:,None],(Q,4,3)).ravel()
    return sparse.coo_matrix((data,(rows,cols)),shape=(3*Q,3*len(forward.X))).tocsr()[:,forward.free]


def simpson_weights(n):
    if isinstance(n,bool) or not isinstance(n,int) or n<5 or n%2!=1:raise ValueError('Odd Simpson grid n>=5')
    weights=np.ones(n);weights[1:-1:2]=4;weights[2:-1:2]=2
    return weights/(3*(n-1))


def posterior_grid(loglik,*,n,predict=None,chunk=2048,keep=False):
    """Normalized uniform-unit-square prior; positive tensor Simpson weights.

    Marginal CDFs integrate continuous marginal densities, rather than the
    biased inclusive CDF of quadrature point masses. No deterministic-grid ESS.
    """
    w=simpson_weights(n);axis=np.linspace(0,1,n);unit=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
    likelihood=np.concatenate([np.asarray(loglik(unit[i:i+chunk]),float) for i in range(0,len(unit),chunk)])
    if likelihood.shape!=(n*n,) or np.isnan(likelihood).any() or np.isposinf(likelihood).any() or not np.isfinite(likelihood).any():raise FloatingPointError('Invalid grid likelihood')
    log_weight=likelihood+np.log(np.outer(w,w).ravel());logZ=float(logsumexp(log_weight));prob=np.exp(log_weight-logZ)
    density=np.exp(likelihood.reshape(n,n)-np.max(likelihood));marginals=np.stack([density@w,w@density]);cdfs=cumulative_simpson(marginals,x=axis,axis=1,initial=0);cdfs/=cdfs[:,-1,None]
    if np.any(np.diff(cdfs,axis=1)<-1e-12):raise FloatingPointError('Underresolved/nonmonotone marginal CDF')
    knots=np.linspace(0,1,33);cdf=np.stack([np.interp(knots,axis,row) for row in cdfs]);quantiles=np.stack([np.interp([.05,.5,.95],row,axis) for row in cdfs])
    mean=prob@unit;cov=(unit-mean).T@(prob[:,None]*(unit-mean))
    result=dict(n=n,log_evidence=logZ,unit_mean=mean,unit_covariance=cov,unit_CDF=cdf,unit_quantiles=quantiles)
    if predict is not None:
        # Positive centered two-pass moments avoid cancellation near zero SD.
        outputs=[];weights=[]
        for i in range(0,len(unit),chunk):
            y=np.asarray(predict(unit[i:i+chunk]),float).reshape(min(chunk,len(unit)-i),-1)
            if not np.isfinite(y).all():raise FloatingPointError('Nonfinite prediction')
            p=prob[i:i+len(y)];outputs.append(p@y);weights.append(p.sum())
        m=np.sum(outputs,axis=0);variance=np.zeros_like(m)
        for i in range(0,len(unit),chunk):
            y=np.asarray(predict(unit[i:i+chunk]),float).reshape(min(chunk,len(unit)-i),-1);variance+=prob[i:i+len(y)]@((y-m)**2)
        result.update(predictive_mean=m,predictive_SD=np.sqrt(variance))
    if keep:result.update(unit_grid=unit,quadrature_probability=prob,log_likelihood=likelihood)
    return result


def compare_grids(coarse,fine):
    errors=dict(log_evidence_difference=abs(coarse['log_evidence']-fine['log_evidence']),
        maximum_CDF_difference=float(np.max(abs(coarse['unit_CDF']-fine['unit_CDF']))),
        maximum_unit_quantile_difference=float(np.max(abs(coarse['unit_quantiles']-fine['unit_quantiles']))))
    passed=errors['log_evidence_difference']<=.01 and errors['maximum_CDF_difference']<=.005 and errors['maximum_unit_quantile_difference']<=.002
    if 'predictive_mean' in fine:
        tolerance=np.maximum(2e-6,.01*fine['predictive_SD'])
        for key in ('predictive_mean','predictive_SD'):
            errors['maximum_'+key+'_difference_m']=float(np.max(abs(coarse[key]-fine[key])))
            passed=passed and bool(np.all(abs(coarse[key]-fine[key])<=tolerance))
    return dict(**errors,passed=bool(passed),scope='Successive deterministic refinements; independent sampler and model fidelity checks remain necessary')


def posterior_gauss(loglik,*,n,predict=None,chunk=2048):
    """Independent positive Gauss-Legendre tensor integration, without CDFs.

    Different nodes/weights cross-check evidence and moments of Simpson
    integration. Successive levels and model fidelity checks remain necessary.
    """
    if isinstance(n,bool) or not isinstance(n,int) or n<4:raise ValueError('Integer Gauss order>=4')
    axis,w=np.polynomial.legendre.leggauss(n);axis=(axis+1)/2;w=w/2
    unit=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
    ll=np.concatenate([np.asarray(loglik(unit[i:i+chunk]),float) for i in range(0,len(unit),chunk)])
    if ll.shape!=(n*n,) or np.isnan(ll).any() or np.isposinf(ll).any() or not np.isfinite(ll).any():raise FloatingPointError('Invalid Gauss likelihood')
    lw=ll+np.log(np.outer(w,w).ravel());logZ=float(logsumexp(lw));p=np.exp(lw-logZ);mean=p@unit
    result=dict(n=n,rule='Gauss-Legendre',log_evidence=logZ,unit_mean=mean,unit_covariance=(unit-mean).T@(p[:,None]*(unit-mean)))
    if predict is not None:
        m=None
        for i in range(0,len(unit),chunk):
            y=np.asarray(predict(unit[i:i+chunk]),float).reshape(min(chunk,len(unit)-i),-1)
            if not np.isfinite(y).all():raise FloatingPointError('Nonfinite prediction')
            weighted=p[i:i+len(y)]@y;m=weighted if m is None else m+weighted
        variance=np.zeros_like(m)
        for i in range(0,len(unit),chunk):
            y=np.asarray(predict(unit[i:i+chunk]),float).reshape(min(chunk,len(unit)-i),-1);variance+=p[i:i+len(y)]@((y-m)**2)
        result.update(predictive_mean=m,predictive_SD=np.sqrt(variance))
    return result


def gauss_partial_CDF(loglik,quantiles,*,n,log_evidence,chunk=2048):
    """Independent positive partial-rectangle integrals at supplied thresholds.

    quantiles[2,Q] are fixed unit-coordinate thresholds; evidence comes from
    the full Gauss integral at the same order. No quadrature-point atomic CDF.
    """
    q=np.asarray(quantiles,float)
    if q.ndim!=2 or q.shape[0]!=2 or not np.isfinite(q).all() or np.any((q<0)|(q>1)) or not np.isfinite(log_evidence):raise ValueError('Finite unit thresholds[2,Q] and log evidence')
    if isinstance(n,bool) or not isinstance(n,int) or n<4:raise ValueError('Integer Gauss order>=4')
    axis,w=np.polynomial.legendre.leggauss(n);axis=(axis+1)/2;w=w/2
    base=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2);lw=np.log(np.outer(w,w).ravel());answer=np.zeros_like(q)
    for dimension in range(2):
        for index,bound in enumerate(q[dimension]):
            if bound==0:continue
            unit=base.copy();unit[:,dimension]*=bound
            ll=np.concatenate([np.asarray(loglik(unit[i:i+chunk]),float) for i in range(0,len(unit),chunk)])
            if ll.shape!=(n*n,) or np.isnan(ll).any() or np.isposinf(ll).any():raise FloatingPointError('Invalid partial likelihood')
            answer[dimension,index]=np.exp(logsumexp(ll+lw)+np.log(bound)-log_evidence)
    if not np.isfinite(answer).all() or np.any(answer<0) or np.any(answer>1+1e-8):raise FloatingPointError('Underresolved partial CDF or evidence')
    return answer
