"""Prospective exact-rest-LU preconditioned Newton directions; no executions.

The supplied rest operator must be SPD (a caller/model assumption, not a claim
that LU certifies SPD). The current tangent is never shifted, truncated or
symmetrized. This helper only verifies its linear direction and descent; final
nonlinear equilibrium, stability, strain and Jacobian gates remain external.
"""
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,cg,splu

DESIGN=dict(schema='exact-rest-pcg-v01',maximum_CG_iterations=200,CG_rtol=1e-12,CG_atol_N=1e-15,
    acceptance_absolute_residual_N=1e-13,acceptance_relative_residual=1e-11,
    fallback='One unmodified current-tangent sparse LU if CG residual, convergence or descent fails; reject if fallback fails',
    artificial_stiffness=False,mode_truncation=False,current_tangent_modified=False,
    scope='Linear Newton-direction performance prototype only; not independent SPD/nonlinear/physics qualification')


class DirectionFailure(FloatingPointError):
    def __init__(self,message,diagnostics):
        super().__init__(message);self.diagnostics=diagnostics


def _symmetric_matrix(matrix,*,shape=None):
    A=sparse.csc_matrix(matrix,dtype=np.float64,copy=True);A.sum_duplicates()
    if len(A.shape)!=2 or A.shape[0]!=A.shape[1]or(shape is not None and A.shape!=shape)or not np.isfinite(A.data).all():
        raise ValueError('Common finite square matrix required')
    scale=max(float(sparse.linalg.norm(A)),np.finfo(float).tiny)
    if not np.isfinite(scale)or sparse.linalg.norm(A-A.T)>1e-12*scale:raise ValueError('Symmetric matrix required; no silent symmetrization')
    return A


class ExactRestPCG:
    def __init__(self,rest,*,maximum_iterations=200):
        if type(maximum_iterations)is not int or not 1<=maximum_iterations<=200:raise ValueError('Bounded1..200 CG iterations')
        self.rest=_symmetric_matrix(rest);self.maximum_iterations=maximum_iterations
        start=time.perf_counter();self.rest_LU=splu(self.rest);self.rest_factorization_s=time.perf_counter()-start
        self.preconditioner=LinearOperator(self.rest.shape,matvec=self.rest_LU.solve,
            rmatvec=lambda x:self.rest_LU.solve(x,trans='T'),dtype=np.float64)
        self.rest_factorizations=1;self.current_fallback_factorizations=0

    def rest_solve(self,rhs):
        """Reuse exactly the same rest LU for a linear response, without refactor."""
        f=np.asarray(rhs,float)
        if f.ndim not in(1,2)or f.shape[0]!=self.rest.shape[0]or not np.isfinite(f).all():raise ValueError('Finite rest RHS')
        rhs_norm=float(np.linalg.norm(f))
        if not np.isfinite(rhs_norm):raise ValueError('Finite RHS norm required')
        x=self.rest_LU.solve(f);error=float(np.linalg.norm(self.rest@x-f));tol=max(1e-13,1e-11*rhs_norm)
        if not np.isfinite(x).all()or not np.isfinite(error)or error>tol:raise DirectionFailure('Rest solve explicit residual failed',dict(residual_N=error,tolerance_N=tol))
        return x

    def direction(self,tangent,rhs):
        """Solve H d=rhs (Newton rhs=-gradient); require positive rhs·d."""
        H=_symmetric_matrix(tangent,shape=self.rest.shape);b=np.asarray(rhs,float)
        if b.shape!=(H.shape[0],)or not np.isfinite(b).all():raise ValueError('Finite Newton RHS vector')
        norm=float(np.linalg.norm(b))
        if not np.isfinite(norm):raise ValueError('Finite Newton RHS norm required')
        tol=max(1e-13,1e-11*norm)
        record=dict(rest_factorizations=self.rest_factorizations,rest_factorization_s=self.rest_factorization_s,
            maximum_CG_iterations=self.maximum_iterations,CG_rtol=1e-12,CG_atol_N=1e-15,
            RHS_norm_N=norm,explicit_residual_tolerance_N=tol,CG_iterations=0,CG_info=None,
            fallback=False,current_tangent_modified=False,SPD_certified_by_this_helper=False)
        if np.all(b==0):
            record.update(method='zero_RHS',accepted=True,explicit_residual_N=0.,RHS_dot_direction=0.)
            return np.zeros_like(b),record
        def acceptable(x):
            if not np.isfinite(x).all():return False,float('inf'),float('nan')
            residual=float(np.linalg.norm(H@x-b));descent=float(b@x)
            return bool(np.isfinite(residual)and np.isfinite(descent)and residual<=tol and descent>0),residual,descent
        def scalar(x):return float(x)if np.isfinite(x)else None
        def callback(x):record['CG_iterations']+=1
        start=time.perf_counter();reason=None
        try:
            d,info=cg(H,b,M=self.preconditioner,rtol=1e-12,atol=1e-15,maxiter=self.maximum_iterations,callback=callback)
            record['CG_info']=int(info);good,residual,descent=acceptable(d)
            record.update(CG_explicit_residual_N=scalar(residual),CG_RHS_dot_direction=scalar(descent))
            if info==0 and good:
                record.update(method='exact_rest_preconditioned_CG',accepted=True,explicit_residual_N=residual,
                    RHS_dot_direction=descent,linear_solve_s=time.perf_counter()-start)
                return d,record
            reason='CG convergence/residual/descent check failed'
        except(FloatingPointError,ValueError,RuntimeError)as error:
            reason=type(error).__name__+': '+str(error)
        record.update(fallback=True,fallback_reason=reason)
        try:
            LU=splu(H);self.current_fallback_factorizations+=1;d=LU.solve(b)
            good,residual,descent=acceptable(d)
            record.update(method='unmodified_current_LU_fallback',accepted=good,explicit_residual_N=scalar(residual),
                RHS_dot_direction=scalar(descent),linear_solve_s=time.perf_counter()-start,
                current_fallback_factorizations=self.current_fallback_factorizations)
        except(RuntimeError,ValueError)as error:
            record.update(method='unmodified_current_LU_fallback',accepted=False,fallback_error=type(error).__name__+': '+str(error))
            raise DirectionFailure('Unmodified direct-LU fallback failed',record)from error
        if not good:raise DirectionFailure('No verified descent direction; no stiffness modification permitted',record)
        return d,record
