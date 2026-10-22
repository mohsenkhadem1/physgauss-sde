"""Strict FP64 PCG for the unchanged FULL consistent FE mass matrix.

A diagonal appears only in the preconditioner. It never replaces the physical
matrix, kinetic energy, body load or fixture reaction. No solve at import.
"""
import numpy as np
from scipy import sparse
import torch


class MassSolveFailure(RuntimeError):
    pass


class ConsistentMassPCG:
    """Three coordinate RHS are one block-diagonal system with Frobenius dots.

    Every successful return satisfies the independently recomputed absolute
    original residual. Recurrence convergence alone is never accepted. A false
    recurrence convergence restarts from that original residual, without
    changing the matrix or tolerance. All calls start from zero; no hidden
    history/warm start or physical mass approximation is retained.
    """
    def __init__(self, matrix, *, device='cpu', absolute_tolerance_N=1e-12,
                 max_iterations=256, true_residual_every=8, relative_tolerance=1e-11):
        if not sparse.isspmatrix_csr(matrix):
            raise ValueError('Explicit original SciPy CSR consistent mass required')
        a=matrix.copy()
        if (a.shape[0]!=a.shape[1] or not a.shape[0] or a.dtype!=np.float64 or
            not np.isfinite(a.data).all() or np.any(a.diagonal()<=0)):
            raise ValueError('Finite FP64 square positive-diagonal mass required')
        if (not np.isfinite(absolute_tolerance_N) or absolute_tolerance_N<=0 or
            type(max_iterations) is not int or max_iterations<1 or
            type(true_residual_every) is not int or true_residual_every<1 or
            not np.isfinite(relative_tolerance) or not 0<relative_tolerance<1):
            raise ValueError('Prospective positive strict tolerance/iteration limits required')
        delta=a-a.T
        if len(delta.data) and np.max(abs(delta.data))>64*np.finfo(float).eps*np.max(abs(a.data)):
            raise ValueError('Symmetric physical mass required; no symmetrization')
        self.device=torch.device(device);self.size=a.shape[0]
        self.absolute_tolerance_N=float(absolute_tolerance_N)
        self.max_iterations=max_iterations;self.true_residual_every=true_residual_every;self.relative_tolerance=float(relative_tolerance)
        self.matrix=torch.sparse_csr_tensor(torch.tensor(a.indptr.astype(np.int64),device=self.device),
            torch.tensor(a.indices.astype(np.int64),device=self.device),
            torch.tensor(a.data,device=self.device),size=a.shape,dtype=torch.float64,device=self.device,check_invariants=True)
        self.device=self.matrix.device
        self.preconditioner_diagonal=torch.tensor(a.diagonal(),dtype=torch.float64,device=self.device)[:,None]
        self.last_record=None

    def action(self,x):
        return torch.sparse.mm(self.matrix,x)

    @staticmethod
    def _norm(x):return float(torch.linalg.vector_norm(x))

    def solve_tensor(self,rhs):
        if (not isinstance(rhs,torch.Tensor) or rhs.dtype!=torch.float64 or
            rhs.device!=self.device or rhs.shape!=(self.size,3) or
            not bool(torch.isfinite(rhs).all())):
            raise ValueError('Matching finite FP64 [free_nodes,3] RHS required')
        b=rhs.detach().clone();x=torch.zeros_like(b);bnorm=self._norm(b)
        tolerance=min(self.absolute_tolerance_N,self.relative_tolerance*bnorm)
        record=dict(completed=False,passed=False,iterations=0,restarts=0,
            rhs_norm_N=bnorm,absolute_tolerance_N=self.absolute_tolerance_N,relative_tolerance=self.relative_tolerance,effective_tolerance_N=tolerance,
            residual_checks=[],physical_matrix='Unchanged full consistent CSR',
            preconditioner='Positive original diagonal ONLY, never physical row lump')
        self.last_record=record
        def checked(iteration):
            true=b-self.action(x);norm=self._norm(true)
            if not np.isfinite(norm):raise MassSolveFailure('Nonfinite original physical mass residual')
            record['residual_checks'].append(dict(iteration=iteration,true_residual_N=norm))
            record['true_residual_N']=norm
            return true,norm
        if torch.count_nonzero(b)==0:
            record.update(completed=True,passed=True,true_residual_N=0.,zero_RHS=True)
            return x
        r=b.clone();z=r/self.preconditioner_diagonal;p=z.clone();rz=float(torch.sum(r*z))
        try:
            for iteration in range(1,self.max_iterations+1):
                Ap=self.action(p);pAp=float(torch.sum(p*Ap))
                record['iterations']=iteration
                if not np.isfinite([pAp,rz]).all() or pAp<=0 or rz<=0:
                    raise MassSolveFailure('Nonpositive/nonfinite original mass PCG curvature; no repair')
                alpha=rz/pAp;x=x+alpha*p;r=r-alpha*Ap
                recurrence_norm=self._norm(r)
                if not np.isfinite(recurrence_norm):raise MassSolveFailure('Nonfinite mass PCG recurrence')
                if iteration%self.true_residual_every==0 or recurrence_norm<=tolerance or iteration==self.max_iterations:
                    true,norm=checked(iteration)
                    if norm<=tolerance:
                        record.update(completed=True,passed=True,recurrence_residual_N=recurrence_norm)
                        return x
                    if recurrence_norm<=tolerance:
                        # Recurrence roundoff may not certify the physical equation.
                        # Restart from the independently recomputed residual only.
                        r=true;z=r/self.preconditioner_diagonal;p=z.clone();rz=float(torch.sum(r*z));record['restarts']+=1
                        continue
                z=r/self.preconditioner_diagonal;newrz=float(torch.sum(r*z))
                if not np.isfinite(newrz) or newrz<=0:raise MassSolveFailure('Nonpositive mass PCG preconditioned norm')
                p=z+(newrz/rz)*p;rz=newrz
            checked(self.max_iterations)
            raise MassSolveFailure('Original mass residual exceeds frozen tolerance at iteration cap')
        except BaseException as error:
            record['error']=repr(error)
            # Keep a failed solve record; never return an uncertified acceleration.
            raise

    def solve(self,rhs):
        b=np.asarray(rhs)
        if b.dtype!=np.float64 or b.shape!=(self.size,3) or not np.isfinite(b).all():
            raise ValueError('Finite original FP64 NumPy [free_nodes,3] RHS required')
        return self.solve_tensor(torch.tensor(b,device=self.device)).detach().cpu().numpy()
