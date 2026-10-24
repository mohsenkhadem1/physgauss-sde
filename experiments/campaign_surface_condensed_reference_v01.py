"""Exact static elimination of interior DOFs before generalized diagonalization.

Foundation-free interior equations contain only E*Kunit. Eliminating them is
an algebraic Schur complement, without a displacement basis approximation,
eigenvalue threshold, discarded mode, or change to the finite element model.
All boundary generalized modes are retained. No automatic experiments or I/O.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from experiments.campaign_surface_inverse_reference_v01 import ModalStaticReference


class CondensedStaticReference(ModalStaticReference):
    def __init__(self,elastic,foundation,forces,queries):
        A=sparse.csc_matrix(elastic,dtype=float);B=sparse.csr_matrix(foundation,dtype=float).copy();B.eliminate_zeros()
        f=np.asarray(forces,float);P=sparse.csr_matrix(queries,dtype=float).copy();P.eliminate_zeros();N=A.shape[0]
        if A.shape!=(N,N) or B.shape!=A.shape or f.ndim!=2 or f.shape[0]!=N or P.shape[1]!=N or not all(np.isfinite(x).all()for x in (A.data,B.data,f,P.data)):
            raise ValueError('Finite common static matrices, force[dof,R], query[Q,dof]')
        boundary=np.flatnonzero(np.diff(B.indptr)>0);interior=np.setdiff1d(np.arange(N),boundary)
        if not len(boundary) or not len(interior):raise ValueError('Nonempty foundation boundary and foundation-free interior required')
        if B[interior].nnz or B[:,interior].nnz:raise ValueError('Foundation must vanish exactly in eliminated rows/columns')
        Aii=A[interior][:,interior];Aib=A[interior][:,boundary];Abi=A[boundary][:,interior];Abb=A[boundary][:,boundary]
        lu=splu(Aii);M=lu.solve(Aib.toarray());particular=lu.solve(f[interior])
        residual=Aii@M-Aib.toarray();relative=float(np.linalg.norm(residual)/max(np.linalg.norm(Aib.data),np.finfo(float).tiny))
        if relative>1e-11:raise FloatingPointError('Interior elimination residual failed')
        S=Abb.toarray()-Abi@M;effective_force=f[boundary]-Abi@particular
        effective_query=P[:,boundary].toarray()-P[:,interior]@M
        direct=np.asarray(P[:,interior]@particular)
        asymmetry=float(np.max(abs(S-S.T)))
        # Do not symmetrize, shift or regularize. The generic solver checks
        # analytic symmetry to roundoff and retains every generalized mode.
        super().__init__(S,B[boundary][:,boundary],effective_force,effective_query)
        extra=bool(np.any(direct!=0))
        if extra:
            # Exact particular response P_i*Kii^-1*f_i/E, represented by a
            # separate zero-eigenvalue term, not an additional physical mode.
            self.eigenvalues=np.concatenate([self.eigenvalues,[0.]])
            self.coefficients=np.column_stack([self.coefficients,direct.ravel()])
        self.full_free_DOF=N;self.boundary=boundary;self.interior=interior
        self.diagnostics.update(full_free_DOF=N,condensed_boundary_DOF=len(boundary),eliminated_interior_DOF=len(interior),
            all_boundary_modes_retained=True,interior_basis_truncated=False,foundation_threshold_applied=False,
            interior_elimination_relative_residual=relative,Schur_maximum_asymmetry=asymmetry,
            exact_particular_1_over_E_term=extra,interior_force_exact_zero=bool(np.all(f[interior]==0)),
            interior_query_exact_zero=bool(P[:,interior].nnz==0))
