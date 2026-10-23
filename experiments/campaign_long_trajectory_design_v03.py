"""NEW guard-only repair: assembly-roundoff skew, original full M unchanged.

The physical impulse gates are unchanged. No mass symmetrization, clipping,
row-lumping, projection or force modification occurs here.
"""
import numpy as np
from scipy import sparse
from experiments.campaign_long_trajectory_design import schedule,pair_difference,convergence
# Verified CPU topology upper bound for both pinned long32/40 meshes. Local
# analytic mass entries have a uniform sign for each global nodal pair, so
# sum(abs(local terms)) = abs(exact entry); there is no cancellation factor.
MAX_ASSEMBLY_TERMS_PER_ENTRY=64


def mass_symmetry_guard(M):
    M=M.tocsr() if sparse.issparse(M) else np.asarray(M,dtype=float)
    if M.ndim!=2 or M.shape[0]!=M.shape[1]:raise ValueError('Square full mass required')
    data=M.data if sparse.issparse(M) else M
    if not np.isfinite(data).all():raise ValueError('Finite original full mass required')
    scale=float(abs(data).max()) if data.size else 0.
    D=M-M.T;values=D.data if sparse.issparse(D) else D
    skew=float(abs(values).max()) if values.size else 0.
    eps=np.finfo(np.float64).eps;gamma=MAX_ASSEMBLY_TERMS_PER_ENTRY*eps/(1-MAX_ASSEMBLY_TERMS_PER_ENTRY*eps)
    # Each of two accumulation orders differs from the exact same-sign sum by
    # gamma_n*|exact|, and |exact|<=|assembled|/(1-gamma_n).
    bound=2*gamma*scale/(1-gamma)
    row=dict(maximum_abs_mass_entry_kg=scale,maximum_abs_skew_kg=skew,relative_skew=skew/scale if scale else 0.,assembly_terms_bound=MAX_ASSEMBLY_TERMS_PER_ENTRY,FP64_pair_accumulation_skew_bound_kg=bound,original_matrix_unmodified=True)
    if skew>bound:raise ValueError('Full mass skew exceeds justified FP64 assembly-roundoff envelope: '+str(row))
    return M,row


def full_mass_momentum_ledger(rest,u0,v0,u1,v1,scalar_mass,force_mid,reaction_mid,h):
    x,u,v,y,w,F,R=[np.asarray(a,dtype=float) for a in (rest,u0,v0,u1,v1,force_mid,reaction_mid)]
    M,symmetry=mass_symmetry_guard(scalar_mass)
    if not h>0 or not np.isfinite(h) or any(a.shape!=x.shape or not np.isfinite(a).all() for a in (x,u,v,y,w,F,R)) or x.ndim!=2 or x.shape[1]!=3 or M.shape!=(len(x),len(x)):
        raise ValueError('Finite full mass and matching nodal vectors required')
    q0,q1=x+u,x+y;P0,P1=(M@v).sum(0),(M@w).sum(0)
    L0,L1=np.cross(q0,M@v).sum(0),np.cross(q1,M@w).sum(0)
    impulse=h*(F+R).sum(0);angular_impulse=h*np.cross((q0+q1)/2,F+R).sum(0)
    vmid=(v+w)/2;cross=h*np.cross(vmid,M@vmid).sum(0)
    norms=np.linalg.norm(vmid,axis=1);skew_bound=.5*h*float(norms@(abs(M-M.T)@norms))
    return dict(linear_residual_N_s=float(np.linalg.norm(P1-P0-impulse)),
                angular_residual_N_m_s=float(np.linalg.norm(L1-L0-angular_impulse)),
                kinematic_residual_m=float(np.linalg.norm((q1-q0)-h*(v+w)/2,axis=1).max()),
                mass_symmetry=symmetry,
                evaluated_cross_velocity_angular_impulse_N_m_s=float(np.linalg.norm(cross)),
                exact_arithmetic_skew_cross_velocity_impulse_bound_N_m_s=skew_bound,
                cross_velocity_term_retained_in_original_physical_impulse_residual=True)
