"""Prospective clock, convergence and full-mass momentum controls; NO launcher.

New source only. This module does not admit or start a physical trajectory.
"""
import numpy as np


def schedule(steps):
    if type(steps) is not int or steps not in (32,64,128):
        raise ValueError('Frozen 32/64/128 steps on [0,1] required')
    return dict(h_s=1/steps, midpoint_times_s=(np.arange(steps)+.5)/steps,
                clock_steps=np.arange(9,dtype=int)*(steps//8), clocks_s=np.arange(9)/8)


def pair_difference(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[0]!=9 or a.shape[2]!=3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite same-shape nine-clock vector fields required')
    return float(np.linalg.norm(a-b,axis=2).max())


def convergence(q32,q64,q128,v32,v64,v128,qspace32,qspace40,vspace32,vspace40,hspace32,hspace40):
    """Conservative conditional p>=1 error indicators, for BOTH q and v.

    Temporal three-level observed contraction must support p>=1. Spatial has
    two levels only and is expressly conditional on p>=1, never observed order.
    Indicators are summed, retaining the old 20um / 100um/s combined gates.
    """
    if not np.isfinite([hspace32,hspace40]).all() or not hspace32>hspace40>0:
        raise ValueError('Actual positive decreasing mesh hmax required')
    rows={}
    for name,a,b,c,sa,sb,gate in [('q',q32,q64,q128,qspace32,qspace40,20e-6),('v',v32,v64,v128,vspace32,vspace40,100e-6)]:
        d0,d1=pair_difference(a,b),pair_difference(b,c)
        contracted=(d0==d1==0) or d0>=2*d1
        order=None if d0==0 or d1==0 else float(np.log2(d0/d1))
        spatial=pair_difference(sa,sb)/(hspace32/hspace40-1)
        rows[name]=dict(temporal_coarse_delta=d0,temporal_fine_delta=d1,observed_temporal_order=order,
                       temporal_p_at_least_one=bool(contracted),temporal_indicator=d1,
                       spatial_p_at_least_one_assumed=True,spatial_indicator=spatial,
                       combined_indicator=d1+spatial,threshold=gate,passed=bool(contracted and d1+spatial<=gate))
    return dict(passed=all(r['passed'] for r in rows.values()),quantities=rows,
                scope='Nine fixed clocks/declared material queries; conditional spatial p>=1, not full-field or continuous-time certification')


def full_mass_momentum_ledger(rest,u0,v0,u1,v1,scalar_mass,force_mid,reaction_mid,h):
    """Discrete linear/angular momentum about the original world origin.

    Reaction includes M_cf*a and elastic fixed components, with sign chosen so
    M*a = Fexternal + reaction - gradU. No row-lumping. Symmetry of full scalar
    M makes the cross-velocity contribution cancel for midpoint kinematics.
    Internal torque (if any) remains visible in the reported residual.
    """
    x,u,v,y,w,F,R=[np.asarray(a,dtype=float) for a in (rest,u0,v0,u1,v1,force_mid,reaction_mid)]
    # Keep sparse full mass sparse: a dense fine-mesh matrix would be infeasible.
    from scipy import sparse
    M=scalar_mass.tocsr() if sparse.issparse(scalar_mass) else np.asarray(scalar_mass,dtype=float)
    mass_finite=np.isfinite(M.data).all() if sparse.issparse(M) else np.isfinite(M).all()
    symmetric=(M-M.T).nnz==0 if sparse.issparse(M) else np.array_equal(M,M.T)
    if not h>0 or not np.isfinite(h) or any(a.shape!=x.shape or not np.isfinite(a).all() for a in (x,u,v,y,w,F,R)) or x.ndim!=2 or x.shape[1]!=3 or M.shape!=(len(x),len(x)) or not mass_finite or not symmetric:
        raise ValueError('Finite full symmetric scalar mass and matching nodal vectors required')
    q0,q1=x+u,x+y;P0,P1=(M@v).sum(0),(M@w).sum(0)
    L0,L1=np.cross(q0,M@v).sum(0),np.cross(q1,M@w).sum(0)
    impulse=h*(F+R).sum(0);angular_impulse=h*np.cross((q0+q1)/2,F+R).sum(0)
    return dict(linear_residual_N_s=float(np.linalg.norm(P1-P0-impulse)),
                angular_residual_N_m_s=float(np.linalg.norm(L1-L0-angular_impulse)),
                kinematic_residual_m=float(np.linalg.norm((q1-q0)-h*(v+w)/2,axis=1).max()))
