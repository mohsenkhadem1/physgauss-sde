"""Point-anchored geometric retraction of a linearized displacement field.

Uniform-node leastsquares fits omega from u_i≈omega×(rest_i−rest_pin).
Only the fitted firstorder rotation is replaced by exp([omega]×). The remaining
linear elastic residual is retained unchanged. No dynamics, parameter fit,
forces, masses, likelihoods or observations are used. This is NOT an exact
finite-strain/rigid-handle solver or a corrected time integrator.
"""
from __future__ import annotations
import numpy as np


def _skew(omega):
    w=np.asarray(omega);result=np.zeros((*w.shape[:-1],3,3),dtype=np.float64)
    result[...,0,1]=-w[...,2];result[...,0,2]=w[...,1]
    result[...,1,0]=w[...,2];result[...,1,2]=-w[...,0]
    result[...,2,0]=-w[...,1];result[...,2,1]=w[...,0]
    return result


def retract_point_rotation(rest_points_m,displacement_m,fixture_node):
    """Return retracted displacement and diagnostic arrays, preserving the pin.

    rest:[N,3], displacement:[...,N,3], float64; the suppliedpointpin must
    already have EXACTzero displacement. No drift projection or regularizer.
    The normalmatrix uses fixed uniformweights over ALL reference nodes.
    """
    rest=np.asarray(rest_points_m);u=np.asarray(displacement_m)
    if (rest.ndim!=2 or rest.shape[1]!=3 or len(rest)<3 or rest.dtype!=np.dtype('float64')
        or u.ndim<2 or u.shape[-2:]!=rest.shape or u.dtype!=np.dtype('float64')
        or not np.isfinite(rest).all() or not np.isfinite(u).all()
        or type(fixture_node) is not int or not 0<=fixture_node<len(rest)):
        raise ValueError('Finitefloat64 rest[N,3]/displacement[...,N,3] andvalidpointpin required')
    if np.any(u[...,fixture_node,:]!=0):raise ValueError('Pointpin must already be exact; no silentprojection')
    r=rest-rest[fixture_node];normal=np.eye(3)*np.sum(r*r)-r.T@r
    condition=float(np.linalg.cond(normal))
    if not np.isfinite(condition) or condition>1e12:raise ValueError('Nondegenerate anchoredrotation fit required; no regularization')
    rhs=np.cross(r,u).sum(axis=-2);omega=np.linalg.solve(normal,rhs.reshape(-1,3).T).T.reshape(rhs.shape)
    k=_skew(omega);k2=k@k;theta2=np.sum(omega*omega,axis=-1);theta=np.sqrt(theta2)
    # Stableexp−I−K; do NOT subtract nearlyidentity matrices at tinyangles.
    a_minus1=np.sinc(theta/np.pi)-1
    small=theta2<1e-4
    series=-theta2/6+theta2**2/120-theta2**3/5040+theta2**4/362880
    a_minus1=np.where(small,series,a_minus1)
    b=.5*np.sinc(theta/(2*np.pi))**2
    correction_matrix=a_minus1[...,None,None]*k+b[...,None,None]*k2
    correction=np.einsum('...ij,nj->...ni',correction_matrix,r)
    retracted=u+correction;linear_rotation=np.cross(omega[...,None,:],r)
    return dict(displacement_m=retracted,correction_m=correction,fitted_infinitesimal_rotation_rad=omega,
        fitted_rotation_matrix=np.eye(3)+k+correction_matrix,
        linear_residual_coordinate_RMS_m=np.sqrt(np.mean((u-linear_rotation)**2,axis=(-2,-1))),
        rotation_fit_normal_condition_number=condition,
        role='Geometry-only exp-retraction of fittedfirstorderrotation, unchangedlinearresidual; no improveddynamics/constraint guarantee')
