"""Prospective local-Gaussian design screen, never posterior inference.

Uniform-unit prior covariance1/12 is represented by a Gaussian precision12I.
Fisher matrices use known independent20um channels. Rankings are only local
linear-response approximations at declared latent points, not policies.
"""
import numpy as np
from experiments.campaign_surface_p2_inference_v01 import physical_parameters
from experiments.campaign_nonlinear_residual_atlas_v01 import barycentric_basis

SIGMA=20e-6
TIE_RELATIVE=1e-6
TIE_ABSOLUTE_M2=1e-20


def basis_and_derivative(query,axis):
    x=np.asarray(axis,float);B=barycentric_basis(np.asarray(query,float),x)
    w=(-1.)**np.arange(len(x));w[[0,-1]]*=.5
    difference=x[:,None]-x[None,:];mask=~np.eye(len(x),dtype=bool)
    D=np.zeros((len(x),len(x)));ratio=w[None,:]/w[:,None]
    D[mask]=ratio[mask]/difference[mask];D[np.diag_indices(len(x))]=-D.sum(axis=1)
    return B,B@D


def atlas_jacobian(atlas,unit,labels):
    u=np.asarray(unit,float);indices=[atlas.labels.index(label)for label in labels]
    if u.ndim!=2 or u.shape[1]!=2:raise ValueError('Unit[batch,2]')
    a,da=basis_and_derivative(u[:,0],atlas.axis);b,db=basis_and_derivative(u[:,1],atlas.axis);table=atlas.delta[:,:,indices,:]
    return np.stack((np.einsum('bi,bj,ijlc->blc',da,b,table,optimize=True),np.einsum('bi,bj,ijlc->blc',a,db,table,optimize=True)),axis=-1)


def linear_jacobian(model,unit,labels):
    theta=physical_parameters(unit);E,K=theta.T;lam=np.asarray(model.modal.eigenvalues)
    indices=[model.labels.index(label)for label in labels];C=model.coefficients[:,indices,:].transpose(1,0,2).reshape(len(labels)*model.channels,-1)
    denominator=E[:,None]+K[:,None]*lam
    if np.any(denominator<=0):raise FloatingPointError('Original positive modal denominator')
    square=1/denominator**2
    a=-(np.log(4.)*E[:,None]*square)@C.T;b=-(np.log(6.)*K[:,None]*lam*square)@C.T
    result=np.stack((a,b),axis=-1).reshape(len(theta),len(labels),model.channels,2)
    if not np.isfinite(result).all():raise FloatingPointError('Finite exact all-mode derivatives')
    return result


def total_jacobian(model,atlas,unit,labels):return linear_jacobian(model,unit,labels)+atlas_jacobian(atlas,unit,labels)


def screen(jprefix,jcandidates,jtargets,*,sigma=SIGMA):
    P=np.asarray(jprefix,float);A=np.asarray(jcandidates,float);T=np.asarray(jtargets,float)
    if P.ndim!=3 or P.shape[-1]!=2 or A.ndim!=4 or A.shape[0]!=len(P)or A.shape[-1]!=2 or T.ndim!=3 or T.shape[0]!=len(P)or T.shape[-1]!=2 or not len(T[0])or sigma!=SIGMA or not all(np.isfinite(x).all()for x in(P,A,T)):raise ValueError('Finite row/label/channel two-parameter Jacobians, known20um noise')
    precision=12*np.eye(2)[None]+np.einsum('bni,bnj->bij',P,P)/sigma**2
    C=np.linalg.inv(precision);risk=np.einsum('bni,bij,bnj->b',T,C,T)/T.shape[1]
    postprecision=precision[:,None]+np.einsum('bani,banj->baij',A,A)/sigma**2
    Cpost=np.linalg.inv(postprecision);postrisk=np.einsum('bni,baij,bnj->ba',T,Cpost,T)/T.shape[1]
    reduction=risk[:,None]-postrisk
    if np.any(risk<0)or np.any(postrisk<0)or not np.isfinite(reduction).all()or np.any(reduction < -(1e-12*risk[:,None]+1e-30)):raise FloatingPointError('Unresolved positive Fisher covariance/risk')
    tolerance=TIE_RELATIVE*risk+TIE_ABSOLUTE_M2
    winners=reduction>=reduction.max(axis=1)[:,None]-tolerance[:,None]
    ordered=np.sort(reduction,axis=1);margin=ordered[:,-1]-ordered[:,-2]
    return dict(prefix_covariance=C,prefix_risk_m2=risk,post_risk_m2=postrisk,reduction_m2=reduction,winner_mask=winners,top_margin_m2=margin,tie_tolerance_m2=tolerance)
