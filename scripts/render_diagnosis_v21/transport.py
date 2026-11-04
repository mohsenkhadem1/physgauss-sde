"""Matched anchor transport for diagnostic interventions, not a predictor."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def valid_anchors(raw,frame):
    valid=np.asarray(raw['object_visibilities'][0],bool)&np.asarray(raw['object_visibilities'][frame],bool)
    if frame>0:valid&=np.asarray(raw['object_motions_valid'][frame-1],bool)
    valid&=np.isfinite(raw['object_points'][0]).all(1)&np.isfinite(raw['object_points'][frame]).all(1)
    return valid

def setup(initial,xyz):
    if len(initial)<17:raise ValueError('Insufficient valid motion anchors')
    tree=cKDTree(initial);neighbors=tree.query(initial,k=17)[1][:,1:]
    distances,ids=tree.query(xyz,k=16)
    weights=1/np.maximum(distances,1e-4)**2;weights/=weights.sum(1,keepdims=True)
    return neighbors,ids,weights,distances[:,0]

def transforms(initial,current,neighbors,affine=False):
    a=initial[neighbors]-initial[:,None];b=current[neighbors]-current[:,None]
    cross=np.einsum('nki,nkj->nij',a,b);u,s,vt=np.linalg.svd(cross)
    sign=np.ones((len(initial),3));sign[:,-1]=np.linalg.det(u@vt);rigid=(u*sign[:,None,:])@vt
    info=dict(affine=False,regularization=None,clipped_anchor_fraction=0.,reflected_anchor_fraction=0.)
    if not affine:return rigid,info
    gram=np.einsum('nki,nkj->nij',a,a)
    regularizer=np.maximum(np.trace(gram,axis1=1,axis2=2)/3*.05,1e-12)
    F=np.linalg.solve(gram+regularizer[:,None,None]*np.eye(3),cross+regularizer[:,None,None]*rigid)
    u,s,vt=np.linalg.svd(F);reflected=np.linalg.det(u@vt)<0;u[reflected,:,-1]*=-1
    clipped=((s<.25)|(s>4.)).any(1);s=np.clip(s,.25,4.)
    F=(u*s[:,None,:])@vt
    return F,dict(affine=True,regularization='0.05 times mean Gram eigenvalue toward local rigid transform',
                  singular_value_bounds=[.25,4.],clipped_anchor_fraction=float(clipped.mean()),reflected_anchor_fraction=float(reflected.mean()))

def warp(initial,current,xyz,scales,quats,binding,affine=False,covariance=False):
    neighbors,ids,weights,_=binding;F,info=transforms(initial,current,neighbors,affine)
    points=np.empty_like(xyz);new_scales=scales.copy();new_quats=quats.copy()
    for start in range(0,len(xyz),12000):
        end=min(start+12000,len(xyz));index=ids[start:end];w=weights[start:end]
        transformed=np.einsum('gki,gkij->gkj',xyz[start:end,None]-initial[index],F[index])+current[index]
        points[start:end]=np.einsum('gk,gki->gi',w,transformed)
        if covariance:
            # Isotropic initial splats: covariance transforms as F^T Sigma F
            # for row-vector coordinates. Blended local F is an approximation.
            blended=np.einsum('gk,gkij->gij',w,F[index]);_,singular,vt=np.linalg.svd(blended)
            singular=np.clip(singular,.25,4.);rotation=vt.transpose(0,2,1).copy()
            negative=np.linalg.det(rotation)<0;rotation[negative,:,0]*=-1
            new_scales[start:end]=scales[start:end]*singular
            new_quats[start:end]=Rotation.from_matrix(rotation).as_quat()[:,[3,0,1,2]]
    return points,new_scales,new_quats,info
