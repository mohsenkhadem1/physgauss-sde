"""Positive high-order P1 surface Gaussian operators; a NEW discretization.

For normalized density psi/Z on the rest boundary, load l_i=integral psi*N_i/Z
and fixture Gram M_ij=integral psi*N_i*N_j/Z. Full world K/D tensors may be
Kronecker-multiplied with M; no extra mass, diagonalization or PSD repair.
This does not modify or relabel the old vertex-lumped physical simulations.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse


def triangle_rule(order):
    """Positive product Gauss-Legendre Duffy rule; reference area is1/2."""
    if type(order) is not int or not 2<=order<=64:raise ValueError('Explicit finite order2..64 required')
    x,w=np.polynomial.legendre.leggauss(order);u=(x+1)/2;w=w/2
    U,V=np.meshgrid(u,u,indexing='ij');W=np.outer(w,w)*U
    bary=np.stack([1-U,U*(1-V),U*V],axis=-1).reshape(-1,3)
    return bary,W.ravel()


@dataclass(frozen=True)
class SurfaceGaussianOperators:
    load_vectors:np.ndarray
    fixture_grams:tuple
    normalizers_m2:np.ndarray
    centroids_material_m:np.ndarray
    second_moments_material_m2:np.ndarray
    quadrature_order:int
    semantics:str='Normalized continuous restsurface Gaussian P1 Galerkin; NEW discrete assembly, not frozen lumped-response replacement'


def assemble_surface_gaussians(points,faces,centers,covariances,*,order=12,face_chunk=128):
    X=np.asarray(points,dtype=np.float64);F=np.asarray(faces);C=np.asarray(centers,dtype=np.float64);S=np.asarray(covariances,dtype=np.float64)
    if (X.ndim!=2 or X.shape[1]!=3 or F.ndim!=2 or F.shape[1]!=3 or F.dtype.kind not in 'iu' or not len(F)
        or np.any(F<0) or np.any(F>=len(X)) or C.ndim!=2 or C.shape[1]!=3 or S.shape!=(len(C),3,3)
        or not len(C) or not all(np.isfinite(a).all() for a in [X,C,S]) or type(face_chunk) is not int or face_chunk<1):raise ValueError('Explicit finite material surface and Gaussian bank required')
    if len(np.unique(np.sort(F,axis=1),axis=0))!=len(F):raise ValueError('Duplicate surface triangle')
    if np.max(abs(S-S.transpose(0,2,1)))>64*np.finfo(float).eps*np.max(abs(S)):raise ValueError('Symmetric covariance required')
    chol=np.linalg.cholesky(S);bary,weight=triangle_rule(order);tri=X[F]
    jac=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    if np.any(jac<=0):raise ValueError('Nondegenerate surface required')
    raw_load=np.zeros((len(C),len(X)));local_grams=np.empty((len(C),len(F),3,3))
    for start in range(0,len(F),face_chunk):
        stop=min(start+face_chunk,len(F));x=np.einsum('qi,fic->fqc',bary,tri[start:stop])
        for p in range(len(C)):
            whitened=np.linalg.solve(chol[p],(x-C[p]).reshape(-1,3).T).T
            density=np.exp(-.5*np.sum(whitened*whitened,axis=1)).reshape(stop-start,-1)
            weighted=density*jac[start:stop,None]*weight[None]
            local_load=np.einsum('fq,qi->fi',weighted,bary)
            for j in range(3):np.add.at(raw_load[p],F[start:stop,j],local_load[:,j])
            local_grams[p,start:stop]=np.einsum('fq,qi,qj->fij',weighted,bary,bary,optimize=True)
    normalizers=raw_load.sum(1)
    if not np.isfinite(normalizers).all() or np.any(normalizers<=0):raise ValueError('Positive resolved Gaussian normalizer required; no nearestnode repair')
    loads=raw_load/normalizers[:,None];rows=np.repeat(F,3,axis=1).ravel();cols=np.tile(F,(1,3)).ravel();grams=[];seconds=[]
    for p in range(len(C)):
        M=sparse.coo_matrix((local_grams[p].ravel()/normalizers[p],(rows,cols)),shape=(len(X),len(X))).tocsr()
        grams.append(M);seconds.append(X.T@M@X)
    return SurfaceGaussianOperators(loads,tuple(grams),normalizers,loads@X,np.asarray(seconds),order)
