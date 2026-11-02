"""Transport appearance Gaussians with the material deformation field.
Known canonical geometry is required. This module does not fit appearance or physics.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MaterialBinding:
    coordinates: np.ndarray
    offset_m: np.ndarray
    covariance_m2: np.ndarray


def bind_gaussians(geometry, centers_m, covariance_m2):
    centers=np.asarray(centers_m,float);cov=np.asarray(covariance_m2,float)
    if centers.ndim!=2 or centers.shape[1]!=3 or cov.shape!=(len(centers),3,3):
        raise ValueError('Matched Gaussian centers and full covariances required')
    if not np.isfinite(centers).all() or not np.isfinite(cov).all() or not np.allclose(cov,cov.swapaxes(-1,-2),atol=1e-15,rtol=1e-10) or np.any(np.linalg.eigvalsh(cov)<=0):
        raise ValueError('Finite centers and symmetric positive definite covariance required')
    L,H,D=geometry.dimensions
    # Clamped material-coordinate anchor; not a Euclidean closest-point claim.
    x=np.clip(centers[:,0]/L,0,1);width=1+geometry.taper*(2*x-1)
    y=np.clip(.5+centers[:,1]/(H*width),0,1)
    z=np.clip(.5+(centers[:,2]-4*geometry.arch*x*(1-x))/(D*width),0,1)
    q=np.column_stack([x,y,z]);offset=centers-geometry.map(q)
    return MaterialBinding(q,offset,cov.copy())


def transport_gaussians(model,displacement,binding):
    q=binding.coordinates
    F=np.eye(3)[None]+model.sample_material(displacement,q,gradient=True)
    if not np.isfinite(F).all() or np.any(np.linalg.det(F)<=0):
        raise ValueError('Gaussian transport requires orientation-preserving material gradients')
    centers=model.geometry.map(q)+model.sample_material(displacement,q)+np.einsum('nij,nj->ni',F,binding.offset_m)
    covariance=F@binding.covariance_m2@F.swapaxes(-1,-2)
    U,_,Vt=np.linalg.svd(F);rotation=U@Vt
    return dict(centers_m=centers,covariance_m2=covariance,material_rotation=rotation,
                determinant=np.linalg.det(F),
                appearance_scope='Geometry/covariance transport only; opacity and radiance are not refitted. Material rotation is returned for a separate radiance-frame adapter.')
