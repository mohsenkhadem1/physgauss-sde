"""Input-only same-initial-appearance transport; no renderer, evaluator or target reader."""
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.compositional_dynamics_v17.gaussian_transport import bind_gaussians,transport_gaussians

def covariance_parameters(covariance):
    cov=np.asarray(covariance,float)
    if cov.ndim!=3 or cov.shape[1:]!=(3,3)or not np.isfinite(cov).all()or not np.allclose(cov,cov.swapaxes(-1,-2),rtol=1e-9,atol=1e-16):raise ValueError('Finite symmetric Gaussian covariances')
    values,vectors=np.linalg.eigh(cov)
    if np.any(values<=0):raise ValueError('Positive definite Gaussian covariance')
    vectors[:,:,0]*=np.where(np.linalg.det(vectors)<0,-1,1)[:,None]
    xyzw=Rotation.from_matrix(vectors).as_quat();wxyz=xyzw[:,[3,0,1,2]]
    return np.sqrt(values),wxyz

def q2_states(model,displacements,centers,covariances):
    """Evaluate committed nodal states only; constitutive weights are never queried."""
    binding=bind_gaussians(model.geometry,centers,covariances)
    outputs=[]
    for u in displacements:
        moved=transport_gaussians(model,u,binding)
        scales,quaternions=covariance_parameters(moved['covariance_m2'])
        outputs.append(dict(centers_m=moved['centers_m'],scales_m=scales,quaternions_wxyz=quaternions,minimum_J=float(moved['determinant'].min())))
    return outputs

def native_states(particle_positions,centers,features_dc,quaternions,opacity):
    """Calls unchanged author K16 incremental motion binding, not our approximation.
    Caller pins the author module and selects its physical GPU before importing.
    """
    import torch
    from gs_render_dynamics import rollout
    if features_dc.ndim!=2 or features_dc.shape[1]!=3:raise ValueError('Author rollout expects DC feature Nx3; higher SH coefficients remain unchanged')
    points=torch.as_tensor(particle_positions,dtype=centers.dtype,device=centers.device)
    if points.ndim!=3 or points.shape[2]!=3 or points.shape[1]<16 or not torch.isfinite(points).all():raise ValueError('Finite native particle frames and K16 support required')
    xyz,rgb,quat,opa=rollout(centers.clone(),features_dc.clone(),quaternions.clone(),opacity.clone(),points,len(points),device=str(centers.device))
    if not all(torch.isfinite(a).all()for a in [xyz,rgb,quat,opa]):raise ValueError('Nonfinite native appearance transport')
    return dict(centers_m=xyz.numpy(),features_dc=rgb.numpy(),quaternions_wxyz=quat.numpy(),opacity=opa.numpy())
