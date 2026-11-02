"""Implicit equilibrium gradients, without differentiating Newton iterations.

For R(u,z)=0 solve H^T lambda=L_u and return L_z-lambda^T R_z.
The provided tangent must be the derivative of this exact residual. Solver
regularization and unconverged states must not silently become model gradients.
"""
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve

def implicit_gradient(equilibrium,latent,residual,objective,*,tangent=None,free_dofs=None,
                      equilibrium_tolerance=1e-7,adjoint_tolerance=1e-8):
    """Return a detached scene-latent gradient and numerical diagnostics.

residual/objective accept(u,z). `tangent` is optional dense/sparse H or callable
(u,z)->H. Default computes a dense autograd Jacobian, intended only for unit tests.
Fixed DOFs must be explicitly listed viafree_dofs and independent of z; parameter-
dependent Dirichlet values require lifting inside residual/objective instead.
"""
    if not torch.is_tensor(latent)or not latent.is_floating_point():raise ValueError('Floating latent tensor required')
    z=latent.detach().clone().requires_grad_(True)
    u=torch.as_tensor(equilibrium,dtype=z.dtype,device=z.device).detach().clone().requires_grad_(True)
    r=residual(u,z)
    if r.shape!=u.shape or not bool(torch.isfinite(r).all()):raise ValueError('Finite matching residual required')
    size=u.numel();free=np.arange(size)if free_dofs is None else np.asarray(free_dofs,dtype=int)
    if free.ndim!=1 or len(free)==0 or len(np.unique(free))!=len(free)or (free<0).any()or (free>=size).any():raise ValueError('Distinct valid free DOFs required')
    residual_norm=float(torch.linalg.vector_norm(r.reshape(-1)[torch.as_tensor(free,device=u.device)]))
    if residual_norm>equilibrium_tolerance:raise ValueError('Equilibrium residual exceeds implicit-gradient tolerance')
    loss=objective(u,z)
    if loss.ndim!=0 or not bool(torch.isfinite(loss)):raise ValueError('Finite scalar objective required')
    du,direct=torch.autograd.grad(loss,(u,z),retain_graph=True,allow_unused=True)
    if du is None:du=torch.zeros_like(u)
    if direct is None:direct=torch.zeros_like(z)
    if tangent is None:
        H=torch.autograd.functional.jacobian(lambda v:residual(v.reshape(u.shape),z).reshape(-1),u.reshape(-1)).detach().cpu().numpy()
    else:
        H=tangent(u,z)if callable(tangent)else tangent
        if torch.is_tensor(H):H=H.detach().cpu().numpy()
    if H.shape!=(size,size):raise ValueError('Tangent dimension mismatch')
    Hf=H[free][:,free]if sparse.issparse(H)else np.asarray(H)[np.ix_(free,free)]
    # Independent Jacobian-vector consistency check catches mismatched assembled tangents.
    probe=np.cos(np.arange(len(free),dtype=float)+.37);probe/=np.linalg.norm(probe)
    full_probe=np.zeros(size);full_probe[free]=probe
    test_vector=torch.as_tensor(full_probe,dtype=u.dtype,device=u.device).reshape(u.shape)
    derivative=torch.autograd.grad((r*test_vector).sum(),u,retain_graph=True,allow_unused=True)[0]
    expected=np.zeros(len(free))if derivative is None else derivative.detach().cpu().numpy().reshape(-1)[free]
    tangent_error=np.linalg.norm(Hf.T@probe-expected)/max(np.linalg.norm(expected),1.)
    if not np.isfinite(tangent_error)or tangent_error>1e-7:raise ValueError('Supplied tangent disagrees with residual derivative')
    rhs=du.detach().cpu().numpy().reshape(-1)[free]
    lam=spsolve(Hf.T.tocsc(),rhs)if sparse.issparse(Hf)else np.linalg.solve(Hf.T,rhs)
    adjoint_residual=np.linalg.norm(Hf.T@lam-rhs)/max(np.linalg.norm(rhs),1.)
    if not np.isfinite(lam).all()or adjoint_residual>adjoint_tolerance:raise ValueError('Invalid adjoint solve')
    full=np.zeros(size);full[free]=lam
    vector=torch.as_tensor(full,dtype=u.dtype,device=u.device).reshape(u.shape)
    indirect=torch.autograd.grad((r*vector).sum(),z,allow_unused=True)[0]
    if indirect is None:indirect=torch.zeros_like(z)
    gradient=direct-indirect
    if not bool(torch.isfinite(gradient).all()):raise ValueError('Nonfinite latent derivative')
    return dict(loss=float(loss.detach()),gradient=gradient.detach(),direct_gradient=direct.detach(),equilibrium_gradient_correction=indirect.detach(),adjoint=vector.detach(),equilibrium_residual_norm=residual_norm,adjoint_relative_residual=float(adjoint_residual),tangent_direction_relative_error=float(tangent_error),free_dofs=len(free),scope='Exact residual implicit differentiation at a numerically converged equilibrium; no Newton-iteration graph.')

def observation_matrix(model,material_points):
    """Geometry-only fixed linear material-coordinate observation map."""
    points=np.asarray(material_points,float);matrix=np.zeros((len(points),len(model.rest)))
    for node in range(len(model.rest)):
        basis=np.zeros_like(model.rest);basis[node,:]=1
        matrix[:,node]=model.sample_material(basis,points)[:,0]
    if not np.allclose(matrix.sum(1),1,atol=1e-10):raise ValueError('Observation partition of unity failed')
    return matrix

def observe_displacement(u,matrix):
    return torch.as_tensor(matrix,dtype=u.dtype,device=u.device)@u

def linear_reaction_wrenches(model,u,attachment_stiffness):
    """Differentiable analytic bilateral wrench, including direct K dependence.

Optional measured wrench belongs inobjective only when the observation modality
allows it. Prescribed poses, not true forces, drive the mechanics residual.
"""
    if model.interface_energy is not None:raise ValueError('Use model.interface_response for nonlinear interface wrenches')
    if attachment_stiffness.shape!=(2,):raise ValueError('Two explicit support stiffnesses required')
    forces=[];torques=[]
    for side,(K,M,d)in enumerate(zip(attachment_stiffness,model.attachment_unit_matrices,model.anchor_targets)):
        matrix=torch.as_tensor(M.toarray(),dtype=u.dtype,device=u.device);target=torch.as_tensor(d,dtype=u.dtype,device=u.device)
        force=K*(matrix@(u-target));forces.append(force.sum(0))
        lever=torch.as_tensor(model.rest+d-model.tool_centers[side]-model.poses[side,:3],dtype=u.dtype,device=u.device)
        torques.append(torch.linalg.cross(lever,force,dim=-1).sum(0))
    return torch.stack(forces),torch.stack(torques)
