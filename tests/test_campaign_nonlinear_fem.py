import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from campaign_linear_fem import assemble_tetrahedra, checked_cholesky, tensor
from campaign_nonlinear_fem import (FiniteRigidPatchMap, neo_hookean_energy,
    rotation_matrix, InvalidConfiguration, solve_static, equilibrium_metrics)


def skew_tet():
    return np.array([[.2,-.1,.3],[1.3,.1,.4],[.4,1.2,.2],[.3,.2,1.5]]),np.array([[0,1,2,3]])


def test_energy_objective_under_finite_rigid_rotation_and_translation():
    points,tets=skew_tet();E,nu=1234.,.34
    _,_,grad,vol=assemble_tetrahedra(points,tets,E,nu,600.)
    Q=rotation_matrix(tensor([1.2,-.5,.7]))
    # A finite deformation has equal energy after an arbitrary common rotation.
    deformation=tensor([[1.1,.07,0],[.01,.96,.03],[0,.02,1.04]])
    x=tensor(points)@deformation.T
    energy,J=neo_hookean_energy(x,points,tets,grad,vol,E,nu)
    rotated,J2=neo_hookean_energy(x@Q.T+tensor([-.3,.7,1.1]),points,tets,grad,vol,E,nu)
    torch.testing.assert_close(rotated,energy,atol=2e-12,rtol=2e-12)
    torch.testing.assert_close(J2,J,atol=2e-14,rtol=2e-14)
    rigid,_=neo_hookean_energy(tensor(points)@Q.T+tensor([.7,-.2,.4]),points,tets,grad,vol,E,nu)
    assert abs(float(rigid))<1e-12


def test_rest_tangent_matches_independent_linear_tet_assembly():
    points,tets=skew_tet();E,nu=1234.,.34
    K,_,grad,vol=assemble_tetrahedra(points,tets,E,nu,600.)
    def energy(u):
        return neo_hookean_energy(tensor(points)+u.reshape(-1,3),points,tets,grad,vol,E,nu)[0]
    zero=torch.zeros(12,dtype=torch.float64,requires_grad=True)
    tangent=torch.autograd.functional.hessian(energy,zero)
    torch.testing.assert_close(tangent,K,rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(torch.autograd.grad(energy(zero),zero)[0],torch.zeros_like(zero),atol=1e-13,rtol=0)


def test_exact_patch_rotation_preserves_distances_and_fixture_with_zero_tangent_consistency():
    points=np.array([[0.,0,0],[.1,0,0],[0,.1,0],[.2,.1,.1],[.3,.1,.1],[.2,.2,.1],[.4,.2,.2]])
    patches=[[0,1,2],[3,4,5]]
    mapping=FiniteRigidPatchMap(points,[6],patches,[0,3])
    z=torch.zeros(mapping.size,dtype=torch.float64)
    for j,ids in enumerate(mapping.handle_dofs):
        z[ids[:3]]=tensor([.1*j,-.2,.3]);z[ids[3:]]=tensor([1.2,-.3,.4])
    x=mapping.positions(z)
    for patch in patches:
        torch.testing.assert_close(torch.cdist(x[patch],x[patch]),torch.cdist(tensor(points)[patch],tensor(points)[patch]),atol=1e-14,rtol=1e-13)
    torch.testing.assert_close(x[6],tensor(points)[6],atol=0,rtol=0)
    jacobian=torch.autograd.functional.jacobian(lambda value:mapping.positions(value).flatten(),torch.zeros_like(z))
    torch.testing.assert_close(jacobian,mapping.A,atol=1e-14,rtol=1e-14)


def test_energy_force_tangent_finite_differences_away_from_rest_and_inversion_rejected():
    points,tets=skew_tet();E,nu=1234.,.34
    _,_,grad,vol=assemble_tetrahedra(points,tets,E,nu,600.)
    def energy(u):return neo_hookean_energy(tensor(points)+u.reshape(-1,3),points,tets,grad,vol,E,nu)[0]
    u=tensor(np.random.default_rng(17).normal(0,.02,12)).requires_grad_()
    direction=tensor(np.random.default_rng(18).normal(size=12));direction/=direction.norm()
    g=torch.autograd.grad(energy(u),u,create_graph=True)[0]
    H=torch.autograd.functional.hessian(energy,u)
    step=1e-5
    fd_energy=(energy(u+step*direction)-energy(u-step*direction))/(2*step)
    torch.testing.assert_close(fd_energy,g@direction,atol=1e-7,rtol=1e-7)
    def gradient(v):
        v=v.detach().requires_grad_();return torch.autograd.grad(energy(v),v)[0]
    fd_gradient=(gradient(u+step*direction)-gradient(u-step*direction))/(2*step)
    torch.testing.assert_close(fd_gradient,H@direction,atol=2e-6,rtol=2e-7)
    inverted=tensor(points).clone();inverted[[1,2]]=inverted[[2,1]]
    with pytest.raises(InvalidConfiguration):neo_hookean_energy(inverted,points,tets,grad,vol,E,nu)


def test_physical_torque_residual_correctly_converts_finite_rotation_chart_gradient():
    points=np.array([[0.,0,0],[.1,0,0],[0,.1,0]])
    mapping=FiniteRigidPatchMap(points,[],[[0,1,2]],[0])
    z=tensor([.1,-.2,.3,1.2,-.5,.7]).requires_grad_()
    forces=tensor([[.2,.1,0],[.3,-.2,.4],[-.1,.3,.2]])
    x=mapping.positions(z)
    residual=torch.autograd.grad((x*forces).sum(),z)[0]
    expected=torch.linalg.cross(x-x[0],forces).sum(0)
    metrics=equilibrium_metrics(residual,torch.zeros_like(residual),mapping.handle_dofs,.1,z)
    assert metrics['maximum_handle_torque_residual_Nm']==pytest.approx(float(expected.detach().abs().max()),rel=1e-13,abs=1e-14)
    assert abs(metrics['maximum_axis_angle_gradient_Nm']-metrics['maximum_handle_torque_residual_Nm'])>1e-3


def test_nonlinear_equilibrium_small_load_limit_and_force_residual():
    # Three fixed noncollinear nodes eliminate all rigid modes; fourth node free.
    points,tets=skew_tet();E,nu=1234.,.34
    K,_,grad,vol=assemble_tetrahedra(points,tets,E,nu,600.)
    mapping=FiniteRigidPatchMap(points,[0,1,2],[],[])
    Kr=mapping.A.T@K@mapping.A;factor,_=checked_cholesky(Kr)
    force=tensor([.1,-.04,.06]);relative_errors=[]
    for scale in [1.,.1,.01]:
        solution=solve_static(mapping,tets,grad,vol,E,nu,factor,force*scale,relative_tolerance=1e-8)
        linear=torch.cholesky_solve((force*scale)[:,None],factor)[:,0]
        relative_errors.append(float((solution['z']-linear).norm()/linear.norm()))
        assert solution['metrics']['scaled_generalized_residual_norm_N']<=solution['residual_tolerance_N']
        assert solution['minimum_J']>0
    assert relative_errors[1]<.15*relative_errors[0]
    assert relative_errors[2]<.15*relative_errors[1]


def test_invalid_tangent_and_iteration_exhaustion_are_not_repaired():
    with pytest.raises(ValueError,match='no regularizer'):
        checked_cholesky(torch.zeros((3,3),dtype=torch.float64))
    points,tets=skew_tet();E,nu=1234.,.34
    K,_,grad,vol=assemble_tetrahedra(points,tets,E,nu,600.)
    mapping=FiniteRigidPatchMap(points,[0,1,2],[],[])
    factor,_=checked_cholesky(mapping.A.T@K@mapping.A)
    with pytest.raises(RuntimeError,match='did not meet declared tolerance'):
        solve_static(mapping,tets,grad,vol,E,nu,factor,tensor([.1,.2,.3]),maximum_iterations=0)
