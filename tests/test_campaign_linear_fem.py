import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from campaign_linear_fem import (assemble_tetrahedra,condensation,reduced_matrices,
    checked_cholesky,midpoint_dynamics,tensor,skew)


def unit_tet():
    return np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]]),np.array([[0,1,2,3]])


def test_affine_patch_energy_and_density_volume_mass_are_exact():
    x,tets=unit_tet();E,nu,rho=1200.,.3,600.
    K,mass,gradient,volume=assemble_tetrahedra(x,tets,E,nu,rho)
    G=tensor([[.01,.02,-.01],[-.003,.015,.007],[.002,.004,-.005]])
    u=(tensor(x)@G.T+tensor([.1,-.3,.2])).flatten()
    epsilon=.5*(G+G.T);mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
    expected=(.5*lam*torch.trace(epsilon)**2+mu*(epsilon**2).sum())/6
    torch.testing.assert_close(.5*u@K@u,expected,rtol=1e-12,atol=1e-13)
    assert float(volume.sum())==pytest.approx(1/6)
    assert float(mass.sum()/3)==pytest.approx(rho/6)
    torch.testing.assert_close(gradient.sum(1),torch.zeros((1,3),dtype=torch.float64),atol=1e-15,rtol=0)


def test_rigid_translation_rotation_nullspace_and_pure_neumann_rejected():
    x,tets=unit_tet();K,_,_,_=assemble_tetrahedra(x,tets,1000.,.3,600.)
    for u in [np.tile([.2,-.3,.1],(4,1)),np.cross(np.tile([.01,.02,.03],(4,1)),x)]:
        assert torch.linalg.vector_norm(K@tensor(u).flatten())<1e-12
    with pytest.raises(ValueError,match='no regularizer'):checked_cholesky(K)


def test_rigid_patch_exact_linear_kinematics_virtual_work_and_kinetic_energy():
    points=np.array([[0.,0,0],[.1,0,0],[0,.1,0],[1.,0,0],[1.1,0,0],[1.,.1,0],[2.,2,2.]])
    patches=[[0,1,2],[3,4,5]];centers=[0,3]
    A,ids=condensation(points,[],patches,centers)
    z=torch.zeros(A.shape[1],dtype=torch.float64)
    for j in range(2):
        z[ids[j][:3]]=tensor([.01,-.02,.03]);z[ids[j][3:]]=tensor([.04,.02,-.01])
    u=(A@z).reshape(-1,3)
    for j,patch in enumerate(patches):
        expected=z[ids[j][:3]]+torch.linalg.cross(z[ids[j][3:]].expand(3,3),tensor(points[patch]-points[centers[j]]))
        torch.testing.assert_close(u[patch],expected,atol=1e-15,rtol=0)
    f=torch.zeros(3*len(points),dtype=torch.float64);force=tensor([2.,3.,4.]);f[3:6]=force
    projected=A.T@f
    torch.testing.assert_close(projected[ids[0][:3]],force)
    torch.testing.assert_close(projected[ids[0][3:]],torch.linalg.cross(tensor(points[1]-points[0]),force))
    mass=tensor(np.repeat(np.arange(1,len(points)+1)*.001,3));inertias=[[.0001,.0002,.0003],[.0002,.0003,.0004]]
    _,M=reduced_matrices(torch.eye(len(f),dtype=torch.float64),mass,A,ids,[.001,.002],inertias)
    expected=.5*(mass*(A@z)**2).sum()
    for j,m in enumerate([.001,.002]):
        expected+=.5*m*z[ids[j][:3]].square().sum()+.5*(tensor(inertias[j])*z[ids[j][3:]].square()).sum()
    torch.testing.assert_close(.5*z@M@z,expected,atol=1e-16,rtol=1e-13)


def test_fixture_elimination_exact_and_overlap_rejected():
    x=np.arange(18,dtype=float).reshape(6,3)
    A,ids=condensation(x,[0],[[1,2],[3,4]],[1,3])
    assert not A[:3].any()
    with pytest.raises(ValueError,match='Overlapping'):condensation(x,[0],[[0,1]],[0])


def test_implicit_midpoint_free_unforced_translation_and_harmonic_energy():
    # With no force/stiffness, free motion must preserve velocity exactly.
    K=tensor([[0.]]);M=tensor([[2.]])
    u,v,times,_=midpoint_dynamics(K,M,tensor([0.]),0,20,initial_velocity=[.3])
    np.testing.assert_allclose(u[:,0],.3*times,atol=1e-14)
    np.testing.assert_allclose(v[:,0],.3,atol=1e-14)
    K=tensor([[8.]])
    u,v,_,_=midpoint_dynamics(K,M,tensor([0.]),0,80,initial_displacement=[.1])
    energy=.5*2*v[:,0]**2+.5*8*u[:,0]**2
    torch.testing.assert_close(energy,torch.full_like(energy,.04),atol=2e-14,rtol=1e-12)


def test_dynamic_endpoint_error_has_second_order_timestep_contraction():
    errors=[]
    # Unforced oscillator exact solution; fixed3s target avoids response labels.
    for hz in [20,40,80]:
        u,_,times,_=midpoint_dynamics(tensor([[8.]]),tensor([[2.]]),tensor([0.]),0,hz,initial_displacement=[.1])
        errors.append(abs(float(u[-1,0])-.1*np.cos(2*times[-1])))
    assert errors[1]/errors[0]<.3 and errors[2]/errors[1]<.3
