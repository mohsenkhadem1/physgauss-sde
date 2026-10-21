import numpy as np
import pytest
from experiments.campaign_rigid_inertia import auxiliary_mass_points,mass_properties,rigid_point_velocities


def test_exact_mass_COM_and_full_nondiagonal_inertia():
    rng=np.random.default_rng(72)
    for _ in range(12):
        A=rng.normal(size=(3,3));S=A@A.T*1e-8
        inertia=np.trace(S)*np.eye(3)-S;mass=float(rng.uniform(.001,.1))
        r,m=auxiliary_mass_points(mass,inertia)
        total,com,I=mass_properties(r,m)
        assert total==pytest.approx(mass,rel=1e-15)
        np.testing.assert_allclose(com,0,atol=1e-17)
        np.testing.assert_allclose(I,inertia,rtol=1e-13,atol=1e-22)


def test_finite_rotated_kinetic_energy_linear_and_angular_momentum_match():
    I=np.diag([2.,3.,4.])*1e-8;mass=.001
    r,m=auxiliary_mass_points(mass,I)
    angle=.83;R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    v=np.array([.02,-.03,.07]);omega=np.array([.4,.8,-1.3])
    velocity=rigid_point_velocities(r,R,v,omega);world_r=r@R.T
    expected=.5*mass*(v@v)+.5*omega@(R@I@R.T)@omega
    assert .5*np.sum(m[:,None]*velocity**2)==pytest.approx(expected,rel=1e-14)
    np.testing.assert_allclose((m[:,None]*velocity).sum(0),mass*v,rtol=1e-14,atol=1e-20)
    np.testing.assert_allclose(np.cross(world_r,m[:,None]*velocity).sum(0),R@I@R.T@omega,rtol=1e-13,atol=1e-20)


def test_planar_mass_distribution_has_no_artificial_thickness_or_inertia():
    I=np.diag([2.,3.,5.])*1e-8
    r,m=auxiliary_mass_points(.001,I)
    assert np.max(abs(r[:,2]))==0
    np.testing.assert_allclose(mass_properties(r,m)[2],I,atol=1e-22,rtol=1e-14)


@pytest.mark.parametrize('mass,inertia',[(0,[1,1,1]),(.001,[1,1,4]),(.001,[0,1,1]),(.001,[[1,.1,0],[0,1,0],[0,0,1]])])
def test_invalid_mass_or_unphysical_inertia_rejected(mass,inertia):
    with pytest.raises(ValueError):auxiliary_mass_points(mass,inertia)


def test_invalid_rotation_is_not_silently_projected():
    r,m=auxiliary_mass_points(.001,[2e-8,3e-8,4e-8])
    with pytest.raises(ValueError,match='SO'):rigid_point_velocities(r,np.eye(3)*1.01,[0,0,0],[0,0,1])
