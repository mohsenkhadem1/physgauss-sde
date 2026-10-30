import copy
import numpy as np
from scripts.coupled_contact_v13.physics import make_block,Material,Collider,FEMSystem
from scripts.coupled_contact_v13.consistent_mass import ConsistentFEMSystem,ConsistentRigidSystem,ConsistentClippedFEMSystem,with_consistent_mass
from scripts.coupled_contact_v13.clipped_surface_contact import ClippedSurfaceFEMSystem


def setup(cells=(2,2,2),**kwargs):
    size=np.array([.08,.06,.05]);rest,tets=make_block(size=size,cells=cells,center=(.03,-.02,.1))
    return size,rest,tets,ConsistentFEMSystem(rest,tets,material=Material(viscosity=0),**kwargs)


def test_affine_kinetic_energy_exact_continuum_block():
    for cells in [(1,1,1),(2,2,2),(3,3,3)]:
        size,rest,tets,system=setup(cells);center=rest.mean(0);mass=1000*np.prod(size)
        A=np.array([[.3,.2,-.1],[.4,-.2,.6],[-.1,.2,.7]]);velocity=np.array([.2,-.3,.1])
        system.v=velocity+(rest-center)@A.T
        expected=.5*mass*(velocity@velocity+np.trace(A@np.diag(size**2/12)@A.T))
        np.testing.assert_allclose(system.kinetic_energy(),expected,rtol=1e-13,atol=1e-15)
        np.testing.assert_allclose(system.momentum(),mass*velocity,rtol=1e-13,atol=1e-15)


def test_rigid_rotation_energy_momentum_and_exact_inertia():
    size,rest,tets,system=setup();mass=1000*np.prod(size);center=rest.mean(0)
    omega=np.array([.3,-.4,.2]);velocity=np.array([.1,.2,-.1])
    inertia=mass/12*np.diag([size[1]**2+size[2]**2,size[0]**2+size[2]**2,size[0]**2+size[1]**2])
    system.v=velocity+np.cross(omega,rest-center)
    np.testing.assert_allclose(system.kinetic_energy(),.5*mass*(velocity@velocity)+.5*omega@inertia@omega,rtol=1e-13)
    np.testing.assert_allclose(system.angular_momentum(),np.cross(center,mass*velocity)+inertia@omega,atol=1e-15)
    rigid=ConsistentRigidSystem(rest,tets,gravity=(0,0,0));np.testing.assert_allclose(rigid.body_inertia,inertia,rtol=1e-13,atol=1e-15)
    rigid.v=system.v.copy();p=rigid.momentum();l=rigid.angular_momentum()
    for _ in range(25):rigid.step(1e-5)
    np.testing.assert_allclose(rigid.momentum(),p,atol=1e-14);np.testing.assert_allclose(rigid.angular_momentum(),l,atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(rigid.deformation()),1,atol=1e-12)


def test_uniform_gravity_and_static_fixed_constraint_coupling():
    size,rest,tets,system=setup(gravity=(0,0,-9.81));dt=1e-5;system.step(dt)
    np.testing.assert_allclose(system.v,np.broadcast_to([0,0,-9.81*dt],rest.shape),atol=1e-13)
    fixed=np.isclose(rest[:,0],rest[:,0].min());clamped=ConsistentFEMSystem(rest,tets,fixed=fixed)
    clamped.step(dt)
    np.testing.assert_array_equal(clamped.x[fixed],rest[fixed]);np.testing.assert_array_equal(clamped.v[fixed],0)
    force=clamped.mass[:,None]*clamped.gravity;acc=clamped.v/dt
    reaction=clamped.mass_matrix@acc-force
    np.testing.assert_allclose(reaction[~fixed],0,atol=1e-13)
    np.testing.assert_allclose(clamped.last_fixed_reaction,reaction[fixed].sum(0),atol=1e-13)
    assert np.linalg.norm(reaction[fixed])>0
    # Coupling matters: reactions are not simply minus local body forces.
    assert np.linalg.norm(reaction[fixed]+force[fixed])>1e-3


def test_mass_momentum_torque_balance_under_nodal_impulse():
    size,rest,tets,system=setup(gravity=(0,0,0));rng=np.random.default_rng(13)
    system.v=rng.normal(0,.01,rest.shape);force=rng.normal(0,.001,rest.shape)
    p=system.momentum();l=system.angular_momentum();dt=1e-6
    system.v+=dt*system.mass_acceleration(force);system.x+=dt*system.v
    np.testing.assert_allclose(system.momentum()-p,dt*force.sum(0),atol=1e-17)
    np.testing.assert_allclose(system.angular_momentum()-l,dt*np.cross(rest,force).sum(0),atol=1e-17)


def test_same_contact_and_native_shared_mass_operator():
    from scripts.coupled_contact_v13.native_adapter import NativeSpringSystem
    rest,tets=make_block(cells=(1,1,1),center=(0,0,.019));cols=[Collider(kind='plane',name='floor')]
    old=ClippedSurfaceFEMSystem(rest,tets,colliders=copy.deepcopy(cols));new=ConsistentClippedFEMSystem(rest,tets,colliders=copy.deepcopy(cols))
    old.v[:]=[.01,0,-.02];new.v[:]=old.v
    for a,b in zip(old.contact_forces()[:3],new.contact_forces()[:3]):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(old.mass,new.mass)
    native=with_consistent_mass(NativeSpringSystem)(rest,tets)
    np.testing.assert_array_equal(native.mass_matrix.toarray(),new.mass_matrix.toarray())
    assert native.step.__func__ is new.step.__func__
