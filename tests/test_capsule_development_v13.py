import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.coupled_contact_v13.capsule_development import CapsuleCollider,build
from scripts.coupled_contact_v13 import engineered_scenarios
from scripts.coupled_contact_v13.physics import make_block
from scripts.coupled_contact_v13.consistent_mass import ConsistentClippedFEMSystem
from scripts.coupled_contact_v13.surface_contact import SurfaceContactMixin


def test_capsule_distance_normal_lipschitz_and_gradient():
    c=CapsuleCollider(radius=.008,segment_length=.124)
    points=np.array([[.008,0,0],[0,.07,0],[0,0,.008],[0,0,0],[.018,.062,0]])
    phi,n=c.query(points)
    np.testing.assert_allclose(phi,[0,0,0,-.008,.01],atol=1e-15)
    np.testing.assert_allclose(n[[0,1,2,4]],[[1,0,0],[0,1,0],[0,0,1],[1,0,0]],atol=1e-15)
    rng=np.random.default_rng(9);x=rng.uniform(-.1,.1,(200,3));y=rng.uniform(-.1,.1,(200,3))
    a,normal=c.query(x);b,_=c.query(y)
    assert np.all(abs(a-b)<=np.linalg.norm(x-y,axis=1)+1e-14)
    grad=np.column_stack([(c.query(x+np.eye(3)[k]*1e-7)[0]-c.query(x-np.eye(3)[k]*1e-7)[0])/2e-7 for k in range(3)])
    np.testing.assert_allclose(grad,normal,atol=1e-8)
    with pytest.raises(ValueError):CapsuleCollider(mass=1.)


def test_capsule_objectivity_and_rail_geometry_preserves_original():
    R=Rotation.from_rotvec([.2,-.4,.3]).as_matrix();t=np.array([.1,.2,-.1])
    c=CapsuleCollider(radius=.008);d=CapsuleCollider(radius=.008,center=t,rotation=R)
    x=np.array([[.008,.01,.003],[-.02,.09,.04]])
    a,n=c.query(x);b,m=d.query(x@R.T+t)
    np.testing.assert_allclose(a,b,atol=1e-15);np.testing.assert_allclose(n@R.T,m,atol=1e-15)
    old=engineered_scenarios.build('edge_drop');new=build('edge_drop')
    np.testing.assert_array_equal(old.rest,new.rest)
    assert old.colliders[-1].kind=='box' and new.colliders[-1].kind=='capsule'
    rail=new.colliders[-1]
    assert np.isclose(rail.segment_length+2*rail.radius,.14)
    assert np.isclose(rail.center[2]+rail.radius,.05)
    assert rail.stiffness==old.colliders[-1].stiffness==76800.


def test_capsule_sparse_equals_exhaustive_and_force_torque_balance():
    rest,tets=make_block(size=(.025,.03,.02),cells=(2,2,2),center=(.007,0,.01))
    collider=CapsuleCollider(center=(0,0,0),radius=.008,friction=.4,stiffness=1000.,velocity=(.02,0,.01),omega=(0,.2,0))
    system=ConsistentClippedFEMSystem(rest,tets,colliders=[collider])
    rng=np.random.default_rng(5);system.x+=rng.normal(0,.0001,system.x.shape);system.v[:]=rng.normal(0,.01,system.v.shape)
    sparse=system.contact_forces();dense=SurfaceContactMixin.contact_forces(system)
    for a,b in zip(sparse[:3],dense[:3]):np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-11)
    force,reactions,torques,depth,_,count=sparse
    assert depth>0 and count>0
    np.testing.assert_allclose(force.sum(0)+reactions[0],0,atol=1e-11)
    np.testing.assert_allclose(np.cross(system.x,force).sum(0)+np.cross(collider.center,reactions[0])+torques[0],0,atol=1e-11)
    # Same objective force law after a rigid world-frame transformation.
    R=Rotation.from_rotvec([.1,.3,-.2]).as_matrix();shift=np.array([.2,-.1,.05])
    before=system.contact_forces()
    system.x=system.x@R.T+shift;system.v=system.v@R.T
    collider.center=collider.center@R.T+shift;collider.rotation=R@collider.rotation
    collider.velocity=collider.velocity@R.T;collider.omega=collider.omega@R.T
    after=system.contact_forces()
    for a,b in zip(before[:3],after[:3]):np.testing.assert_allclose(np.asarray(a)@R.T,b,rtol=1e-10,atol=1e-10)
