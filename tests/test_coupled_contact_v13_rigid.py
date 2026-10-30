import numpy as np
import pytest
from scripts.coupled_contact_v13.physics import make_block,Collider
from scripts.coupled_contact_v13.rigid import RigidSystem


def test_free_rigid_preserves_shape_and_momenta():
    rest,tets=make_block(cells=(1,1,1));system=RigidSystem(rest,tets,gravity=(0,0,0))
    system.v[:]=[.01,.02,.03]+np.cross([.3,-.1,.4],rest-system.center)
    p=system.momentum();l=system.angular_momentum()
    for _ in range(100):system.step(.0001)
    np.testing.assert_allclose(system.momentum(),p,atol=1e-14)
    np.testing.assert_allclose(system.angular_momentum(),l,atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(system.deformation()),1,atol=1e-12)
    old=np.linalg.norm(rest[:,None]-rest[None],axis=2)
    new=np.linalg.norm(system.x[:,None]-system.x[None],axis=2)
    np.testing.assert_allclose(new,old,atol=1e-14)


def test_rigid_dynamic_contact_balances_force_torque():
    rest,tets=make_block(cells=(1,1,1))
    sphere=Collider(kind='sphere',center=[.04,0,.04],radius=.035,mass=.03,friction=.4)
    system=RigidSystem(rest,tets,colliders=[sphere],gravity=(0,0,0));system.v[:]=[0,.02,0]
    p=system.momentum();l=system.angular_momentum()
    system.step(.00001)
    assert system.contact_samples>0
    np.testing.assert_allclose(system.momentum(),p,atol=1e-13)
    np.testing.assert_allclose(system.angular_momentum(),l,atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(system.deformation()),1,atol=1e-12)


def test_rigid_fixed_fixture_explicitly_rejected():
    rest,tets=make_block(cells=(1,1,1));fixed=np.zeros(len(rest),bool);fixed[0]=True
    with pytest.raises(ValueError,match='fixed-node'):RigidSystem(rest,tets,fixed=fixed)
