import numpy as np
from scipy.spatial.transform import Rotation
from scripts.coupled_contact_v13.physics import FEMSystem,Material,make_block
from scripts.coupled_contact_v13.native_adapter import NativeSpringSystem


def test_native_kernel_parity_force_balance_and_mass():
    rest,tets=make_block(cells=(1,1,1));material=Material(young=10000.,viscosity=.2)
    model=NativeSpringSystem(rest,tets,material=material)
    shared=FEMSystem(rest,tets,material=material)
    np.testing.assert_array_equal(model.mass,shared.mass)
    model.x=rest*np.array([1.04,.98,1.01]);model.v=np.arange(rest.size).reshape(rest.shape)*.0001
    force,_=model.internal_forces()
    edge=model.edges;delta=model.x[edge[:,1]]-model.x[edge[:,0]];length=np.linalg.norm(delta,axis=1);direction=delta/length[:,None]
    velocity=model.v[edge[:,1]]-model.v[edge[:,0]]
    magnitude=model.coefficients*(length/model.lengths-1)+model.dashpot*np.sum(velocity*direction,axis=1)
    expected=np.zeros_like(rest);np.add.at(expected,edge[:,0],magnitude[:,None]*direction);np.add.at(expected,edge[:,1],-magnitude[:,None]*direction)
    np.testing.assert_allclose(force,expected,rtol=1e-4,atol=2e-7)
    np.testing.assert_allclose(force.sum(0),0,atol=1e-7)
    assert model.metadata()['full_author_method'] is False


def test_native_rigid_rotation_no_elastic_response():
    rest,tets=make_block(cells=(1,1,1));model=NativeSpringSystem(rest,tets,material=Material(viscosity=0.))
    model.x=rest@Rotation.from_rotvec([.1,.4,-.2]).as_matrix()+[.1,.2,.3]
    force,J=model.internal_forces()
    np.testing.assert_allclose(J,1,atol=1e-12)
    np.testing.assert_allclose(force,0,atol=2e-6)
    assert model.elastic_energy()<1e-20


def test_native_shared_contact_force_and_reaction():
    from scripts.coupled_contact_v13.physics import Collider
    import copy
    rest,tets=make_block(cells=(1,1,1),center=(0,0,.019))
    collider=Collider(kind='plane',center=np.zeros(3),name='floor')
    native=NativeSpringSystem(rest,tets,colliders=[copy.deepcopy(collider)])
    fem=FEMSystem(rest,tets,colliders=[copy.deepcopy(collider)])
    native.v[:]=[.03,0,-.01];fem.v[:]=native.v
    nf,nr,nt,nd,npower,ncount=native.contact_forces()
    ff,fr,ft,fd,fpower,fcount=fem.contact_forces()
    np.testing.assert_array_equal(nf,ff)
    np.testing.assert_array_equal(nr,fr)
    np.testing.assert_array_equal(nt,ft)
    assert (nd,npower,ncount)==(fd,fpower,fcount)
    assert ncount>0
