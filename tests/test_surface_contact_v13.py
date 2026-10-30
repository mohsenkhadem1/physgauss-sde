import numpy as np
from scripts.coupled_contact_v13.physics import make_block,Material,Collider
from scripts.coupled_contact_v13.surface_contact import SurfaceFEMSystem,box_surface_quadrature


def build(n=2,colliders=None):
    x,t=make_block((.08,.06,.05),(n,n,n),(0,0,.025))
    return SurfaceFEMSystem(x,t,Material(),colliders,gravity=(0,0,0))


def test_physical_samples_and_plane_reaction_independent_of_mesh():
    force=[];samples=[]
    for n in (2,3,4):
        s=build(n,[Collider(kind='plane',center=[0,0,.0004],damping_ratio=0,friction=0)])
        f,r,_,_,_,_=s.contact_forces();force.append(f.sum(0));samples.append(s.surface_points_rest)
        assert np.isclose(s.surface_patch_area.sum(),2*(.08*.06+.08*.05+.06*.05))
        assert np.isclose(s.surface_patch_mass.sum(),s.mass.sum())
        assert np.allclose(f.sum(0)+r[0],0,atol=1e-12)
    assert np.array_equal(samples[0],samples[1]) and np.array_equal(samples[1],samples[2])
    assert np.allclose(force,force[0],atol=1e-10)


def test_virtual_work_and_torque_consistent_distribution():
    s=build();rng=np.random.default_rng(21);s.x+=rng.normal(0,.0001,s.x.shape)
    force=rng.normal(size=s.surface_points_rest.shape);dx=rng.normal(size=s.x.shape)
    nodal=s.distribute_surface_force(force);points,_=s.surface_state()
    dp=np.einsum('qi,qic->qc',s.surface_barycentric,dx[s.surface_node_ids])
    assert np.isclose(np.sum(nodal*dx),np.sum(force*dp),rtol=1e-12)
    assert np.allclose(nodal.sum(0),force.sum(0),atol=1e-10)
    assert np.allclose(np.cross(s.x,nodal).sum(0),np.cross(points,force).sum(0),atol=1e-10)


def test_contact_updates_after_shape_change_and_passive_friction():
    c=Collider(kind='plane',center=[0,0,-.001]);s=build(colliders=[c]);assert s.contact_forces()[-1]==0
    s.x[:,2]-=.002;s.v[:,0]=.2
    f,r,t,d,p,count=s.contact_forces()
    assert count>0 and d>0 and p<0
    assert np.allclose(f.sum(0)+r[0],0,atol=1e-10)
    assert np.allclose(np.cross(s.x,f).sum(0)+np.cross(c.center,r[0])+t[0],0,atol=1e-10)


def test_narrow_edge_contact_exists_on_even_and_odd_mesh():
    edge=Collider(kind='box',center=[.01,0,.025],half_extents=[.008,.07,.025],damping_ratio=0)
    reactions=[]
    for n in (2,3,4):
        s=build(n,[edge]);s.x[:,0]+=.008;s.x[:,2]+=.049
        f,r,t,d,p,count=s.contact_forces()
        assert count>0 and f[:,2].sum()>0
        reactions.append(r[0])
    assert np.allclose(reactions,reactions[0],atol=1e-9)


def test_isolated_surface_contact_preserves_coupled_momentum():
    ball=Collider(kind='sphere',center=[.057,0,.025],radius=.02,mass=.03,name='ball',stiffness=800)
    s=build(colliders=[ball]);s.v[:]=[.1,.03,0]
    p=s.momentum();angular=s.angular_momentum()
    for _ in range(100):s.step(1e-4)
    assert s.contact_samples>0
    assert np.linalg.norm(s.momentum()-p)<1e-11
    assert np.linalg.norm(s.angular_momentum()-angular)<1e-11
    assert s.friction_work<=0
