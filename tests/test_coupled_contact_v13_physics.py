import numpy as np
import pytest
from scripts.coupled_contact_v13.physics import Material,Collider,FEMSystem,make_block,simulate


def system(**kw):
    x,t=make_block(cells=(2,2,2));return FEMSystem(x,t,gravity=(0,0,0),**kw)


def test_mesh_volume_and_rest_equilibrium():
    s=system();f,J=s.internal_forces()
    assert np.isclose(s.volume.sum(),.06*.04*.04)
    assert np.max(np.abs(f))<1e-12
    assert np.allclose(J,1)


def test_force_is_energy_gradient():
    s=system(material=Material(viscosity=0));rng=np.random.default_rng(44)
    s.x+=rng.normal(size=s.x.shape)*.0002
    f,_=s.internal_forces();direction=rng.normal(size=s.x.shape)
    h=1e-8
    numeric=(s.elastic_energy(s.x+h*direction)-s.elastic_energy(s.x-h*direction))/(2*h)
    assert np.isclose(numeric,-np.sum(f*direction),rtol=2e-7,atol=1e-8)


def test_rigid_objectivity_including_damping():
    s=system();angle=.63
    R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    s.x=s.rest@R.T+[.2,-.3,.1];s.v=np.cross([1.2,.4,-.7],s.x)+[.3,.4,.5]
    f,J=s.internal_forces()
    assert np.max(np.abs(f))<1e-10
    assert np.allclose(J,1)


def test_contact_force_torque_and_friction_balance():
    sphere=Collider(kind='sphere',center=[.035,0,.04],radius=.02,mass=.03,name='ball')
    s=system(colliders=[sphere]);s.v[:]=[.1,.2,.03]
    f,r,t,d,p,n=s.contact_forces()
    assert n>0 and d>0 and p<=0
    assert np.linalg.norm(f.sum(0)+r[0])<1e-12
    assert np.linalg.norm(np.cross(s.x,f).sum(0)+np.cross(sphere.center,r[0])+t[0])<1e-12


def test_isolated_coupled_momentum():
    sphere=Collider(kind='sphere',center=[.049,0,.04],radius=.02,mass=.03,name='ball',stiffness=800)
    s=system(colliders=[sphere]);s.v[:]=[.15,.04,0]
    p=s.momentum();a=s.angular_momentum()
    simulate(s,.03,dt=1e-4)
    assert s.contact_samples>0
    assert np.linalg.norm(s.momentum()-p)<1e-12
    assert np.linalg.norm(s.angular_momentum()-a)<1e-12


def test_unilateral_separation_and_coulomb_bound():
    c=Collider(kind='plane',center=[0,0,.021],friction=.4)
    s=system(colliders=[c]);s.v[:]=[.2,.1,0]
    f,_,_,_,p,_=s.contact_forces()
    assert np.all(f[:,2]>=0)
    assert np.all(np.linalg.norm(f[:,:2],axis=1)<=.4*f[:,2]+1e-12)
    assert p<=0
    c.center[2]=-.1
    assert np.all(s.contact_forces()[0]==0)


def test_box_and_bowl_normals():
    box=Collider(kind='box',half_extents=[1,2,3])
    phi,n=box.query(np.array([[1.2,0,0],[.9,0,0],[0,2.1,0]]))
    assert np.allclose(phi,[.2,-.1,.1]) and np.allclose(n,[[1,0,0],[1,0,0],[0,1,0]])
    bowl=Collider(kind='bowl',radius=1)
    phi,n=bowl.query(np.array([[0,0,-1.1],[0,0,-.9],[0,0,1.1]]))
    assert np.allclose(phi[:2],[-.1,.1]) and np.isinf(phi[2])
    assert np.allclose(n[0],[0,0,1])


def test_timestep_refinement_for_squeeze():
    finals=[]
    for dt in (4e-4,2e-4,1e-4):
        a=Collider(kind='plane',center=[-.031,0,0],normal=[1,0,0],name='left',stiffness=1200)
        b=Collider(kind='plane',center=[.031,0,0],normal=[-1,0,0],name='right',stiffness=1200)
        s=system(colliders=[a,b],material=Material(young=6000,viscosity=2))
        def trajectory(t):return {'left':{'center':[-.031+.025*t,0,0],'velocity':[.025,0,0]},'right':{'center':[.031-.025*t,0,0],'velocity':[-.025,0,0]}}
        out=simulate(s,.12,dt=dt,trajectory=trajectory)
        finals.append(s.x.copy())
        assert out['min_j']>.8 and out['max_penetration_m']<.001
    coarse=np.linalg.norm(finals[0]-finals[2]);medium=np.linalg.norm(finals[1]-finals[2])
    assert medium<.7*coarse
    assert np.sqrt(np.mean((finals[-1]-s.rest)**2))>.0004


def test_invalid_state_fails_explicitly():
    s=system();s.x[:,0]*=-1
    with pytest.raises(FloatingPointError):s.step(1e-4)


def test_dynamic_sphere_static_floor_reaction():
    ball=Collider(kind='sphere',center=[.2,0,.019],radius=.02,mass=.03,name='ball')
    floor=Collider(kind='plane',name='floor')
    s=system(colliders=[ball,floor]);ball.velocity[:]=[.1,0,-.1]
    f,t,d,p,n=s.rigid_contact_forces()
    assert n==1 and d>0 and p<0
    assert np.linalg.norm(f.sum(0))<1e-12 and f[0,2]>0
    angular=sum((np.cross(c.center,fi)+ti for c,fi,ti in zip(s.colliders,f,t)))
    assert np.linalg.norm(angular)<1e-12


def test_dynamic_sphere_pair_reaction():
    a=Collider(kind='sphere',center=[.2,0,.04],radius=.02,mass=.03,name='a')
    b=Collider(kind='sphere',center=[.239,0,.04],radius=.02,mass=.02,name='b')
    s=system(colliders=[a,b]);a.velocity[:]=[.2,.1,0]
    f,t,d,p,n=s.rigid_contact_forces()
    assert n==1 and d>0 and p<0
    assert np.linalg.norm(f.sum(0))<1e-12
    angular=sum((np.cross(c.center,fi)+ti for c,fi,ti in zip(s.colliders,f,t)))
    assert np.linalg.norm(angular)<1e-12


def test_boundary_area_is_resolution_independent_and_interior_excluded():
    for cells in ((2,2,2),(4,4,4)):
        x,t=make_block(size=(.08,.06,.05),cells=cells,center=(0,0,.04))
        s=FEMSystem(x,t,colliders=[Collider(kind='plane',center=(0,0,.1))])
        assert np.isclose(s.surface_area.sum(),2*(.08*.06+.08*.05+.06*.05))
        assert np.isclose(s.contact_area_weights.sum(),s.surface_area.sum()/.0004)
        assert np.any(~s.boundary_nodes)
        f,*_=s.contact_forces()
        assert np.all(f[~s.boundary_nodes]==0)


def test_friction_saturation_is_logged():
    floor=Collider(kind='plane',center=(0,0,.021),friction_speed=.01)
    s=system(colliders=[floor]);s.v[:]=[.1,0,0]
    out=simulate(s,.0001,dt=.0001,fps=10000)
    assert out['friction_active_samples']>0
    assert out['friction_saturated_samples']==out['friction_active_samples']
    assert out['friction_saturated_fraction']==1
