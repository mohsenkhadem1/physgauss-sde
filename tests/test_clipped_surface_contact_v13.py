import numpy as np
from scripts.coupled_contact_v13.physics import make_block,Material,Collider,FEMSystem
from scripts.coupled_contact_v13.clipped_surface_contact import integrate,ClippedSurfaceFEMSystem,load_kernel


def block(n=2,center=(0,0,.025)):
    x,t=make_block((.08,.06,.05),(n,n,n),center)
    return FEMSystem(x,t,Material(),gravity=(0,0,0))


def test_plane_integral_matches_analytic_boundary_pressure_and_mesh_invariance():
    c=Collider(kind='plane',center=(0,0,.001),stiffness=4800,friction=0,damping_ratio=0)
    expected=4800/.0004*(.08*.06*.001+2*(.08+.06)*.001**2/2)
    for n in (2,3,4):
        s=block(n);f,r,t,stats=integrate(s,c)
        assert np.isclose(f[:,2].sum(),expected,rtol=1e-12)
        assert np.allclose(f.sum(0)+r,0,atol=1e-11)
        assert np.allclose(np.cross(s.x,f).sum(0)+np.cross(c.center,r)+t,0,atol=1e-11)


def test_box_normal_force_is_exact_clipped_energy_derivative():
    s=block(2,(.018,0,.074));c=Collider(kind='box',center=(.01,0,.025),half_extents=(.008,.07,.025),stiffness=4800,friction=0,damping_ratio=0)
    rng=np.random.default_rng(82);s.x+=rng.normal(0,.00003,s.x.shape);direction=rng.normal(size=s.x.shape)
    f,r,t,stats=integrate(s,c);initial=s.x.copy();h=1e-8
    s.x=initial+h*direction;plus=integrate(s,c)[3][8]
    s.x=initial-h*direction;minus=integrate(s,c)[3][8];s.x=initial
    assert np.isclose((plus-minus)/(2*h),-np.sum(f*direction),rtol=2e-6,atol=1e-7)
    assert stats[0]>0 and stats[8]>0


def test_box_normal_and_friction_conserve_reaction_torque_and_dissipate():
    s=block(3,(.018,0,.074));s.v[:]=[.05,.03,-.02]
    c=Collider(kind='box',center=(.01,0,.025),half_extents=(.008,.07,.025),stiffness=4800,friction=.4)
    f,r,t,stats=integrate(s,c,1)
    assert np.allclose(f.sum(0)+r,0,atol=1e-10)
    assert np.allclose(np.cross(s.x,f).sum(0)+np.cross(c.center,r)+t,0,atol=1e-10)
    assert stats[1]<=0 and stats[3]>0


def test_penetration_recorded_before_nonadhesive_force_clipping():
    s=block();s.v[:,2]=100
    c=Collider(kind='plane',center=(0,0,.001),friction=0,damping_ratio=1)
    f,_,_,stats=integrate(s,c)
    assert stats[0]>.0009 and stats[8]>0
    assert np.max(np.abs(f))<1e-12


def test_friction_quadrature_refines_without_changing_exact_normal_force():
    s=block(2,(.018,0,.074));s.v[:,0]=.02*(s.x[:,1]/.03);s.v[:,1]=.008+.01*(s.x[:,0]/.04)
    c=Collider(kind='box',center=(.01,0,.025),half_extents=(.008,.07,.025),stiffness=4800,friction=.4,damping_ratio=0)
    rows=[integrate(s,c,k) for k in (0,1,2)]
    assert all(r[3][1]<=0 for r in rows)
    assert np.linalg.norm(rows[1][0]-rows[2][0])<np.linalg.norm(rows[0][0]-rows[2][0])
    assert all(np.isclose(r[3][8],rows[0][3][8]) for r in rows)


def test_wrapper_exposes_compiled_provenance_and_uses_new_contact():
    x,t=make_block((.08,.06,.05),(2,2,2),(0,0,.025))
    s=ClippedSurfaceFEMSystem(x,t,Material(),[Collider(kind='plane',center=(0,0,.001))],gravity=(0,0,0))
    assert s.contact_forces()[-1]>0
    _,metadata=load_kernel();assert metadata['source_sha256'] and metadata['library_sha256']
