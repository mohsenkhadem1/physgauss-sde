import numpy as np
from scripts.coupled_contact_v13.physics import make_block,Material,Collider
from scripts.coupled_contact_v13.surface_contact import SurfaceFEMSystem
from scripts.coupled_contact_v13.sparse_surface_contact import SparseSurfaceFEMSystem


def test_sparse_force_matches_exhaustive_on_deformed_moving_surface():
    class Exhaustive(SurfaceFEMSystem):surface_spacing=.0015
    x,t=make_block((.08,.06,.05),(3,3,3),(0,0,.025))
    def colliders():return [Collider(kind='plane',center=(0,0,.002),name='floor'),Collider(kind='box',center=(.02,0,.02),half_extents=(.008,.07,.025),name='edge'),Collider(kind='sphere',center=(-.038,0,.025),radius=.02,mass=.04,name='ball'),Collider(kind='bowl',center=(0,0,.06),radius=.05,name='bowl')]
    a=Exhaustive(x,t,Material(),colliders());b=SparseSurfaceFEMSystem(x,t,Material(),colliders())
    rng=np.random.default_rng(481)
    for _ in range(3):
        angle=rng.uniform(-.4,.4);R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        state=(x-x.mean(0))@R.T+x.mean(0)+rng.normal(0,.001,x.shape);velocity=rng.normal(0,.1,x.shape)
        a.x[:]=b.x[:]=state;a.v[:]=b.v[:]=velocity
        aa=a.contact_forces();bb=b.contact_forces()
        for i in range(5):assert np.allclose(aa[i],bb[i],rtol=1e-11,atol=1e-11)
        assert aa[5]==bb[5]
        assert a._last_friction_active==b._last_friction_active
        assert a._last_friction_saturated==b._last_friction_saturated


def test_culling_keeps_all_contacting_fine_samples():
    x,t=make_block((.08,.06,.05),(2,2,2),(0,0,.025));c=Collider(kind='sphere',center=(.045,0,.03),radius=.02)
    s=SparseSurfaceFEMSystem(x,t,Material(),[c]);s.x[:,0]*=1.4;s.x[:,1]*=.8
    center=np.einsum('qi,qic->qc',s.broadphase_weights,s.x[s.broadphase_ids]);bound=np.linalg.norm(s.deformation(),axis=(1,2)).max()
    candidates=s.candidate_indices(c,center,bound);points,_=s.surface_state();active=np.flatnonzero(c.query(points)[0]<0)
    assert len(candidates)<len(points) and len(active)>0
    assert np.isin(active,candidates).all()
