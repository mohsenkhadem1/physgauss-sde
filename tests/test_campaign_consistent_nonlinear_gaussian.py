import numpy as np
import pytest
from experiments.campaign_consistent_nonlinear_gaussian import (
    ConsistentSurfaceNonlinearGaussian as Model,canonical_load_definition,tension_command)


def test_quadratic_fixture_gradient_tangent_and_dissipation():
    m=Model.from_design(level=0);rng=np.random.default_rng(8)
    x=m.rest+rng.normal(size=m.rest.shape)*2e-4;v=rng.normal(size=x.shape)*.003
    target=np.broadcast_to(m.rest,(2,*x.shape)).copy()+rng.normal(size=(2,*x.shape))*1e-4
    tv=rng.normal(size=target.shape)*.002;p=rng.normal(size=x.shape);p/=np.linalg.norm(p)
    Us,Fs,diss=m.fixture(x,v,target,tv)
    eps=1e-7
    fd=(m.fixture(x+eps*p,v,target,tv)[0]-m.fixture(x-eps*p,v,target,tv)[0])/(2*eps)
    spring=m.fixture(x,np.zeros_like(v),target,np.zeros_like(tv))[1].sum(0)
    assert fd==pytest.approx(-np.sum(spring*p),rel=1e-8,abs=1e-12)
    fp=m.fixture(x+eps*p,v,target,tv)[1].sum(0);fm=m.fixture(x-eps*p,v,target,tv)[1].sum(0)
    np.testing.assert_allclose((fp-fm)/(2*eps),-(m.K@p.ravel()).reshape(x.shape),rtol=1e-8,atol=1e-10)
    assert Us>=0 and diss>=0
    for A in [*m.Kparts,*m.Cparts]:
        np.testing.assert_allclose(A.toarray(),A.toarray().T,atol=1e-14)
        assert np.linalg.eigvalsh(A.toarray()).min()>-1e-13
    # Full Gram has off-node blocks, so this is not diagonal row-sum lumping.
    assert m.fixture_operators.fixture_grams[0].nnz>len(m.rest)


def test_incremental_objective_gradient_and_hessian():
    m=Model.from_design();rng=np.random.default_rng(14);h=.013
    x=m.rest+rng.normal(size=m.rest.shape)*1e-4;v=rng.normal(size=x.shape)*.001
    old=np.broadcast_to(m.rest,(2,*x.shape)).copy();new=old+rng.normal(size=old.shape)*2e-5
    tv=(new-old)/h;force=m.known_force(np.array([[.02,.01,0],[-.01,.005,0]]))
    dx=rng.normal(size=x.shape)*1e-5;p=rng.normal(size=x.shape);p/=np.linalg.norm(p)
    def calc(d):
        U,g,H,_=m.elastic(x+d,tangent=True);Us,Fs,_=m.fixture(x+d,d/h,new,tv)
        val=.5*np.sum(m.mass*(d-h*v).ravel()**2)/h**2+U+Us+.5/h*m.patch_quadratic(d[None]-h*tv,m.Cparts)-np.sum(force*d)
        r=(m.mass*((d/h-v)/h).ravel()).reshape(x.shape)+g-Fs.sum(0)-force
        A=H+np.diag(m.mass/h**2)+m.K.toarray()+m.C.toarray()/h
        return val,r,np.asarray(A)
    eps=2e-8;vp,rp,_=calc(dx+eps*p);vm,rm,_=calc(dx-eps*p);_,r,H=calc(dx)
    assert (vp-vm)/(2*eps)==pytest.approx(np.sum(r*p),rel=1e-7,abs=2e-10)
    np.testing.assert_allclose((rp-rm)/(2*eps),(H@p.ravel()).reshape(x.shape),rtol=1e-7,atol=1e-6)


def test_load_normalization_rest_centrality_and_mesh_moments():
    reference=canonical_load_definition();coarse=Model.from_design(level=0);fine=Model.from_design(level=2)
    for m in [coarse,fine]:
        H2=tension_command(.2,1.,.4,axis=m.load_axis_world);F=m.known_force(H2)
        np.testing.assert_allclose(m.load_vectors.sum(1),1,atol=3e-16)
        np.testing.assert_allclose(F.sum(0),0,atol=5e-16)
        np.testing.assert_allclose(m.load_vectors@m.rest,reference['centroids_material_m'],atol=2e-12)
        assert np.linalg.norm(m.load_torque(m.rest,H2))<3e-12
    # The dead load must report the actual torque in a deformed configuration.
    x=coarse.rest.copy();x[:,0]*=1.1
    assert np.linalg.norm(coarse.load_torque(x,H2))>1e-5
    assert np.all(tension_command(.4,1.,.4)==0)


def test_full_step_moving_target_stationarity_energy_and_patch_work():
    m=Model.from_design(level=0);x=m.rest.copy();v=np.zeros_like(x);h=1/240
    old=np.broadcast_to(x,(2,*x.shape)).copy();new=old.copy()
    # Nonuniform moving patch target exercises consistent off-node operators.
    new[0]+=np.array([2e-5,-1e-5,3e-5])+.0001*(x-x.mean(0))
    new[1]+=np.array([-1e-5,2e-5,0])
    force=m.known_force(tension_command(.1,1.,.4,axis=m.load_axis_world))
    q,w,L=m.step(x,v,old,new,force,h,relative_tolerance=1e-8,maximum_iterations=50)
    U,g,_,J=m.elastic(q);Us,Fs,d=m.fixture(q,w,new,(new-old)/h)
    residual=(m.mass*((w-v)/h).ravel()).reshape(x.shape)+g-Fs.sum(0)-force
    assert np.linalg.norm(residual)<=L['threshold_N']
    assert J.min()>.99 and L['iterations']<10
    assert L['target_work_J']==pytest.approx(np.sum(Fs*(new-old)),abs=1e-20)
    assert L['damping_dissipation_J']==pytest.approx(h*d,abs=1e-20)
    assert abs(L['balance_error_J']-L['free_residual_work_J'])<5e-15
    # Independently reconstruct the complete accepted-state energy ledger.
    delta=.5*np.sum(m.mass*(w.ravel()**2-v.ravel()**2))+U+Us-m.elastic(x)[0]-m.fixture(x,v,old,np.zeros_like(old))[0]
    de=(q-x)[None]-(new-old)
    numerical=.5*np.sum(m.mass*(w-v).ravel()**2)+.5*m.patch_quadratic(de,m.Kparts)+np.sum(g*(q-x))-(U-m.elastic(x)[0])
    assert abs(delta-np.sum(force*(q-x))-np.sum(Fs*(new-old))+h*d+numerical)<5e-15


def test_passive_frame_covariance_all_physical_fields_and_step():
    ang=.47;Q=np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1.]])
    shift=np.array([.31,-.21,.16]);a=Model.from_design();b=Model.from_design(frame_rotation=Q,frame_translation=shift)
    rng=np.random.default_rng(22);x=a.rest+rng.normal(size=a.rest.shape)*1e-4;v=rng.normal(size=x.shape)*.001
    old=np.broadcast_to(a.rest,(2,*x.shape)).copy();new=old+np.array([[[2e-5,0,-1e-5]],[[-1e-5,3e-5,0]]])
    F=tension_command(.05,.1,.4,axis=a.load_axis_world);force=a.known_force(F)
    rot=lambda y:y@Q.T;pos=lambda y:rot(y)+shift
    ua,fa,da=a.fixture(x,v,new,(new-old)*240);ub,fb,db=b.fixture(pos(x),rot(v),pos(new),rot((new-old)*240))
    assert ub==pytest.approx(ua,rel=1e-11);assert db==pytest.approx(da,rel=1e-11)
    np.testing.assert_allclose(fb,rot(fa),atol=1e-15)
    np.testing.assert_allclose(b.known_force(rot(F)),rot(force),atol=1e-16)
    qa,va,la=a.step(x,v,old,new,force,1/240)
    qb,vb,lb=b.step(pos(x),rot(v),pos(old),pos(new),b.known_force(rot(F)),1/240)
    np.testing.assert_allclose(qb,pos(qa),atol=2e-16,rtol=0)
    np.testing.assert_allclose(vb,rot(va),atol=6e-14,rtol=0)
    assert lb['external_work_J']==pytest.approx(la['external_work_J'],abs=1e-17)


def test_constraints_remain_exact_and_invalid_frames_rejected():
    mask=np.zeros((5,3),dtype=bool);mask[0]=True;m=Model.from_design(constraint_mask=mask)
    target=np.broadcast_to(m.rest,(2,*m.rest.shape)).copy();specified=m.rest.copy();specified[0]+=[1e-6,0,0]
    q,v,L=m.step(m.rest,np.zeros_like(m.rest),target,target,np.zeros_like(m.rest),1/240,constraint_positions=specified)
    np.testing.assert_array_equal(q[0],specified[0]);assert L['constraint_max_error_m']==0
    with pytest.raises(ValueError):Model.from_design(frame_rotation=np.diag([-1,1,1]))
