"""Prospective extension controls only: geometry/operators and tiny synthetic data."""
import hashlib
from pathlib import Path
import numpy as np
from experiments import campaign_surface_p2_static_v01 as old
from experiments import campaign_surface_p2_static_v02 as new


def test_existing_geometry_operators_and_normal_are_byte_identical():
    for n in (2,4):
        for a,b in zip(old.geometry(n),new.geometry(n)):
            assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
        a=old.SparseStaticForward(n);b=new.SparseStaticForward(n)
        for key in ('Kunit','W_scalar','W','Kunit_free','W_free'):
            x=getattr(a,key);y=getattr(b,key)
            for value in ('data','indices','indptr'):
                assert getattr(x,value).tobytes()==getattr(y,value).tobytes()
        original=old.qualification_loads(n);revised=new.qualification_loads(n)
        assert original['normal'].tobytes()==revised['normal'].tobytes()
        np.testing.assert_allclose(revised['paired_axial'],.75*original['paired_axial'],atol=2e-18,rtol=2e-15)


def test_new_n8_geometry_boundary_volume_and_load_moments():
    X,T,F,fixed,free=new.geometry(8);t=new._topology(8)
    assert X.shape==(18785,3) and T.shape==(12288,10) and len(free)==54621
    assert np.all((X[fixed,0]==0)|(X[fixed,0]==.36))
    corners=t['corner_X'][t['corner_tets']]
    volumes=np.linalg.det((corners[:,1:]-corners[:,:1]).transpose(0,2,1))/6
    assert np.all(volumes>0);np.testing.assert_allclose(volumes.sum(),.002304,atol=1e-16)
    vertices=X[F[:,:3]]
    external=np.any(np.all(vertices==[0,0,0],axis=1)|np.all(vertices==[.36,.08,.08],axis=1),axis=1)
    assert np.all(external)
    for name,force in new.qualification_loads(8).items():
        expected=[0,0,-.1] if name=='normal' else [.36,0,0]
        np.testing.assert_allclose(force.sum(0),expected,atol=2e-13,rtol=0)
        np.testing.assert_allclose(np.cross(X-[.18,.04,.04],force).sum(0),0,atol=2e-14,rtol=0)
        assert not np.any(force[fixed])


def test_n8_quadratic_queries_and_random_nested_prolongation():
    rng=np.random.default_rng(2026102491);points=rng.uniform(size=(257,3))*[.36,.08,.08]
    X=new.geometry(8)[0]
    polynomial=lambda x:np.column_stack([x[:,0]**2+x[:,1]*x[:,2],x[:,1]**2+x[:,0]*x[:,2],x[:,2]**2+x[:,0]*x[:,1]])
    np.testing.assert_allclose(new.field_queries(8,polynomial(X),points),polynomial(points),atol=1e-15,rtol=1e-13)
    coarse=new.geometry(4)[0];u=rng.normal(size=coarse.shape)
    prolonged=new.field_queries(4,u,X)
    np.testing.assert_allclose(new.field_queries(8,prolonged,points),new.field_queries(4,u,points),atol=1e-12,rtol=1e-12)
    volume=0.;square=np.zeros(3)
    for q,w in new.bulk_query_blocks():
        volume+=w.sum();square+=w@(q*q)
    np.testing.assert_allclose(volume,.002304,atol=1e-16)
    np.testing.assert_allclose(square/volume,np.array([.36,.08,.08])**2/3,atol=1e-15)


def test_prospective_design_keeps_gates_and_versions_separate():
    d=new.design();assert d['qualification_levels']==[2,4,8] and d['target_mesh_n']==4
    assert d['design_tuned_on_v01'] and d['no_automatic_execution']
    assert new.GATES==old.GATES
    assert old.design()['schema']=='surface-p2-static-v01'
