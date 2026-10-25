import numpy as np
import pytest
from experiments import campaign_active_fixture_bank_v01 as m


def test_four_fixture_gram_normalization_firstmoment_PSD_and_energy_gradient():
    X,_,faces,_,_=m.core.geometry(1)
    for spec in m.LAYOUTS:
        W,d=m.consistent_fixture(X,faces,spec['id']);ref,dr=m.consistent_fixture(X,faces,spec['id'],order=20)
        assert d['normalization_error']<1e-10 and d['first_moment_error_m']<1e-10
        assert np.linalg.norm((W-ref).toarray())/np.linalg.norm(ref.toarray())<1e-10
        assert np.linalg.eigvalsh(W.toarray()).min()>-1e-14
        u=np.sin(np.arange(X.size).reshape(X.shape))*.001;v=np.cos(np.arange(X.size).reshape(X.shape))
        K=12.;energy=lambda a:.5*K*np.sum(a*(W@a))
        eps=1e-7;fd=(energy(u+eps*v)-energy(u-eps*v))/(2*eps)
        assert abs(fd-K*np.sum(v*(W@u)))<1e-12
        R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        assert abs(energy(u@R)-energy(u))<1e-20
        assert np.allclose(W.toarray(),W.T.toarray(),rtol=0,atol=1e-16)


def test_all_controls_same_force_budget_and_exact_reference_wrench():
    catalog,info=m.load_catalog(1)
    assert len(catalog)==11 and len(m.load_specs()[1:9])==8
    for spec in m.load_specs():
        d=info[spec['id']]
        assert d['normalization_error']<1e-12
        assert d['net_force_error_N']<1e-12 and d['moment_error_Nm']<1e-13
        if spec['id'].startswith('probe'):assert spec['force_N']==.08
        f2,d2=m.surface_patch(1,spec,order=20)
        assert np.max(abs(f2-catalog[spec['id']]))<1e-12
    moments=np.array([info[s['id']]['moment_Nm']for s in m.load_specs()[1:9]])
    assert np.any(abs(moments[:,0])>1e-4)and np.any(abs(moments[:,1])>1e-4)and np.any(abs(moments[:,2])>1e-4)


def test_measurement_projection_affine_reproduction_and_layout_independent_controls():
    X,_,_,_,_=m.core.geometry(1);A=np.array([[.01,.02,.03],[-.01,.01,.02],[.02,-.01,.03]])
    u=X@A.T+np.array([.001,-.002,.003]);fields={s['id']:np.stack([u,2*u])for s in m.load_specs()}
    specs=m.measurement_specs();answer=m.measurement_response(1,fields)
    assert answer.shape==(2,60,1)
    for j,s in enumerate(specs):
        expected=(np.array(s['point_m'])@A.T+np.array([.001,-.002,.003]))@s['projection']
        np.testing.assert_allclose(answer[:,j,0],[expected,2*expected],atol=2e-17,rtol=0)
    assert [s['role']for s in specs].count('prefix')==4
    assert [s['role']for s in specs].count('candidate')==8
    assert [s['role']for s in specs].count('target')==48
    assert all(abs(np.linalg.norm(s['projection'])-1)<1e-14 for s in specs)
    assert 'identity'not in m.load_catalog.__code__.co_varnames


def test_nonrigid_metric_rejects_rigid_motion_and_preserves_compatible_strain():
    f=m.make_forward(1,m.LAYOUTS[0]['id'])
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    rigid=f.X@R-f.X+np.array([1.,2.,3.]);strain=.01*f.X
    values=m.nonrigid_rms(f,np.stack([rigid,strain]))
    assert values[0]<1e-14 and values[1]>1e-4
    assert np.all(f.X[f.fixed,0]==np.where(f.X[f.fixed,0]==0,0,.36))
    with pytest.raises(ValueError):m.layout_spec('oracle-layout')
