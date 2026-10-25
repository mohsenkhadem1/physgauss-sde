import copy
import numpy as np
from scipy.sparse.linalg import splu
from experiments import campaign_fixture_balanced_held_v01 as m


def test_new_ids_known_norms_and_exact_first12_channel_and16point_ABI():
    old=m.bank.design();new=m.design();assert m.bank.design()==old
    assert m.load_specs()[:9]==m.bank.load_specs()[:9]
    assert m.measurement_specs()[:12]==m.bank.measurement_specs()[:12]
    np.testing.assert_array_equal(m.measurement_points(),m.bank.measurement_points());assert m.measurement_points().shape==(16,3)
    assert len(m.labels()['prefix'])==4 and len(m.labels()['candidate'])==8 and len(m.labels()['target'])==48
    assert all(s['load'].startswith('held-oblique-') for s in m.measurement_specs()[12:])
    assert not any('held-offaxis' in s['id'] for s in m.measurement_specs()[12:])
    for spec in m.load_specs()[9:]:
        assert spec['force_N']==.06
        np.testing.assert_allclose(np.linalg.norm(spec['direction']),1.,atol=2e-16,rtol=0)
    assert new['first9loads_first12channels_unchanged'] and new['geometry_supports_material_priors_unchanged']


def test_public180_pairing_centroids_covariance_wrench_and_prefix_asymmetry():
    R=np.diag([-1.,-1.,1.]);center=np.array([.18,.04,0.]);a,b=m.load_specs()[9:]
    np.testing.assert_allclose(center+R@(np.array(a['center_m'])-center),b['center_m'],atol=1e-16,rtol=0)
    np.testing.assert_array_equal(R@np.asarray(a['direction']),b['direction'])
    _,ca=m.bank.gaussian_integrals(a['center_m'][:2]);_,cb=m.bank.gaussian_integrals(b['center_m'][:2])
    np.testing.assert_allclose(center+R@(ca-center),cb,atol=1e-16,rtol=0)
    covariance=np.diag([.035**2,.025**2,0]);np.testing.assert_array_equal(R@covariance@R.T,covariance)
    points=m.bank.original.TARGET_POINTS;rotated=center+(points-center)@R.T
    assert all(np.min(np.linalg.norm(points-p,axis=1))<1e-16 for p in rotated)
    prefix=m.bank.original.PREFIX_POINTS;rotated=center+(prefix-center)@R.T
    assert any(np.min(np.linalg.norm(prefix-p,axis=1))>1e-4 for p in rotated)


def test_all_declared_new_meshes_force16vs20_and_old_first9_byteidentity():
    for n in (3,6,12):
        new,info=m.load_catalog(n);old,_=m.geometry.load_catalog(n);ref,more=m.load_catalog(n,order=20)
        assert len(new)==11
        for spec in m.load_specs():
            k=spec['id'];d=info[k]
            assert d['normalization_error']<1e-10 and d['first_moment_error_m']<1e-10
            assert d['net_force_error_N']<1e-12 and d['moment_error_Nm']<1e-13
            np.testing.assert_allclose(new[k],ref[k],atol=1e-12,rtol=0)
        for k in list(old)[:9]:np.testing.assert_array_equal(new[k],old[k])
        a,b=m.load_specs()[9:];R=np.diag([-1.,-1.,1.])
        np.testing.assert_allclose(R@np.asarray(info[a['id']]['analytic_moment_Nm']),info[b['id']]['analytic_moment_Nm'],atol=2e-18,rtol=0)


def test_manufactured_n1_direct_fullmode_new_targets_and_unchanged_first12_predictions():
    # Small operator/callback control, not actualNEW36heldcases.
    f=m.make_forward(1,m.LAYOUTS[0]['id']);loads,_=m.load_catalog(1);oldloads,_=m.geometry.load_catalog(1)
    model=m.make_measurement_modal(f,loads);oldmodel=m.original.make_measurement_modal(f,oldloads)
    assert model.modal.diagnostics['all_boundary_modes_retained'] and model.modal.diagnostics['exact_particular_1_over_E_term']
    assert model.modal.eigenvalues[-1]==0.
    labels=[s['id'] for s in m.measurement_specs()];oldlabels=[s['id'] for s in m.bank.measurement_specs()]
    for E,K in ((2000.,5.),(4000.,float(np.sqrt(150.))),(8000.,30.)):
        rhs=np.column_stack([v.ravel() for v in loads.values()]);u=np.zeros_like(rhs);u[f.free]=splu(E*f.Kunit_free+K*f.W_free).solve(rhs[f.free]);fields=u.T.reshape(11,len(f.X),3)
        direct=m.measurement_response(1,{label:fields[j:j+1] for j,label in enumerate(loads)})
        unit=np.log([E/2000,K/5])/np.log([4,6]);pred=model.select(labels)(unit[None])
        np.testing.assert_allclose(pred,direct,atol=2e-12,rtol=0)
        np.testing.assert_array_equal(pred[:,:12],oldmodel.select(oldlabels)(unit[None])[:,:12])
