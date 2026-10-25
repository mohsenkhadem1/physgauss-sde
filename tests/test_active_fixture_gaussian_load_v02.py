import numpy as np
import pytest
from scipy.integrate import quad
from experiments import campaign_active_fixture_gaussian_load_v02 as m


def test_analytic_rectangle_normalizer_centroid_and_reflections_independent_quadrature():
    for c in ([.09,.02],[.09,.06],[.27,.02],[.27,.06],[.18,.04]):
        G,center=m.gaussian_integrals(c);integrals=[];means=[]
        for a,(length,std) in enumerate(zip((.36,.08),m.LOAD_STD_M)):
            fn=lambda x:np.exp(-.5*((x-c[a])/std)**2)
            I=quad(fn,0,length,epsabs=1e-14,epsrel=1e-13)[0];integrals.append(I);means.append(quad(lambda x:x*fn(x),0,length,epsabs=1e-14,epsrel=1e-13)[0]/I)
        np.testing.assert_allclose(G,np.prod(integrals),atol=2e-18,rtol=2e-14)
        np.testing.assert_allclose(center,[*means,.08],atol=2e-16,rtol=0)
    G,c=m.gaussian_integrals([.09,.02]);H,d=m.gaussian_integrals([.27,.06])
    np.testing.assert_allclose(G,H,atol=2e-18,rtol=2e-14)
    np.testing.assert_allclose(c+d,[.36,.08,.16],atol=1e-16,rtol=0)
    assert c[1]>.02 and c[0]>.09


def test_every_load_exact_common_budget_firstmoment_and16_vs20_n1():
    f,info=m.load_catalog(1);ref,_=m.load_catalog(1,order=20)
    assert len(f)==11
    for spec in m.load_specs():
        d=info[spec['id']];expected=.04 if spec['id']=='prefix-center-down' else .06
        assert spec['force_N']==expected and d['equal_candidate_continuum_traction_norm_N']==expected
        assert d['normalization_error']<1e-10 and d['first_moment_error_m']<1e-10
        assert d['net_force_error_N']<1e-12 and d['moment_error_Nm']<1e-13
        np.testing.assert_allclose(f[spec['id']],ref[spec['id']],atol=1e-12,rtol=0)
    # Authored center is deliberately NOT the true moment arm near boundaries.
    spec=m.load_specs()[1];d=info[spec['id']]
    center_arm=np.cross(np.array(spec['center_m'])-m.ORIGIN_M,spec['force_N']*np.array(spec['direction']))
    assert np.linalg.norm(center_arm-np.array(d['analytic_moment_Nm']))>1e-5


def test_supports_geometry_and_observation_ABI_unchanged_no_old_globals_mutated():
    assert m.LAYOUTS==m.original.LAYOUTS and m.measurement_specs()==m.original.measurement_specs()
    before=m.original.design();new=m.design()
    assert before==m.original.design() and before['loads'][0]['force_N']==.02
    assert 'patch_width_m' not in new and new['support_bank_changed'] is False
    a=m.make_forward(1,m.LAYOUTS[0]['id']);b=m.original.make_forward(1,m.LAYOUTS[0]['id'])
    np.testing.assert_array_equal(a.X,b.X);np.testing.assert_array_equal(a.W.data,b.W.data)
    np.testing.assert_array_equal(a.Kunit.data,b.Kunit.data)
    for a,b in zip(m.load_specs(),m.original.load_specs()):
        for key in ('id','center_m','direction'):assert a[key]==b[key]


def test_invalid_public_load_parameters_fail_closed():
    spec=m.load_specs()[0]
    for change in ({'force_N':-1},{'direction':[0,0,2]},{'center_m':[.18,.04,0]}):
        with pytest.raises(ValueError):m.surface_patch(1,dict(spec,**change))
    with pytest.raises(ValueError):m.gaussian_integrals([.09,.02],[-.01,.02])
