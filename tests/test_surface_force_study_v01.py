"""Prospective load/sensor controls, no static solve or posterior calls."""
import numpy as np
import pytest
from experiments import campaign_surface_force_study_v01 as s


def test_all_meshes_preserve_physical_resultant_and_expected_moment():
    for n in(2,4,8):
        X,_,_,fixed,_=s.shared.geometry(n)
        catalog=s.load_catalog(n)
        assert len(catalog)==6
        for center in s.CENTERS:
            label=f'x{round(center*1000):03d}';normal=catalog['normal_'+label];axial=catalog['paired_axial_'+label]
            np.testing.assert_allclose(normal.sum(0),[0,0,-.1],atol=1e-12,rtol=0)
            np.testing.assert_allclose(axial.sum(0),[.48,0,0],atol=1e-12,rtol=0)
            np.testing.assert_allclose(np.cross(X-[.18,.04,.04],normal).sum(0),[0,.1*(center-.18),0],atol=1e-13,rtol=0)
            np.testing.assert_allclose(np.cross(X-[.18,.04,.04],axial).sum(0),0,atol=1e-13,rtol=0)
            assert np.count_nonzero(normal[fixed])==0 and np.count_nonzero(axial[fixed])==0
            _,meta=s.surface_patch_force(n,center,[0,0,-1],.1)
            assert abs(meta['patch_area_m2']-.0072)<1e-15 and abs(meta['integrated_weight_mass']-1)<1e-11


def test_central_catalog_exactly_matches_shared_physics_design():
    for n in(2,4,8):
        new=s.load_catalog(n);old=s.shared.qualification_loads(n)
        np.testing.assert_array_equal(new['normal_x180'],old['normal'])
        np.testing.assert_array_equal(new['paired_axial_x180'],old['paired_axial'])


def test_quadrature_and_affine_surface_work_at_shifted_patches():
    X,*_=s.shared.geometry(2);u=X@np.array([[.01,.002,0],[.003,-.004,.001],[0,.001,.005]]).T+[.001,.002,.003]
    for center in s.CENTERS:
        f8,_=s.surface_patch_force(2,center,[0,0,-1],.1,order=8)
        f12,_=s.surface_patch_force(2,center,[0,0,-1],.1,order=12)
        f16,_=s.surface_patch_force(2,center,[0,0,-1],.1,order=16)
        np.testing.assert_allclose(f12,f16,atol=1e-13,rtol=1e-10)
        np.testing.assert_allclose(f8,f12,atol=1e-12,rtol=1e-8)
        assert abs(np.sum((f12-f16)*u))<1e-14


def test_static_sensor_projection_is_affine_exact_and_has_no_clamped_sensors():
    points=s.observation_points();assert points.shape==(8,3)
    assert np.all((points[:,0]>0)&(points[:,0]<.36))
    assert np.all((points[:,2]==.08)|(points[:,1]==.08))
    A=np.array([[.01,.002,0],[.003,-.004,.001],[0,.001,.005]]);b=np.array([.001,.002,.003]);expected=(points@A.T+b).reshape(24)
    for n in(2,4,8):
        X,*_=s.shared.geometry(n);field=X@A.T+b
        np.testing.assert_allclose(s.observe(n,field),expected,atol=3e-17,rtol=0)
        np.testing.assert_allclose(s.observe(n,np.stack([field,2*field])),np.stack([expected,2*expected]),atol=6e-17,rtol=0)
    points[:]=0;assert np.count_nonzero(s.observation_points())>0


def test_prospective_scope_budget_and_fail_closed_inputs():
    d=s.design();assert d['prefix_acquired_load']not in d['held_out_loads']and len(d['held_out_loads'])==5
    assert len(d['probing_strata']['normal_only'])==3 and len(d['probing_strata']['paired_axial_only'])==3
    assert d['sigma_m']==20e-6 and d['no_automatic_execution']
    with pytest.raises(ValueError):s.surface_patch_force(2,.10,[0,0,-1],.1)
    with pytest.raises(ValueError):s.surface_patch_force(2,.09,[0,0,-2],.1)
