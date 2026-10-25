import inspect
import numpy as np
import pytest
from experiments import render_fixture_balanced_nh_comparison_v01 as r

def test_exact_T6_four_triangle_display_area_and_mids():
    X=np.array([[0,0,0],[2,0,0],[0,2,0],[1,0,0],[0,1,0],[1,1,0]],float);tri=r.surface_triangles(np.array([[0,1,2,3,4,5]]));area=np.linalg.norm(np.cross(X[tri[:,1]]-X[tri[:,0]],X[tri[:,2]]-X[tri[:,0]]),axis=1)/2
    assert tri.shape==(4,3)and np.all(area==.5)and area.sum()==2.
    with pytest.raises(ValueError):r.surface_triangles(np.zeros((1,3),int))

def test_negative_audited_preset_is_prominently_marked_not_filtered():
    assert 'FAILED PRESET GATES'in r.status_label(False)
    assert 'PASSED'in r.status_label(True)
    assert r.STYLE['frames']==2*r.STYLE['frames_per_load']==96 and r.STYLE['frames']/r.STYLE['fps']==8

def test_root_release_before_any_geometry_or_arrays(monkeypatch):
    def refuse(digest):raise ValueError('No root release')
    monkeypatch.setattr(r,'validate',refuse)
    monkeypatch.setattr(r.np,'load',lambda *a,**k:pytest.fail('Arrays read before release'))
    with pytest.raises(ValueError,match='release'):r.render('toy')
    source=inspect.getsource(r.render);assert source.index('validate(digest)')<source.index('resolution.geometry(6)')<source.index('np.load')

def test_recipe_is_static_orbit_true_parameter_oracle_no_displacement_ramp():
    source=inspect.getsource(r.render)
    assert "if local==0:"in source and 'X+20*preset' in source
    assert 'local/47' in source and 'displacement ×20' in source
    assert 'LINEAR ORACLE — same true parameters' in source and 'NONLINEAR REFERENCE (NH80)'in source
    assert 'range(96)'in source and "count in(0,48)"in source
