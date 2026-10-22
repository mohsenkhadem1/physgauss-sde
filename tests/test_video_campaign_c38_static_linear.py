"""Manufactured display topology and completion-gate controls only."""
import json
import numpy as np
import pytest
from experiments import video_campaign_c38_static_linear as m
from experiments.campaign_static_p2_block import enrich


def test_quadratic_boundary_midpoints_and_display_triangles_preserve_affine_geometry():
    corners=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    X,e=enrich(corners,np.array([[0,1,2,3]]));f=np.array([[0,1,2],[0,3,1]])
    f6=m.boundary6(e,f);q=X@np.array([[1,.1,0],[0,1.2,0],[0,0,.9]])+[.2,.3,-.4]
    surface=m.display_surface(q,f6)
    assert surface.shape==(8,3,3)
    np.testing.assert_array_equal(surface[0],q[f6[0,[0,3,5]]])
    np.testing.assert_allclose(X[f6[:,3:]],np.stack([(X[f[:,0]]+X[f[:,1]])/2,(X[f[:,1]]+X[f[:,2]])/2,(X[f[:,2]]+X[f[:,0]])/2],1))
    conflict=np.concatenate([e,e]);conflict[1,4]=conflict[1,5]
    with pytest.raises(ValueError,match='shared P2'):m.boundary6(conflict,f)


def test_static_state_schedule_has_only_held_acquisitions_not_fake_time():
    schedule=m.schedule();assert len(schedule)==180
    assert np.array_equal(schedule,np.repeat([0,1,2],60))
    with pytest.raises(ValueError):m.schedule(1.5)


def test_incomplete_comparison_rejected_before_any_array_load(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);path=tmp_path/m.SCORE_SUMMARY;path.parent.mkdir(parents=True);path.write_text(json.dumps({'completed':False,'rows':[]}))
    monkeypatch.setattr(m.score,'validate',lambda *a,**kw:{'pairs':[]})
    monkeypatch.setattr(np,'load',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Physical arrays read before completion gate')))
    with pytest.raises(ValueError,match='Complete sealed28'):m.load_inputs()
