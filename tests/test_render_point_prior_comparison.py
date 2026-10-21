import json
from unittest.mock import patch
import numpy as np
import pytest
from experiments import render_point_prior_comparison as renderer


def test_missing_or_incomplete_aggregate_cannot_open_any_array(tmp_path):
    with patch.object(renderer,'ROOT',tmp_path),patch.object(renderer.np,'load',side_effect=AssertionError('Array access before seal')):
        with pytest.raises(FileNotFoundError):renderer.gate('absent.json','bad')
        seal=tmp_path/'bad.json';seal.write_text(json.dumps(dict(schema='point-prior-reserved-view-evaluation-v0.1',completed=False,objects=32)))
        with pytest.raises(ValueError,match='Final complete32 reserved aggregate'):renderer.gate('bad.json',renderer.sha(seal))


def test_synthetic_three_panel_video_shared_scale_and_verified_encoding(tmp_path):
    rest=np.array([[0.,0.,0.],[.2,0.,0.],[0.,.1,0.],[0.,0.,.1]])
    faces=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
    times=np.array([0.,.5,1.,1.2,1.55,3.]);actual=np.repeat(rest[None],len(times),axis=0)
    actual[:,1,2]=.004*np.sin(np.pi*times/3)
    predicted=rest+.85*(actual-rest);active=rest+.95*(actual-rest)
    result=renderer.render_sequence(rest,faces,times,[actual,predicted,active],tmp_path/'synthetic.mp4',caption='SYNTHETIC TEST',fps=5)
    assert result['frames']==6 and result['bytes']>1000
    assert result['same_camera_scale_colors'] and result['motion_magnification']==1.
    assert result['maximum_vertices_outside_fixed_bounds']==[0,0,0]
    assert (tmp_path/'synthetic.first-frame.png').exists()
    with pytest.raises(ValueError,match='Fresh LOCAL'):renderer.render_sequence(rest,faces,times,[actual,predicted,active],tmp_path/'synthetic.mp4')
