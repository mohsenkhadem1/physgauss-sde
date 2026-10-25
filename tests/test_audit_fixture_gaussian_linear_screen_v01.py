import json
import numpy as np
import pytest
from experiments import audit_fixture_gaussian_linear_screen_v01 as m


def test_terminal_failure_blocks_before_result_or_npz_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(m.stage,'LOCAL',tmp_path);monkeypatch.setattr(m.stage,'validate',lambda digest:{})
    (tmp_path/'protocol.json').write_text('{}');(tmp_path/'execution.json').write_text(json.dumps(dict(terminal=True,completed=False)))
    monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('No array read allowed'))
    with pytest.raises(ValueError,match='process barrier'):m.metadata_barrier()
    assert not (tmp_path/'result-LOCAL.json').exists()


def test_all_state_support_separation_and_mesh_summary_noise_units_no_category_drop():
    raw=[];rows=[]
    for q in m.stage.mechanics.worklist():
        category=next(i for i,s in enumerate(m.stage.mechanics.bank.LAYOUTS) if s['id']==q['layout'])
        raw.append(dict(q,response=np.full(60,category*20e-6),field_queries=np.full((11,16,3),q['n']*1e-6)))
        rows.append(dict(q,nonrigid_coordinate_RMS_m=[1e-4]*11,maximum_linear_symmetric_strain=[.02]*11,minimum_sampled_linear_J=[.99]*11))
    separation,pairs,summary=m.summarize(raw,rows)
    assert len(separation)==108 and len(pairs)==36
    assert summary['6']['states']==36 and summary['6']['held48_layout_RMS_range_m']==[20e-6,60e-6]
    np.testing.assert_allclose(pairs[0]['maximum_common_point_coordinate_difference_m'],3e-6,atol=1e-21,rtol=0)
    with pytest.raises(ValueError,match='four categories'):m.summarize(raw[:-1],rows[:-1])
