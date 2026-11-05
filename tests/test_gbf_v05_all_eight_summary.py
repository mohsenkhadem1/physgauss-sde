import json
from pathlib import Path
import pytest
from scripts.gaussian_boundary_flow_v02 import phystwin_all_eight_summary as module


def test_missing_cases_remain_in_denominator(tmp_path, monkeypatch):
    score = tmp_path / 'score.json'
    value = {'coordinate_rmse_mm': 2., 'aligned_shape_rmse_mm': 1.,
             'total_frames': 3, 'valid_frames': 3, 'failed_frames': 0, 'no_reference_frames': 0}
    score.write_text(json.dumps({'methods': {'rigid_contact': {'incoming_v22': {'summary': value}}}}))
    def paths(case):
        scores = {channel: tmp_path / 'missing' for channel, _ in module.METHODS.values()}
        if case == module.ORIGINAL[0]: scores['baseline'] = score
        return scores, {}, {}
    monkeypatch.setattr(module, 'REPORT', tmp_path)
    monkeypatch.setattr(module, 'paths', paths)
    monkeypatch.setattr(module, 'validate_channel_identity', lambda *args: {})
    module.summarize()
    report = json.loads((tmp_path / 'phystwin-all-eight-comparison.json').read_text())
    assert len(report['cases']) == 8
    assert report['aggregates']['rigid_contact']['complete_cases'] == 1
    assert report['aggregates']['rigid_contact']['failed_or_missing_cases'] == 7
    assert report['paired_differences']['flow_contact']['rigid_contact']['complete_pairs'] == 0
    assert not report['all_required_methods_complete']


def test_changed_status_bound_score_rejected(tmp_path, monkeypatch):
    score = tmp_path / 'score.json'; score.write_text('{}')
    status = tmp_path / 'status.json'
    status.write_text(json.dumps({'state': 'complete', 'score_sha256': 'wrong'}))
    monkeypatch.setattr(module, 'REPORT', tmp_path)
    monkeypatch.setattr(module, 'paths', lambda case: ({'baseline': score}, {}, {'baseline': status}))
    with pytest.raises(ValueError, match='Changed audited'):
        module.summarize()
    assert not (tmp_path / 'phystwin-all-eight-comparison.json').exists()
