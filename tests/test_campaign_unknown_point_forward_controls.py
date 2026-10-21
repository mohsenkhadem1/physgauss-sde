"""Ordering/hash guards without opening physical truth or a response."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from experiments import campaign_unknown_point_forward_controls as control


def test_absent_marker_stops_before_input_reads(tmp_path):
    with patch.object(control, 'resolve', return_value=tmp_path/'absent.json'), \
            patch.object(control, 'read', side_effect=AssertionError('Input opened')):
        with pytest.raises(ValueError, match='marker absent'):
            control.frozen_gate()


def test_updated_ordering_flags_reject_obsolete_marker(tmp_path):
    marker=tmp_path/'posterior-frozen.json'
    marker.write_text(json.dumps(dict(physical_truth_opened=False,
        held_motion_opened=False, held_motion_evaluated=False)))
    with patch.object(control,'resolve',return_value=marker):
        with pytest.raises(ValueError,match='ordering marker invalid'):
            control.frozen_gate()


def test_all_three_archive_hashes_checked_before_design_or_truth(tmp_path):
    marker=tmp_path/'posterior-frozen.json'
    summary=tmp_path/'reference-summaries.json';summary.write_text('[]')
    marker.write_text(json.dumps(dict(physical_truth_opened=False,
        held_motion_used_for_inference=False,held_motion_evaluated=False,
        reference_summaries_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
        references=[dict(path=f'reference-{s}.npz',sha256='x')
                    for s in (20261112,20261113,20261114)])))
    checked=[]
    def verify(record):
        checked.append(Path(record['path']).name)
        if len(checked)==3:raise ValueError('Archive changed')
    with patch.object(control,'resolve',return_value=marker),patch.object(control,'checked',side_effect=verify):
        with pytest.raises(ValueError,match='Archive changed'):
            control.frozen_gate()
    assert checked==[f'reference-{s}.npz' for s in (20261112,20261113,20261114)]
