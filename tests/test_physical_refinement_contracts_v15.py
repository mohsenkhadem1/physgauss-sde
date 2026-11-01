import json
import numpy as np
import pytest
from scripts.physical_refinement_v15.contracts import load_design_bank, sha


def test_reserved_response_cannot_be_substituted_into_frozen_design(tmp_path):
    bank = tmp_path / 'bank.npz'
    np.savez(bank, predictions=np.zeros((2, 1, 3)), scales=np.ones(3))
    manifest = tmp_path / 'manifest.json'
    record = dict(role='hypothesis_predictions_for_design', action_ids=['design_twist'],
                  bank_file='bank.npz', bank_sha256=sha(bank))
    manifest.write_text(json.dumps(record))
    frozen = sha(manifest)
    assert load_design_bank(manifest, expected_manifest_sha256=frozen, design_root=tmp_path,
                            reserved_action_ids=['test_twist'])[0].shape == (2, 1, 3)
    with pytest.raises(ValueError, match='disjoint'):
        load_design_bank(manifest, expected_manifest_sha256=frozen, design_root=tmp_path,
                         reserved_action_ids=['design_twist'])
    np.savez(bank, predictions=np.ones((2, 1, 3)), scales=np.ones(3))
    with pytest.raises(ValueError, match='content differs'):
        load_design_bank(manifest, expected_manifest_sha256=frozen, design_root=tmp_path,
                         reserved_action_ids=['test_twist'])


def test_truth_fields_are_not_admitted_even_if_manifest_hashes_match(tmp_path):
    bank = tmp_path / 'bank.npz'
    np.savez(bank, predictions=np.zeros((2, 1, 3)), scales=np.ones(3), truth=np.ones(3))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(role='hypothesis_predictions_for_design', action_ids=['design'],
                                        bank_file='bank.npz', bank_sha256=sha(bank))))
    with pytest.raises(ValueError, match='observations or truth'):
        load_design_bank(manifest, expected_manifest_sha256=sha(manifest), design_root=tmp_path,
                         reserved_action_ids=['test'])
