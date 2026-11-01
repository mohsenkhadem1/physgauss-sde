"""Hash-bound bank admission for acquisition design, separate from evaluation.

This is a reproducibility/data-flow guard, not an operating-system sandbox.
Frozen production protocols must bind the manifest SHA passed to this loader.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_design_bank(manifest_path, *, expected_manifest_sha256, design_root, reserved_action_ids):
    root = Path(design_root).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    if not manifest_path.is_relative_to(root):
        raise ValueError('Acquisition manifest is outside the admitted design directory')
    if sha(manifest_path) != expected_manifest_sha256:
        raise ValueError('Design manifest differs from frozen protocol')
    record = json.loads(manifest_path.read_text())
    if record['role'] != 'hypothesis_predictions_for_design':
        raise ValueError('Truth/evaluation banks are not acquisition inputs')
    actions = record['action_ids']
    if not actions or len(actions) != len(set(actions)) or set(actions) & set(reserved_action_ids):
        raise ValueError('Design and reserved evaluation action IDs must be disjoint')
    bank = (manifest_path.parent / record['bank_file']).resolve(strict=True)
    if not bank.is_relative_to(root) or sha(bank) != record['bank_sha256']:
        raise ValueError('Design array path or content differs from frozen manifest')
    with np.load(bank, allow_pickle=False) as archive:
        if set(archive.files) != {'predictions', 'scales'}:
            raise ValueError('Design bank admits predictions/scales only, not observations or truth labels')
        predictions = archive['predictions'].copy()
        scales = archive['scales'].copy()
    if predictions.ndim != 3 or predictions.shape[1] != len(actions):
        raise ValueError('Design predictions must be hypotheses by actions by features')
    if scales.shape != (predictions.shape[2],) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Positive prespecified feature scales required')
    if not np.isfinite(predictions).all():
        raise ValueError('Failed model hypotheses must not silently enter a finite design bank')
    return predictions, scales, record
