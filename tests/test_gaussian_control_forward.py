"""Boundary guards and saved-grid matching for controlled teacher collection."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from train_gaussian_control_forward import decimate_passive_training


def test_decimation_uses_saved_frames_and_keeps_validation_dense():
    times = np.arange(61, dtype=np.float32) / 20
    q = np.arange(61 * 2 * 3).reshape(61, 2, 3)
    rows = decimate_passive_training([
        dict(split='train', q=q, times=times),
        dict(split='validation', q=q, times=times)], times[::2])
    np.testing.assert_array_equal(rows[0]['q'], q[::2])
    np.testing.assert_array_equal(rows[1]['q'], q)
    assert rows[1]['q'] is q


def test_decimation_refuses_interpolation_or_duplicate_nearest_frames():
    row = dict(split='train', q=np.zeros((3, 2, 3)), times=np.array([0., 1., 2.]))
    with pytest.raises(ValueError, match='actual requested saved'):
        decimate_passive_training([row], np.array([0., .5, 1.]))
    with pytest.raises(ValueError, match='actual requested saved'):
        decimate_passive_training([row], np.array([0., .1, 1.]))
