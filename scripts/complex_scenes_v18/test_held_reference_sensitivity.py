import numpy as np
import pytest
from scripts.complex_scenes_v18.held_reference_sensitivity import compare


def test_known_translation_and_changed_sample_rejection():
    a = dict(times_s=np.array([0., .1, .2]), material_queries=np.zeros((2, 3)),
             surface_positions_m=np.zeros((3, 1, 2, 3)))
    b = {k: v.copy() for k, v in a.items()}
    b['surface_positions_m'][1:, :, :, 0] = .001
    result = compare(a, b)
    assert result['noninitial_vector_rms_mm'] == pytest.approx(1.)
    assert result['peak_frame_vector_rms_mm'] == pytest.approx(1.)
    b['material_queries'][0, 0] = 1
    with pytest.raises(ValueError, match='Common material'): compare(a, b)
