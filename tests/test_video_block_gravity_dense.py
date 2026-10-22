import numpy as np
import pytest
from experiments.video_block_gravity_dense import frame_schedule


def test_frame_schedule_shows_all_native_impact_states_without_interpolation():
    t=np.arange(145)/240;idx,desired=frame_schedule(t)
    np.testing.assert_array_equal(np.unique(idx),np.arange(145))
    assert len(idx)==181 and idx[0]==0 and idx[-1]==144
    assert np.max(abs(t[idx]-desired))<=1/480+1e-14
    with pytest.raises(ValueError,match='every recorded'):frame_schedule(t,slow=1,fps=30)
