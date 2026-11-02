import numpy as np
import pytest
from scripts.compositional_dynamics_v17_reporting.cohort_videos import panel

def test_failed_slots_visible_and_finite_image_contract(tmp_path):
    missing_prediction=panel(None,848,480,error=True)
    missing_reference=panel(None,848,480,error=False)
    assert missing_prediction.size==(848,480)
    assert not np.array_equal(np.asarray(missing_prediction),np.asarray(missing_reference))
    assert np.any(np.asarray(missing_prediction)!=240)
    p=tmp_path/'image.npz';np.savez(p,rgb=np.full((480,848,3),.5))
    assert np.all(np.asarray(panel(p,848,480))==128)
    np.savez(p,rgb=np.full((480,848,3),np.nan))
    with pytest.raises(ValueError,match='finite'):panel(p,848,480)
