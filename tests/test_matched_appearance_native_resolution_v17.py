from types import SimpleNamespace
import pytest
from scripts.compositional_dynamics_v17 import matched_appearance_native_resolution as m

def test_actual_reader_size_mismatch_rejected():
    spec=dict(eye=[.23,-.3,.17],target=[.06,0,0],width=848,height=480,focal=750.)
    K,_=m.helper.calibration(spec)
    m.validate_camera(SimpleNamespace(image_width=848,image_height=480,K=K),spec)
    with pytest.raises(ValueError,match='dimensions'):m.validate_camera(SimpleNamespace(image_width=256,image_height=192,K=K),spec)
    with pytest.raises(ValueError,match='intrinsic'):m.validate_camera(SimpleNamespace(image_width=848,image_height=480,K=K*.5),spec)
