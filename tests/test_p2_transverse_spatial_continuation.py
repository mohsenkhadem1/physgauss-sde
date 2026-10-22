import pytest
from experiments.run_p2_transverse_spatial_continuation import indicator

def test_changed_refinement_ratio_has_exact_correct_conditional_factor():
 assert indicator(1e-6,16,20)==pytest.approx(4e-6,rel=1e-15)
 assert indicator(1e-6,20,24)==pytest.approx(5e-6,rel=1e-15)
 assert indicator(1e-6,24,32)==pytest.approx(3e-6,rel=1e-15)
 assert indicator(7e-6,24,32)>20e-6
 assert indicator(6e-6,24,32)<20e-6
 with pytest.raises(ValueError):indicator(1e-6,24,24)
