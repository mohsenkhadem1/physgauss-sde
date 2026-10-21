from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from campaign_free_handle_clock_adapter import validated_native_clock,derive_analyzer


def test_native_float32_clock_accepted_and_double_getter_mismatch_explicit():
    for hz in [120,960]:
        n=3*hz;nominal=np.full(n,1/hz);actual=nominal.astype(np.float32).astype(np.float64)
        times=.2+np.r_[0.,np.cumsum(actual)]
        result=validated_native_clock({'world_times_s':times,'world_dt_s':nominal,'manager_dt_s':nominal})
        np.testing.assert_allclose(result['measured_step_dt_s'],actual,atol=1e-15,rtol=0)
        assert result['nominal_getter_elapsed_error_s']>1e-7
        assert result['maximum_step_error_vs_float32_getter_dt_s']<1e-15


def test_real_clock_drift_or_missing_step_is_not_excused_as_rounding():
    nominal=np.full(3,1/120)
    for times in [[0,.1,.2,.3],[0,0,0,0]]:
        with pytest.raises(ValueError):validated_native_clock({'world_times_s':times,'world_dt_s':nominal,'manager_dt_s':nominal})
    with pytest.raises(ValueError):validated_native_clock({'world_times_s':[0,.1],'world_dt_s':nominal,'manager_dt_s':nominal})


def test_analyzer_adapter_leaves_physics_and_force_gate_code_unchanged():
    _,original,derived=derive_analyzer()
    assert "dt=data['world_dt_s'];elapsed=" in original
    assert "clock_audit=validated_native_clock(data)" in derived
    for untouched in ["relative_velocity_error<=protocol['config']['relative_momentum_tolerance']",
                      "reference=free_held_reference", "hardware_valid=", "commands=command_schedule(hz)"]:
        assert untouched in original and untouched in derived
