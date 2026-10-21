"""Physical wrench and public-clock tests, without dataset or truth access."""
import numpy as np
import pytest
from experiments.campaign_balanced_probe import balanced_probe_clock, wrench_diagnostics
from experiments.campaign_fixed_actuator_clock import repeat_for_integration


def test_pair_has_reference_zero_wrench_and_exact_held_refinement():
    x = np.array([[0., 0., .21], [.04, .06, .25]])
    c = balanced_probe_clock(x, amplitude_N=.2)
    d = wrench_diagnostics(c, x)
    assert d['maximum_resultant_N'] == 0
    assert d['maximum_moment_about_world_origin_Nm'] < 2e-17
    assert .199 < d['maximum_per_handle_force_N'] <= .2
    refined = repeat_for_integration(c, integration_hz=240)
    np.testing.assert_array_equal(refined.handle_forces_world_N, np.repeat(c.handle_forces_world_N, 2, axis=0))
    # A fixed WORLD force ceases to be central as handle geometry changes.
    moved = np.broadcast_to(x, (360, 2, 3)).copy(); moved[:, 1, 0] += .01
    assert wrench_diagnostics(c, moved)['maximum_moment_about_world_origin_Nm'] > 1e-4


def test_probe_is_covariant_to_passive_rigid_frame_change():
    x = np.array([[0., 0., .21], [.04, .06, .25]])
    q = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    c = balanced_probe_clock(x); other = balanced_probe_clock(x@q.T+[1., 2., 3.])
    np.testing.assert_allclose(other.handle_forces_world_N, c.handle_forces_world_N@q.T, atol=2e-15, rtol=2e-14)
    with pytest.raises(ValueError): balanced_probe_clock(np.zeros((2, 3)))
    with pytest.raises(ValueError): balanced_probe_clock(x, amplitude_N=-1)
