import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from audit_campaign_superposition import linearity_diagnostics
from isaac_probe_linearity_diagnostic import diagnostic_menu
from analyze_isaac_linearity import affine_diagnostics


def test_exact_force_linear_response_has_zero_joint_residual():
    rng = np.random.default_rng(12)
    force = np.array([[.02, 0, 0], [0, .02, 0], [0, 0, .02], [.01, .01, .005]])
    response = np.einsum('ai,itnc->atnc', force, rng.normal(size=(3, 5, 7, 3)))
    d = linearity_diagnostics(force, response)
    np.testing.assert_allclose(d['combination_coefficients'], [.5, .5, .25])
    assert d['diagonal_from_three_axis_rmse_m'] < 1e-16
    assert d['best_joint_force_linear_pooled_rmse_m'] < 1e-16


def test_nonlinear_diagonal_response_cannot_be_fitted_by_joint_linear_map():
    force = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [.5, .5, .25]])
    y = np.zeros((4, 5, 7, 3))
    y[3] = 1
    d = linearity_diagnostics(force, y)
    assert d['diagonal_from_three_axis_rmse_m'] == 1
    assert d['best_joint_force_linear_pooled_rmse_m'] > .1


def test_rank_deficient_force_queries_are_rejected():
    with pytest.raises(ValueError, match='span'):
        linearity_diagnostics(np.zeros((4, 3)), np.zeros((4, 1, 1, 3)))


def test_zero_force_menu_preserves_pulse_commands_and_known_hardware():
    centers = [[0., 0, .25], [.1, 0, .25]]
    menu = diagnostic_menu(centers, np.eye(3), np.eye(4))
    assert [a['action_id'] for a in menu['candidates']] == ['poke_0', 'poke_1', 'poke_2', 'zero_force']
    np.testing.assert_allclose(np.array([a['peak_force_N'] for a in menu['candidates'][:3]]), np.eye(3)*.02)
    np.testing.assert_array_equal(menu['candidates'][3]['peak_force_N'], [0, 0, 0])
    np.testing.assert_allclose(menu['evaluation_query']['peak_force_N'], [.01, .01, .005])
    assert menu['known_handle_centers_m'] == centers


def test_measured_zero_removes_arbitrary_support_motion_or_drift_offset():
    force = np.array([[.02, 0, 0], [0, .02, 0], [0, 0, .02], [0, 0, 0], [.01, .01, .005]])
    rng = np.random.default_rng(7)
    offset = rng.normal(size=(6, 7, 3))
    response = np.einsum('ai,itnc->atnc', force, rng.normal(size=(3, 6, 7, 3)))+offset
    metrics, combined = affine_diagnostics(force, response)
    np.testing.assert_allclose(combined, response[4], atol=1e-14)
    assert metrics['best_affine_pooled_rmse_m'] < 1e-13


def test_affine_analysis_requires_a_true_zero_command():
    force = np.eye(5, 3)
    force[3] = 1
    with pytest.raises(ValueError, match='zero response'):
        affine_diagnostics(force, np.zeros((5, 2, 3, 3)))
