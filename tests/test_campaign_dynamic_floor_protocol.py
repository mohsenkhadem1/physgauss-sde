"""Supplemental accuracy requests cannot silently change physical inputs."""
import json

import pytest

from experiments.campaign_dynamic_floor_protocol import (
    PROTOCOL_PATH, PROTOCOL_SHA256, digest, validate_supplemental_floor,
)


def request(**changes):
    values = dict(
        episode='t0003', hz=480, response='zero_external_force',
        relative_tolerance=1e-7, absolute_tolerance_N=1e-12,
        maximum_iterations=250, translation=[0., 0., 0.],
        inputs_json='configs/gaussian_independent_diagnostic_inputs_v01.json',
        core_path='experiments/campaign_nonlinear_dynamics.py',
        gate_path='configs/gaussian_campaign_independent_dynamic_gate_v01.json',
    )
    values.update(changes)
    return values


def test_public_declaration_allows_only_two_new_effective_accuracy_levels():
    assert digest(PROTOCOL_PATH) == PROTOCOL_SHA256
    for floor in [1e-8, 1e-12]:
        metadata = validate_supplemental_floor(PROTOCOL_PATH, **request(absolute_tolerance_N=floor))
        assert metadata['physical_model_unchanged'] is True
        assert metadata['absolute_residual_floor_N'] == floor
        assert metadata['baseline_absolute_residual_floor_N_reused'] == 1e-10


@pytest.mark.parametrize('changes', [
    dict(episode='v0003'), dict(episode='t0001'), dict(hz=240),
    dict(response='diagonal_pulse_handle0'), dict(relative_tolerance=1e-9),
    dict(absolute_tolerance_N=1e-10), dict(absolute_tolerance_N=1e-14),
    dict(absolute_tolerance_N=float('nan')), dict(maximum_iterations=251),
    dict(translation=[0., 0., -.25]), dict(inputs_json=None),
])
def test_rejects_out_of_scope_or_duplicate_baseline_requests(changes):
    with pytest.raises(ValueError):
        validate_supplemental_floor(PROTOCOL_PATH, **request(**changes))


def test_changed_protocol_core_gate_and_fixture_are_rejected(tmp_path):
    altered = tmp_path/'altered.json'
    protocol = json.loads(PROTOCOL_PATH.read_text())
    protocol['absolute_residual_floors_N'][-1] = 1e-14
    altered.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='declaration'):
        validate_supplemental_floor(altered, **request())
    altered.write_text('a different physical input or implementation')
    for key in ['core_path', 'gate_path', 'inputs_json']:
        with pytest.raises(ValueError):
            validate_supplemental_floor(PROTOCOL_PATH, **request(**{key: altered}))
