import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.campaign_authored_commands import (compile_authored_commands, mathematical_total_impulse,
    mathematical_impulse_diagnostics, forces_at)


FIXTURE = Path(__file__).resolve().parents[1]/'configs/gaussian_independent_diagnostic_inputs_v01.json'


@pytest.fixture
def specs():
    return [case['boundaries'] for case in json.loads(FIXTURE.read_text())['cases']]


@pytest.mark.parametrize('family', ['push_pulse', 'pull_hold', 'shear_oscillation', 'twist_couple'])
def test_original_commands_and_clock_preserved_for_every_layout(specs, family):
    for spec in specs:
        original = copy.deepcopy(spec)
        result = compile_authored_commands(spec, force_family=family, physics_hz=120, output_hz=10,
                                           information_role='known_actuator_command')
        source = dict(spec, force_type=family)
        expected = np.stack([forces_at(source, (k+.5)/120) for k in range(360)]).astype(np.float64)
        np.testing.assert_array_equal(result['held_handle_forces_world_N'], expected)
        np.testing.assert_array_equal(result['command_boundary_times_s'], np.arange(361)/120)
        np.testing.assert_array_equal(result['requested_output_times_s'], result['command_boundary_times_s'][::12])
        assert spec == original
        assert result['metadata']['exact_authored_peak_force_N'] == spec['peak_force_N']
        assert result['metadata']['force_family_intervention'] == (family != spec['force_type'])


@pytest.mark.parametrize('family', ['push_pulse', 'pull_hold', 'shear_oscillation', 'twist_couple'])
def test_independent_closed_form_impulses_and_noncancelling_checks(specs, family):
    # Independent quadrature: integrate formulas directly, without envelope/forces_at.
    spec = specs[1]
    x = (np.arange(200000)+.5)/200000
    if family in ['push_pulse', 'twist_couple']:
        y = np.where((x>.1)&(x<.55), np.sin(np.pi*(x-.1)/.45)**2, 0)
        vector = -np.array(spec['normal_direction_world']) if family == 'push_pulse' else np.array(spec['couple_direction_world'])
    elif family == 'pull_hold':
        y = np.sin(np.pi/2*np.clip((x-.1)/.2,0,1))**2*np.cos(np.pi/2*np.clip((x-.65)/.25,0,1))**2
        vector = np.array(spec['normal_direction_world'])
    else:
        p = np.clip((x-.1)/.8,0,1)
        y = np.sin(4*np.pi*p)*np.sin(np.pi*p)**2
        vector = np.array(spec['shear_direction_world'])
    expected = np.zeros((2,3)); expected[0] = y.mean()*spec['seconds']*spec['peak_force_N']*vector
    if family == 'twist_couple': expected[1] = -expected[0]
    np.testing.assert_allclose(mathematical_total_impulse(spec, family), expected, atol=2e-12, rtol=0)
    diagnostics = mathematical_impulse_diagnostics(spec, family)
    expected_absolute = np.abs(expected)
    expected_moment = expected.copy()
    expected_absolute[0] = np.abs(vector)*spec['peak_force_N']*spec['seconds']*np.abs(y).mean()
    expected_moment[0] = vector*spec['peak_force_N']*spec['seconds']**2*(x*y).mean()
    if family == 'twist_couple':
        expected_absolute[1] = expected_absolute[0]
        expected_moment[1] = -expected_moment[0]
    np.testing.assert_allclose(diagnostics['continuous_componentwise_absolute_impulse_Ns'], expected_absolute, atol=1e-10, rtol=0)
    np.testing.assert_allclose(diagnostics['continuous_first_time_moment_Ns2'], expected_moment, atol=1e-10, rtol=0)
    result = compile_authored_commands(spec, force_family=family, physics_hz=480, output_hz=10,
                                       information_role='candidate_force_hypothesis')
    assert result['metadata']['maximum_compiled_vs_continuous_impulse_error_Ns'] < 2e-8
    if family == 'twist_couple':
        np.testing.assert_array_equal(result['held_handle_forces_world_N'].sum(1), 0)
    if family == 'shear_oscillation':
        assert np.abs(result['held_handle_forces_world_N']).max() > .1*spec['peak_force_N']


@pytest.mark.parametrize('override', [dict(physics_hz=True), dict(physics_hz=120.), dict(output_hz=7),
                                     dict(information_role='recipe_is_public'), dict(force_family='unlisted')])
def test_ambiguous_roles_and_incomplete_grids_rejected(specs, override):
    request = dict(force_family='push_pulse', physics_hz=120, output_hz=10, information_role='known_actuator_command')
    request.update(override)
    with pytest.raises(ValueError): compile_authored_commands(specs[0], **request)


@pytest.mark.parametrize('key,value', [('seconds', float('nan')), ('peak_force_N', 0),
                                      ('normal_direction_world', [1,1,0])])
def test_invalid_physical_spec_rejected(specs, key, value):
    spec = dict(specs[0]); spec[key] = value
    with pytest.raises(ValueError):
        compile_authored_commands(spec, force_family='push_pulse', physics_hz=120, output_hz=10,
                                  information_role='known_actuator_command')
