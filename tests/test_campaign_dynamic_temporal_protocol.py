"""Exact prospective 960 Hz controls; no model construction or integration."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import campaign_dynamic_temporal_protocol as temporal


DRAFT_PATH = Path('runs/operator-dev/floor-runner-extension-draft/campaign_nonlinear_dynamics_case.py')
RUNNER_PATH = DRAFT_PATH if DRAFT_PATH.exists() else Path('experiments/campaign_nonlinear_dynamics_case.py')


def request(**changes):
    values = dict(episode='t0003', hz=960, response='zero_external_force',
        relative_tolerance=1e-7, absolute_tolerance_N=1e-10,
        maximum_iterations=250, translation=[0., 0., 0.],
        inputs_json='configs/gaussian_independent_diagnostic_inputs_v01.json',
        core_path='experiments/campaign_nonlinear_dynamics.py',
        gate_path='configs/gaussian_campaign_independent_dynamic_gate_v01.json')
    values.update(changes)
    return values


@pytest.mark.parametrize('response',['zero_external_force','diagonal_pulse_handle0'])
def test_only_two_declared_requests_preserve_initial_failed_comparisons(response):
    assert temporal.digest(temporal.PROTOCOL_PATH) == temporal.PROTOCOL_SHA256
    metadata = temporal.validate_temporal_extension(temporal.PROTOCOL_PATH, **request(response=response))
    assert metadata['physics_hz'] == 960 and metadata['response'] == response
    assert metadata['initial_temporal_failure_preserved'] is True
    assert metadata['physical_model_unchanged'] is True
    assert metadata['reference_qualification_implied'] is False
    assert temporal.digest(metadata['trigger_audit_path']) == temporal.TRIGGER_SHA256


@pytest.mark.parametrize('changes',[
    dict(episode='v0003'),dict(episode='t0000'),dict(hz=480),dict(hz=1920),dict(hz=960.),
    dict(response='poke0'),dict(relative_tolerance=1e-5),dict(relative_tolerance=float('nan')),
    dict(absolute_tolerance_N=1e-12),dict(absolute_tolerance_N=float('inf')),
    dict(maximum_iterations=251),dict(maximum_iterations=250.),
    dict(translation=[0.,0.,-.25]),dict(translation=[0.,float('nan'),0.]),
    dict(inputs_json=None),
])
def test_rejects_unapproved_rates_cases_forces_or_accuracy(changes):
    with pytest.raises(ValueError):
        temporal.validate_temporal_extension(temporal.PROTOCOL_PATH, **request(**changes))


def test_pinned_protocol_core_gate_fixture_and_trigger_cannot_change(tmp_path,monkeypatch):
    altered=tmp_path/'altered.json'
    protocol=json.loads(temporal.PROTOCOL_PATH.read_text());protocol['new_physics_hz']=1920
    altered.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='declaration'):
        temporal.validate_temporal_extension(altered,**request())
    altered.write_text('changed dependency')
    for key in ['core_path','gate_path','inputs_json']:
        with pytest.raises(ValueError):
            temporal.validate_temporal_extension(temporal.PROTOCOL_PATH,**request(**{key:altered}))
    trigger=Path(protocol['trigger_audit_path']);original_digest=temporal.digest
    monkeypatch.setattr(temporal,'digest',lambda path:'0'*64 if Path(path)==trigger else original_digest(path))
    with pytest.raises(ValueError,match='trigger evidence'):
        temporal.validate_temporal_extension(temporal.PROTOCOL_PATH,**request())


@pytest.fixture
def draft(monkeypatch):
    # Locally inspect draft bytes without applying them. A clean checkout uses
    # the published runner, so tests do not require ignored operator artifacts.
    spec=importlib.util.spec_from_file_location('prospective_dynamic_runner',RUNNER_PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'__file__',str(Path('experiments/campaign_nonlinear_dynamics_case.py').resolve()))
    return module


class StopBeforeInputs(Exception):pass


@pytest.mark.parametrize('response',['zero_external_force','diagonal_pulse_handle0'])
def test_draft_960_guard_reaches_static_input_boundary_without_creating_output(tmp_path,monkeypatch,draft,response):
    def stop(*args,**kwargs):raise StopBeforeInputs
    monkeypatch.setattr(draft,'load_case_inputs',stop)
    output=tmp_path/'must-not-exist'
    with pytest.raises(StopBeforeInputs):
        draft.run(output,episode='t0003',hz=960,response=response,
            inputs_json='configs/gaussian_independent_diagnostic_inputs_v01.json',
            temporal_extension_protocol=temporal.PROTOCOL_PATH)
    assert not output.exists()


@pytest.mark.parametrize('changes',[
    dict(temporal_extension_protocol=None),dict(hz=1920),dict(episode='t0002'),
    dict(absolute_tolerance_N=1e-12),dict(relative_tolerance=1e-9),
    dict(supplemental_floor_protocol='configs/gaussian_campaign_dynamic_floor_refinement_v01.json'),
])
def test_draft_rejects_unapproved_conditional_requests_before_inputs_and_output(tmp_path,monkeypatch,draft,changes):
    def inputs_must_not_open(*args,**kwargs):pytest.fail('Invalid request reached static input loading')
    monkeypatch.setattr(draft,'load_case_inputs',inputs_must_not_open)
    output=tmp_path/'must-not-exist'
    args=dict(episode='t0003',hz=960,response='zero_external_force',
        inputs_json='configs/gaussian_independent_diagnostic_inputs_v01.json',
        temporal_extension_protocol=temporal.PROTOCOL_PATH)
    args.update(changes)
    with pytest.raises(ValueError):draft.run(output,**args)
    assert not output.exists()


def test_draft_retains_initial_matrix_and_separate_floor_guard(tmp_path,monkeypatch,draft):
    gate=json.loads(draft.GATE_PATH.read_text())
    for case in draft.CASE_BOUNDARIES:
        for hz in [120,240,480]:
            np.testing.assert_array_equal(draft.validate_request(gate,case,hz,'diagonal_pulse_handle0',1e-7,[0,0,0]),[0,0,0])
    def stop(*args,**kwargs):raise StopBeforeInputs
    monkeypatch.setattr(draft,'load_case_inputs',stop)
    for floor in [1e-8,1e-12]:
        output=tmp_path/str(floor)
        with pytest.raises(StopBeforeInputs):
            draft.run(output,episode='t0003',hz=480,response='zero_external_force',absolute_tolerance_N=floor,
                inputs_json='configs/gaussian_independent_diagnostic_inputs_v01.json',
                supplemental_floor_protocol='configs/gaussian_campaign_dynamic_floor_refinement_v01.json')
        assert not output.exists()
