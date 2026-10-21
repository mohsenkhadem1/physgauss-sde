"""Prospective matrix/command checks only; no solver, dataset or response read."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import campaign_authored_commands as commands


ROOT=Path(__file__).parent
FAMILIES=['push_pulse','pull_hold','shear_oscillation','twist_couple']


def read():return json.loads((ROOT/'protocol-draft-v01.json').read_text())
def fixture():return json.loads(Path(read()['source_fixture']).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_frozen_public_inputs_compiler_and_physical_core_hashes():
    d=read();assert digest(d['source_fixture'])==d['source_fixture_sha256']
    for path,wanted in d['frozen_source_sha256'].items():assert digest(path)==wanted,path
    assert d['frozen_source_sha256']['experiments/campaign_nonlinear_dynamics.py']=='da0db7018b4913020bb228c2fc4df3c240e7dd872f17db4868de147eaaf78fa2'


def test_exact_cartesian_family_interventions_retain_source_peaks_and_static_identity():
    d=read();cases={case['episode_id']:case for case in fixture()['cases']}
    assert {(s['source_case'],s['requested_force_family']) for s in d['scenarios']}=={(case,family) for case in cases for family in FAMILIES}
    assert len(d['scenarios'])==16
    for row in d['scenarios']:
        original=cases[row['source_case']];spec=original['boundaries']
        assert row['source_recipe_force_family']==spec['force_type']=='push_pulse'
        assert row['family_intervention']==(row['requested_force_family']!='push_pulse')
        assert row['exact_authored_amplitude_coefficient_N']==spec['peak_force_N']
        assert row['source_recipe_sha256']==original['recipe']['recipe_sha256']
        expected=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert row['source_spec_canonical_sha256']==expected
    assert cases['t0000']['boundaries']['peak_force_N']==.030249222859091197
    assert all(row['exact_authored_amplitude_coefficient_N']==.030249222859091197 for row in d['scenarios'] if row['source_case']=='t0000')


def test_exact_119_plan_is_staged_51_then_68_without_duplicate_or_silent_missing_coverage():
    d=read();rows=d['initial_cell_plan'];assert len(rows)==119==len({row['id'] for row in rows})
    first=[row for row in rows if row['stage']=='initial_temporal'];second=[row for row in rows if row['stage']=='matched_after_temporal']
    assert len(first)==51 and len(second)==68
    for scenario in d['scenarios']:
        time=[row for row in first if row['scenario_id']==scenario['id']]
        assert sorted(row['hz'] for row in time)==scenario['initial_rates_hz']
        matched=[row for row in second if row['scenario_id']==scenario['id']]
        assert len(matched)==4
        assert sorted(row['relative_tolerance'] for row in matched if row['kind']=='nonlinear_tolerance')==[1e-9,1e-5]
        assert sum(row['kind']=='translated' for row in matched)==sum(row['kind']=='reset_replay' for row in matched)==1
    assert sum(row['response']=='zero_external_force' for row in first)==3
    assert sum(row['response']=='zero_external_force' for row in second)==4
    assert all(row['output_hz']==20 and row['maximum_iterations']==250 for row in rows)


def test_matched_rows_resolve_only_to_passing_candidate_never_an_assumed_default_rate():
    d=read()
    for row in d['initial_cell_plan']:
        if row['stage']=='matched_after_temporal':
            assert 'hz' not in row
            assert row['hz_selector'] in ['accepted_temporal_candidate_rate_for_this_scenario','each_distinct_accepted_moving_case_candidate_rate']
            assert 'publish' in row['execution_condition'].lower()
    assert 'no default' in d['resolved_manifest_requirements']['candidate_resolution'].lower()


@pytest.mark.parametrize('family',FAMILIES)
def test_all_four_force_laws_preserve_exact_promoted_authored_commands_at_all_initial_clocks(family):
    d=read();cases={case['episode_id']:case for case in fixture()['cases']}
    for scenario in [s for s in d['scenarios'] if s['requested_force_family']==family]:
        spec=cases[scenario['source_case']]['boundaries'];intervention=dict(spec,force_type=family)
        for hz in scenario['initial_rates_hz']:
            result=commands.compile_authored_commands(spec,force_family=family,physics_hz=hz,output_hz=20,information_role='known_actuator_command')
            assert len(result['requested_output_times_s'])==61
            np.testing.assert_array_equal(result['requested_output_times_s'],result['command_boundary_times_s'][::hz//20])
            np.testing.assert_array_equal(result['force_midpoints_s'],(np.arange(3*hz)+.5)/hz)
            direct=np.stack([commands.forces_at(intervention,t) for t in result['force_midpoints_s']])
            assert direct.dtype==np.float32
            np.testing.assert_array_equal(result['held_handle_forces_world_N'],direct.astype(np.float64))
            worker_expression=np.stack([commands.forces_at(intervention,(step+1)/hz-.5/hz) for step in range(3*hz)])
            np.testing.assert_array_equal(direct,worker_expression)
            assert result['metadata']['exact_authored_peak_force_N']==spec['peak_force_N']
            assert result['metadata']['native_engine_or_trajectory_equivalence'] is False
            assert 'Compiled integer grid only' in result['metadata']['clock_evidence']


def test_oscillation_diagnostics_cannot_be_replaced_by_zero_signed_impulse():
    spec=fixture()['cases'][1]['boundaries']
    result=commands.compile_authored_commands(spec,force_family='shear_oscillation',physics_hz=480,output_hz=20,information_role='known_actuator_command')
    meta=result['metadata']
    assert np.max(abs(np.asarray(meta['continuous_unquantized_impulse_Ns'])))==0
    assert np.max(meta['compiled_componentwise_absolute_impulse_Ns'])>.01
    expected=commands.mathematical_peak_vectors(spec,'shear_oscillation')*(.24/np.pi)
    np.testing.assert_allclose(meta['continuous_first_time_moment_Ns2'],expected,atol=1e-17,rtol=0)


def test_couple_is_zero_resultant_but_moving_origin_moment_is_not_constant_torque():
    spec=fixture()['cases'][1]['boundaries']
    result=commands.compile_authored_commands(spec,force_family='twist_couple',physics_hz=480,output_hz=20,information_role='known_actuator_command')
    forces=result['held_handle_forces_world_N'];np.testing.assert_array_equal(forces.sum(1),0.)
    F=forces[np.argmax(np.linalg.norm(forces[:,0],axis=1))]
    old=np.array([[.1,.02,.25],[-.1,-.04,.21]]);new=old.copy();new[1]+=[.02,.03,-.01]
    old_moment=np.cross(old,F).sum(0);new_moment=np.cross(new,F).sum(0)
    assert np.linalg.norm(new_moment-old_moment)>1e-5
    np.testing.assert_allclose(np.cross(new+[.5,-.25,4.],F).sum(0),new_moment,atol=1e-16,rtol=0)
    assert 'ACTUAL moving handle origins' in read()['force_semantics']['diagnostics'][-1]


def test_zero_reference_reuse_does_not_fabricate_moving_frames_or_static_reset_evidence():
    d=read();zeros=[row for row in d['initial_cell_plan'] if row['response']=='zero_external_force']
    assert len(zeros)==7 and all(row['case']=='t0003' and row['output_hz']==20 for row in zeros)
    assert d['counts']['analytic_static_zero_reference_grids']==9
    assert 'never labelled captured' in d['reuse_policy']['static_zero']
    assert 'No upsampling' in d['reuse_policy']['moving_zero']
    assert 'not captured/replayed resets' in d['resolved_manifest_requirements']['static_analytic']


def test_conditional_rates_are_bounded_distinct_and_not_automatic_authorization():
    d=read();optional=d['conditional_refinement']['allowed_next_rates_hz']
    assert optional['moving_patch']==[1920,3840]
    assert all(optional[name]==[960,1920] for name in ['point_pin','fixed_patch','two_patches'])
    for scenario in d['scenarios']:
        assert not set(scenario['initial_rates_hz']).intersection(optional[scenario['boundary_type']])
    assert 'never automatic' in d['conditional_refinement']['status']
    assert d['cpu_policy_proposal']['optional_max_parallel_workers_after_separate_resource_predeclaration']==8
    assert d['cpu_policy_proposal']['threads_per_worker']==2


def test_geometry_guards_match_discrete_model_without_inventing_continuous_contact_claims():
    g=read()['geometry_contact_guards']
    assert 'tetJ>=0.1' in g['accepted_endpoints']
    assert 'NOT MODELED' in g['contact_model']
    assert 'not proof of continuous no-contact' in g['surface_diagnostic']
    assert 'No continuous-chord determinant model' in g['not_required']
    assert 'No invented principal-stretch interval as real-material/admissibility gate' in g['not_required']
    assert 'BEFORE cell termination' in g['failure_preservation']


def test_draft_remains_unqualified_and_requires_pending_proof_runner_auditor_manifest_freeze():
    d=read();assert 'NOT_LAUNCH_AUTHORIZATION' in d['status']
    pending=d['proof_pin_pending_before_launch']
    assert pending['small_force_complete60_audit_sha256']=='53b4a43c2c86485180ee6786bc4d38b7e8897c9e3dcc737de85f7e7adfa8444d'
    assert all(pending[key] is None for key in ['new_envelope_runner_sha256','new_envelope_auditor_sha256','resolved_stage1_manifest_sha256'])
    assert d['frames']==61 and d['seconds']==3.
    assert d['physical_model_semantics']['native_elasticity_damping_reproduced'] is False
    assert 'new_dataset_generation_qualified' in d['dataset_admission']['gates_false']


def test_internal_refinement_clocks_are_not_latent_bc_labels_in_neural_public_inputs():
    clock=read()['dataset_admission']['public_clock_fidelity']
    assert 'internal numerical operating points' in clock['engineering_only']
    assert 'reveal the true latentBC' in clock['no_latent_label_leakage']
    assert 'common known actuator-command clock independent of latentBC' in clock['production_requirement']
    assert 'identical public known commands/times' in clock['hypothesis_consistency']
    assert 'does NOT automatically qualify' in clock['scope_limit']


def test_candidate_proof_requires_complete_response_evidence_not_a_solver_success_flag():
    proof=read()['candidate_proof_contract']
    assert 'passed flag alone is insufficient' in proof['rate_guard']
    assert 'zero+driven' in proof['rate_guard']
    assert 'immutable object' in proof['initial_false_retained']
    assert 'independently reviewed authored-family auditor' in proof['authority']
