"""Optional temporal evidence/schema math; no model or time integration."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import campaign_dynamic_extension_audit as audit


def manifest(tmp_path):
    return {'parent_gate_sha256':audit.base.GATE_SHA,'temporal_protocol_sha256':audit.TEMPORAL_SHA,
        'baseline_manifest_sha256':audit.BASE_MANIFEST_SHA,'trigger_audit_sha256':audit.TRIGGER_SHA,
        'approved_runner_source_sha256':[audit.RUNNER_SHA],
        'temporal_dependency_sha256':{'campaign_dynamic_temporal_protocol.py':audit.HELPER_SHA},
        'cells':[{'case':'t0003','response':response,'hz':960,'relative_tolerance':1e-7,
            'absolute_tolerance_N':1e-10,'translation_m':[0,0,0],'kind':'temporal_extension',
            'output':str(tmp_path/response)} for response in audit.base.RESPONSES]}


def test_manifest_has_exactly_two_fresh_declared_cells(tmp_path):
    data=manifest(tmp_path)
    assert len(audit.validate_manifest(data,{'cells':[]}))==2
    with pytest.raises(audit.base.AuditError,match='fresh'):
        audit.validate_manifest(data,{'cells':[{'output':data['cells'][0]['output']}]})


@pytest.mark.parametrize('field,value',[
    ('parent_gate_sha256','0'*64),('temporal_protocol_sha256','0'*64),
    ('baseline_manifest_sha256','0'*64),('trigger_audit_sha256','0'*64),
    ('approved_runner_source_sha256',['unreviewed']),('temporal_dependency_sha256',{}),
])
def test_unapproved_manifest_dependencies_fail(tmp_path,field,value):
    data=manifest(tmp_path);data[field]=value
    with pytest.raises(audit.base.AuditError):audit.validate_manifest(data,{'cells':[]})


@pytest.mark.parametrize('changes',[
    dict(case='v0003'),dict(case='t0001'),dict(hz=480),dict(hz=1920),dict(relative_tolerance=1e-9),
    dict(absolute_tolerance_N=1e-12),dict(translation_m=[0,0,-.25]),dict(kind='reset_replay'),
])
def test_out_of_scope_extension_cells_rejected(tmp_path,changes):
    data=manifest(tmp_path);data['cells'][0].update(changes)
    with pytest.raises(audit.base.AuditError):audit.validate_manifest(data,{'cells':[]})


def test_duplicate_slot_or_output_cannot_count_as_two_cells(tmp_path):
    data=manifest(tmp_path);data['cells'][1]=copy.deepcopy(data['cells'][0])
    with pytest.raises(audit.base.AuditError):audit.validate_manifest(data,{'cells':[]})
    data=manifest(tmp_path);data['cells'][1]['output']=data['cells'][0]['output']
    with pytest.raises(audit.base.AuditError):audit.validate_manifest(data,{'cells':[]})


def conditional_protocol(cell):
    return {'source_input_dependency_and_gate_sha256':{
        'experiments/campaign_nonlinear_dynamics_case.py':audit.RUNNER_SHA,
        'experiments/campaign_dynamic_temporal_protocol.py':audit.HELPER_SHA,
        str(audit.TEMPORAL_PATH):audit.TEMPORAL_SHA,
        'runs/operator-dev/independent-dynamic-t0003-refinement-trigger-v01.json':audit.TRIGGER_SHA},
        'conditional_temporal_protocol':{'protocol_sha256':audit.TEMPORAL_SHA,'parent_gate_sha256':audit.base.GATE_SHA,
            'trigger_audit_sha256':audit.TRIGGER_SHA,'episode_id':'t0003','physics_hz':960,
            'response':cell['response'],'initial_temporal_failure_preserved':True,
            'physical_model_unchanged':True,'reference_qualification_implied':False},
        'supplemental_accuracy_protocol':None}


@pytest.mark.parametrize('corruption',[None,'helper','trigger','conditional_response','qualified','combined'])
def test_conditional_provenance_role_is_checked_beyond_base_byte_integrity(tmp_path,corruption):
    cell=manifest(tmp_path)['cells'][0];root=Path(cell['output']);root.mkdir()
    protocol=conditional_protocol(cell)
    if corruption in ['helper','trigger']:
        key='experiments/campaign_dynamic_temporal_protocol.py' if corruption=='helper' else 'runs/operator-dev/independent-dynamic-t0003-refinement-trigger-v01.json'
        protocol['source_input_dependency_and_gate_sha256'][key]='0'*64
    elif corruption=='conditional_response':protocol['conditional_temporal_protocol']['response']='diagonal_pulse_handle0'
    elif corruption=='qualified':protocol['conditional_temporal_protocol']['reference_qualification_implied']=True
    elif corruption=='combined':protocol['supplemental_accuracy_protocol']={'absolute_residual_floor_N':1e-12}
    (root/'protocol.json').write_text(json.dumps(protocol))
    if corruption is None:audit.verify_conditional_provenance(cell)
    else:
        with pytest.raises(audit.base.AuditError):audit.verify_conditional_provenance(cell)


def test_four_rate_metric_preserves_failed_coarse_levels_and_all_adjacent_errors():
    zero=np.zeros((31,4,3));diagonal=zero+1e-4
    arrays=[zero+q for q in [0,16e-6,24e-6,28e-6]]
    result=audit.extension_metrics(arrays,zero,diagonal)
    np.testing.assert_allclose(result['full_coordinate_rmse_m'],[16e-6,8e-6,4e-6],rtol=0,atol=1e-18)
    assert result['actual_rates_hz']==[120,240,480,960]
    assert result['full_coordinate_rmse_m'][1]>result['tolerance_m']
    assert result['finest_adjacent_comparison_passed']
    assert result['initial_480_failure_is_not_replaced']


def test_zero_cancellation_and_moving_motion_cannot_hide_extension_error():
    background=np.ones((31,4,3))*.008
    zeros=[background+q for q in [0,16e-6,24e-6,28e-6]]
    diagonals=[q+1e-6 for q in zeros]
    result=audit.extension_metrics(diagonals,zeros[-1],diagonals[-1])
    assert result['tolerance_m']==2e-6 and not result['finest_adjacent_comparison_passed']
    np.testing.assert_allclose(diagonals[-1]-zeros[-1],diagonals[0]-zeros[0],atol=1e-18,rtol=0)


def test_finest_error_must_contract_except_the_published_floor_exception():
    q=np.zeros((31,4,3));signal=q+1e-3
    failed=audit.extension_metrics([q,q+10e-6,q+16e-6,q+24e-6],q,signal)
    assert failed['full_coordinate_rmse_m'][-1]<failed['tolerance_m']
    assert not failed['finest_adjacent_comparison_passed']
    floor=audit.extension_metrics([q,q+.1e-6,q+.4e-6,q+.8e-6],q,q)
    assert floor['finest_adjacent_comparison_passed'] and floor['final_two_differences_below_2micrometre_floor']
    with pytest.raises(audit.base.AuditError):audit.extension_metrics([q,q,q],q,q)


def initial_comparisons():
    return copy.deepcopy(json.loads(audit.TEMPORAL_PATH.read_text())['trigger_comparisons'])


def test_initial_failure_objects_are_returned_unchanged_and_cannot_be_reclassified():
    original=initial_comparisons();current=copy.deepcopy(original)
    assert audit.preserve_initial_comparisons(current,original) is original
    current[0]['passed']=True
    with pytest.raises(audit.base.AuditError):audit.preserve_initial_comparisons(current,original)
    current=copy.deepcopy(original);current[1]['coordinate_rmse_m'][-1]=0
    with pytest.raises(audit.base.AuditError):audit.preserve_initial_comparisons(current,original)


def test_proposed_final960_matrix_is_eight_new_cells_and_not_execution_authorization():
    config=json.loads(Path('configs/gaussian_campaign_dynamic_960_validation_v01.json').read_text())
    assert config['status']=='PROPOSED_ONLY_NOT_EXECUTION_AUTHORIZATION'
    assert len(config['new_cells'])==config['minimum_new_runs']==8
    assert len(config['reused_temporal_cells'])==2 and config['total960_operating_point_cells_with_reuse']==10
    assert config['prior480_absolute_floor_study_remains_required'] is True
    cells=config['new_cells'];assert len({cell['id'] for cell in cells})==8
    for kind in ['nonlinear_tolerance','absolute_floor','translated','reset_replay']:
        assert sum(cell['kind']==kind for cell in cells)==2
    assert {cell['absolute_tolerance_N'] for cell in cells if cell['kind']=='absolute_floor'}=={1e-8,1e-12}


def synthetic_run(tmp_path,monkeypatch,*,include_extension=True,valid_handle_work=True):
    """Exercise orchestration with a stubbed, separately tested base verifier.

    Tiny analytic arrays reproduce the declared coarse errors; no synthetic
    state is passed off as a mechanics check or dynamic model trajectory.
    """
    baseline_path=tmp_path/'baseline.json';extension_path=tmp_path/'extension.json'
    baseline={'approved_runner_source_sha256':[audit.BASE_RUNNER_SHA],
        'frozen_source_input_and_gate_sha256':{'campaign_nonlinear_dynamics.py':audit.CORE_SHA},'cells':[]}
    for response in audit.base.RESPONSES:
        for hz in [120,240,480]:
            baseline['cells'].append({'case':'t0003','response':response,'hz':hz,'relative_tolerance':1e-7,
                'absolute_tolerance_N':1e-10,'translation_m':[0,0,0],'kind':'temporal','output':str(tmp_path/f'original-{response}-{hz}')})
    baseline_path.write_text(json.dumps(baseline));extension=manifest(tmp_path)
    extension_path.write_text(json.dumps(extension))
    for cell in extension['cells']:
        root=Path(cell['output']);root.mkdir();(root/'protocol.json').write_text(json.dumps(conditional_protocol(cell)))
    original_read=audit.base.read;original_sha=audit.base.sha
    declaration=original_read(audit.TEMPORAL_PATH);declared=declaration['trigger_comparisons']
    trigger_path=Path(declaration['trigger_audit_path'])
    def read(path):
        if Path(path)==trigger_path:return {'comparisons':declared}
        return original_read(path)
    def sha(path):
        if Path(path)==baseline_path:return audit.BASE_MANIFEST_SHA
        if Path(path)==trigger_path:return audit.TRIGGER_SHA
        return original_sha(path)
    monkeypatch.setattr(audit.base,'read',read);monkeypatch.setattr(audit.base,'sha',sha)
    z0,z1=declared[0]['coordinate_rmse_m'];d0,d1=declared[1]['coordinate_rmse_m']
    signal=declared[0]['finest_force_induced_coordinate_RMS_m'];zero480=z0+z1;diag480=zero480+signal
    values={audit.base.RESPONSES[0]:{120:0.,240:z0,480:zero480,960:zero480+4e-6},
        audit.base.RESPONSES[1]:{120:diag480-d1-d0,240:diag480-d1,480:diag480,960:diag480+4e-6}}
    def verified(cell,*args,**kwargs):
        report={'passed':True,'state':'verified','external_handle_work_independently_reconstructed_from_saved_step_positions':valid_handle_work}
        return report,{'positions_m':np.ones((31,2,3))*values[cell['response']][cell['hz']]}
    monkeypatch.setattr(audit.base,'audit_cell',verified)
    monkeypatch.setattr(audit.base,'mechanics_audit',lambda *args,**kwargs:{'passed':True})
    return audit.run(baseline_path,tmp_path/'report.json',temporal_manifest_path=extension_path if include_extension else None)


def test_complete_passing_extension_remains_temporal_candidate_not_reference(tmp_path,monkeypatch):
    result=synthetic_run(tmp_path,monkeypatch)
    assert result['conditional_960_temporal_candidate_passed'] is True
    assert result['initial_moving_temporal_gate_passed'] is False
    assert result['initial_comparisons_independently_recomputed'] is True
    assert result['matched_960_operating_point_checks_completed'] is False
    assert result['overall_independent_reference_qualified'] is False
    assert result['prior_480_absolute_floor_study_still_required'] is True
    assert len(result['initial_120_240_480_comparisons_preserved'])==len(result['extension_comparisons'])==2


def test_missing_extension_is_incomplete_and_initial_failed_objects_survive(tmp_path,monkeypatch):
    result=synthetic_run(tmp_path,monkeypatch,include_extension=False)
    assert result['temporal_manifest_state']=='missing'
    assert result['indexed_temporal_cells']==6 and result['required_temporal_cells']==8
    assert result['conditional_960_temporal_candidate_passed'] is False
    assert all(c['passed'] is False for c in result['initial_120_240_480_comparisons_preserved'])


def test_revised_runner_without_promised_handle_increment_evidence_cannot_pass(tmp_path,monkeypatch):
    result=synthetic_run(tmp_path,monkeypatch,valid_handle_work=False)
    assert result['conditional_960_temporal_candidate_passed'] is False
    rejected=[cell for cell in result['cells'] if cell['evidence_role']=='conditional_temporal']
    assert len(rejected)==2 and all(cell['state']=='invalid' for cell in rejected)
