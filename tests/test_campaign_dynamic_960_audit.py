"""Matched960 strict coverage/comparison tests; synthetic arrays are not physics."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from experiments import campaign_dynamic_960_audit as audit


def declared():return json.loads(audit.guard.PROTOCOL_PATH.read_text())
def manifest():
    d=declared()
    return {'matched_protocol_sha256':audit.guard.PROTOCOL_SHA256,'approved_runner_source_sha256':[audit.RUNNER_SHA],
        'matched_dependency_sha256':{'campaign_dynamic_960_protocol.py':audit.HELPER_SHA},
        'completed_proof_sha256':{key:item['sha256'] for key,item in d['completed_proofs'].items()},
        'cells':[dict(c,case='t0003',hz=960,output='/tmp/matched960-test-'+c['id']) for c in d['new_cells']]}


def test_exact_eight_ids_and_complete_proof_linkage_required():
    m=manifest();assert len(audit.validate_manifest(m,declared(),[]))==8


@pytest.mark.parametrize('mutation',[
    lambda m:m['cells'].pop(),lambda m:m['cells'].append(deepcopy(m['cells'][0])),
    lambda m:m['cells'][0].update(case='v0003'),lambda m:m['cells'][0].update(case='t0000'),
    lambda m:m['cells'][0].update(hz=480),lambda m:m['cells'][0].update(hz=1920),lambda m:m['cells'][0].update(hz=960.),
    lambda m:m['cells'][0].update(relative_tolerance=1e-7),lambda m:m['cells'][0].update(absolute_tolerance_N=1e-12),
    lambda m:m['cells'][0].update(translation_m=[0,0,-.25]),lambda m:m['cells'][0].update(kind='reset_replay'),
    lambda m:m['cells'][0].update(id='undeclared'),lambda m:m['cells'][0].update(response='zero_external_force'),
    lambda m:m['cells'][0].update(output=m['cells'][1]['output']),
    lambda m:m.update(approved_runner_source_sha256=['0'*64]),lambda m:m.update(matched_dependency_sha256={}),
    lambda m:m.update(completed_proof_sha256={}),lambda m:m.update(matched_protocol_sha256='0'*64),
])
def test_rejects_undeclared_or_incomplete_matrix(mutation):
    m=manifest();mutation(m)
    with pytest.raises(ValueError):audit.validate_manifest(m,declared(),[])


def test_rejects_reusing_any_prior_output():
    m=manifest()
    with pytest.raises(ValueError):audit.validate_manifest(m,declared(),[dict(output=m['cells'][0]['output'])])


def arrays(offset):
    q=np.full((3,2,3),offset,dtype=float)
    return dict(positions_m=q,velocities_point_chord_m_s=q*.1,handle_centers_m=q.copy(),
        handle_rotations=np.broadcast_to(np.eye(3),(3,2,3,3)).copy(),force_step_commands_N=np.ones((2,2,3)))


def samples():
    reused=dict(zip(audit.base.RESPONSES,[arrays(.003),arrays(.0031)]));loaded={}
    for cell in declared()['new_cells']:
        data=deepcopy(reused[cell['response']])
        if cell['kind']=='translated':data['positions_m']+=np.array(cell['translation_m'])
        loaded[cell['id']]=data
    return loaded,reused


def test_six_controls_report_both_accuracy_pairs_and_bitwise_replays():
    loaded,reused=samples();c,signal,tol=audit.compare_controls(loaded,reused)
    assert len(c)==6 and all(x['passed'] for x in c)
    assert signal==pytest.approx(.0001) and tol==pytest.approx(.000005)
    assert all(len(x['coordinate_rmse_m'])==2 for x in c if x['kind'] in ['absolute_floor','nonlinear_tolerance'])
    assert all(all(x['replay_arrays_bitwise'].values()) for x in c if x['kind']=='reset_replay')


def test_common_moving_support_error_cannot_cancel_via_zero_subtraction():
    loaded,reused=samples()
    for response,short in zip(audit.base.RESPONSES,['zero','diagonal']):
        loaded[f'moving960-{short}-translated']['positions_m']+=.0001
    c,_,_=audit.compare_controls(loaded,reused)
    assert all(not x['passed'] for x in c if x['kind']=='translated')


def test_reset_gate_checks_velocities_poses_and_commands_beyond_positions():
    loaded,reused=samples();loaded['moving960-zero-fresh-reset']['force_step_commands_N'][0,0,0]+=.0001
    c,_,_=audit.compare_controls(loaded,reused);record=next(x for x in c if x['kind']=='reset_replay' and x['response']=='zero_external_force')
    assert record['coordinate_rmse_m']==0. and record['passed'] is False
    assert record['replay_arrays_bitwise']['force_step_commands_N'] is False


def test_missing_controls_stay_incomplete():
    loaded,reused=samples();del loaded['moving960-zero-floor1e-12']
    c,_,_=audit.compare_controls(loaded,reused);assert len(c)==5 and not any(x['kind']=='absolute_floor' for x in c)


def test_only_tightest_accuracy_pair_determines_agreement_and_all_differences_reported():
    loaded,reused=samples();loaded['moving960-diagonal-rtol1e-5']['positions_m']+=.001
    c,_,_=audit.compare_controls(loaded,reused);record=next(x for x in c if x['kind']=='nonlinear_tolerance')
    assert record['coordinate_rmse_m'][0]>.0009 and record['coordinate_rmse_m'][1]==0 and record['passed'] is True
