"""Independent gate math and persisted-evidence corruption tests; no dynamics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from experiments import campaign_independent_dynamic_audit as audit
from experiments.campaign_nonlinear_dynamics_case import load_case_inputs,hardware_descriptor


def indexed_cells(supplemental=False):
    return [{'id':str(i),'case':row[0],'response':row[1],'hz':row[2],
        'relative_tolerance':row[3],'translation_m':list(row[4]),'kind':row[5],
        'absolute_tolerance_N':row[6],'output':f'fresh-output-{i}'}
        for i,row in enumerate(audit.required_slots(supplemental))]


def test_exact_factorized_matrix_cannot_hide_missing_floor_cells_or_reuse_replay_output():
    base=indexed_cells();assert len(base)==48
    report=audit.coverage(base,True);assert len(report['missing_slots'])==2
    complete=indexed_cells(True);report=audit.coverage(complete,True)
    assert len(complete)==50 and not any(report[k] for k in ['missing_slots','unexpected_slots','duplicate_slots','duplicate_output_paths'])
    complete[-1]['output']=complete[0]['output']
    assert audit.coverage(complete,True)['duplicate_output_paths']
    complete.append(complete[0].copy());assert audit.coverage(complete,True)['duplicate_slots']


def test_temporal_metric_uses_full_responses_not_cancelling_shared_zero_error():
    # Subtracting zero before trajectory comparison would cancel the entire
    # timestep error here and falsely pass; only the denominator subtracts it.
    zeros=[np.full((31,4,3),bias) for bias in [0.,1e-4,3e-4]]
    diagonal=[q+1e-5 for q in zeros]
    result=audit.temporal_metrics(diagonal,zeros[-1],diagonal[-1])
    np.testing.assert_allclose(result['coordinate_rmse_m'],[1e-4,2e-4],atol=1e-18)
    assert result['tolerance_m']==2e-6 and not result['passed']
    assert all(np.array_equal(a-b,diagonal[0]-zeros[0]) or np.allclose(a-b,1e-5,atol=1e-18) for a,b in zip(diagonal,zeros))


def test_moving_boundary_motion_does_not_inflate_force_signal_tolerance():
    zero=np.full((31,4,3),.008);diagonal=zero+1e-5
    result=audit.temporal_metrics([diagonal,diagonal+3e-6,diagonal+7e-6],zero,diagonal)
    assert result['tolerance_m']==2e-6
    assert not result['passed']  # Using motion-from-rest would allow400um.


def test_temporal_contraction_has_only_the_predeclared_two_micrometre_exception():
    shape=(31,4,3);zero=np.zeros(shape);signal=np.ones(shape)*1e-3
    decreasing=audit.temporal_metrics([zero,zero+1e-5,zero+1.4e-5],zero,signal)
    assert decreasing['passed']
    increasing=audit.temporal_metrics([zero,zero+1e-5,zero+2.2e-5],zero,signal)
    assert not increasing['passed']  # Finest<50um alone is insufficient.
    floor=audit.temporal_metrics([zero,zero+.3e-6,zero+.7e-6],zero,zero)
    assert floor['passed'] and floor['both_differences_below_2micrometre_floor']


def test_reserved_case_rejected_before_reading_any_output(tmp_path):
    cell=indexed_cells()[0];cell.update(case='v0000',output=str(tmp_path/'absent'))
    result,data=audit.audit_cell(cell,{}, {},set(),{})
    assert result['state']=='invalid' and data is None and 'Reserved' in result['error']


def zero_evidence(tmp_path):
    """Analytical exact-rest solution, authored without calling dynamic core."""
    root=tmp_path/'analytical-zero';(root/'provenance').mkdir(parents=True)
    canonical=audit.read(audit.INPUT_PATH);gate=audit.read(audit.GATE_PATH)
    case,spec,arrays,_,_=load_case_inputs('t0001',None,audit.INPUT_PATH)
    points=arrays['rest_points_m'];n=360;hz=120
    names=audit.CORE+['campaign_nonlinear_dynamics_case.py']
    files=[Path('experiments')/name for name in names]+[audit.GATE_PATH,audit.INPUT_PATH]
    hashes={str(path):audit.sha(path) for path in files}
    for path in files:(root/'provenance'/path.name).write_bytes(path.read_bytes())
    np.savez_compressed(root/'provenance/static-reference.npz',**arrays)
    protocol={'episode_id':'t0001','boundary_kind':'fixed_patch','predictive_reserved_responses_opened':False,
        'physics_hz':120,'seconds':3.,'output_hz':10,'response':'zero_external_force',
        'relative_nonlinear_residual_tolerance':1e-7,'absolute_residual_floor_N':1e-10,'nonlinear_iteration_cap':250,
        'translation_m':[0.,0.,0.],'source_input_dependency_and_gate_sha256':hashes,'declared_material':case['material'],
        'body_only_viscous_damping_s_inverse':case['material']['linear_damping_s_inverse'],
        'native_elasticity_damping_reproduced':False,'hardware':hardware_descriptor(points,spec),
        'saved_regenerated_static_reference_sha256':audit.sha(root/'provenance/static-reference.npz')}
    (root/'protocol.json').write_text(json.dumps(protocol))
    intervals=np.column_stack([np.arange(n)/hz,np.arange(1,n+1)/hz])
    data={'times_s':np.arange(31)/10,'actual_state_times_s':np.arange(31)/10,
        'positions_m':np.broadcast_to(points,(31,*points.shape)),
        'velocities_point_chord_m_s':np.zeros((31,*points.shape)),
        'handle_centers_m':np.broadcast_to(points[spec['load_nodes']],(31,2,3)),
        'handle_rotations':np.broadcast_to(np.eye(3),(31,2,3,3)),
        'force_interval_actual_times_s':intervals,'force_midpoints_actual_s':intervals.mean(1),
        'force_step_commands_N':np.zeros((n,2,3)),'requested_step_s':np.full(n,1/hz),'nonlinear_iterations':np.zeros(n,dtype=int)}
    np.savez_compressed(root/'trajectory.npz',**data)
    rows=[]
    for i in range(n):
        row={'step_index':i,'converged':True,'start_time_s':intervals[i,0],'end_time_s':intervals[i,1],'step_s':1/hz,
            'held_handle_forces_N':np.zeros((2,3)).tolist(),'nodal_force_command':'all_zero','residual_tolerance_N':1e-10,
            'nonlinear_iterations':0,'nonlinear_history':[{'scaled_generalized_residual_norm_N':0.,'incremental_potential_J':0.,'minimum_J':1.}],
            'minimum_J':1.,'geometry':{'minimum_volume_ratio':1.,'maximum_rigid_patch_distance_error_m':0.,'fixture_max_error_m':0.,
                'maximum_absolute_Green_principal_strain':0.,'maximum_point_chord_speed_m_s':0.,'maximum_nodal_displacement_m':0.},
            'fixture_reactions_inertia_elastic_drag_minus_external_N':np.zeros((len(spec['fixture_nodes']),3)).tolist(),
            'old_invariants':{'energy_J':0.,'point_linear_momentum_Ns':[0.,0.,0.]},
            'new_invariants':{'energy_J':0.,'point_linear_momentum_Ns':[0.,0.,0.]},'body_drag_impulse_Ns':[0.,0.,0.],
            'linear_momentum_balance_error_Ns':[0.,0.,0.]}
        for key in ['support_work_J','held_handle_work_J','held_nodal_work_J','body_viscous_dissipation_J','constraint_chord_work_J',
            'implicit_velocity_increment_loss_J','backward_elastic_gradient_defect_J','energy_algebra_identity_residual_J']:row[key]=0.
        rows.append(row)
    (root/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary={'completed':True,'frames':31,'trajectory_sha256':audit.sha(root/'trajectory.npz'),
        'steps_sha256':audit.sha(root/'steps.jsonl'),'command_impulse_Ns':np.zeros((2,3)).tolist()}
    (root/'summary.json').write_text(json.dumps(summary));(root/'execution.json').write_text(json.dumps({'terminal':True,'completed':True,'completed_steps':360}))
    cell={'id':'analytic-rest','case':'t0001','response':'zero_external_force','hz':120,'relative_tolerance':1e-7,
        'translation_m':[0,0,0],'kind':'temporal','absolute_tolerance_N':1e-10,'output':str(root)}
    approved={hashes['experiments/campaign_nonlinear_dynamics_case.py']}
    deps={Path(path).name:digest for path,digest in hashes.items() if Path(path).name in audit.CORE}
    return root,cell,gate,canonical,approved,deps


def test_complete_analytical_rest_evidence_passes_but_does_not_qualify_dataset(tmp_path):
    _,cell,gate,canonical,approved,deps=zero_evidence(tmp_path)
    result,data=audit.audit_cell(cell,gate,canonical,approved,deps)
    assert result['passed'] and data is not None
    assert result['absolute_floor_dominated_steps']==360
    assert result['maximum_accepted_physical_residual_N']==0.
    assert result['energy_conservation_claim'] is False
    assert result['external_handle_work_independently_reconstructed_from_saved_step_positions'] is False


@pytest.mark.parametrize('corrupt_work',[False,True])
def test_optional_full_step_center_evidence_independently_checks_held_work(tmp_path,corrupt_work):
    root,cell,gate,canonical,approved,deps=zero_evidence(tmp_path)
    with np.load(root/'trajectory.npz') as data:centers=data['handle_centers_m'][0].tolist()
    rows=[json.loads(row) for row in (root/'steps.jsonl').read_text().splitlines()]
    for row in rows:row.update(handle_centers_start_m=centers,handle_centers_end_m=centers,
        handle_center_step_increment_m=np.zeros((2,3)).tolist())
    if corrupt_work:rows[6]['held_handle_work_J']=1e-8
    (root/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary=audit.read(root/'summary.json');summary['steps_sha256']=audit.sha(root/'steps.jsonl');(root/'summary.json').write_text(json.dumps(summary))
    result,data=audit.audit_cell(cell,gate,canonical,approved,deps)
    assert result['passed'] is not corrupt_work
    if not corrupt_work:assert result['external_handle_work_independently_reconstructed_from_saved_step_positions'] is True
    else:assert 'Independent held-force handle work' in result['error']


def test_supplemental_cells_cannot_use_baseline_provenance_alone(tmp_path):
    _,cell,gate,canonical,approved,deps=zero_evidence(tmp_path)
    cell['kind']='absolute_floor'
    result,data=audit.audit_cell(cell,gate,canonical,approved,deps)
    assert not result['passed'] and data is None
    assert 'Saved supplemental protocol differs' in result['error']


@pytest.mark.parametrize('corruption',['clock','force','residual','hardware','digest','revision'])
def test_corrupt_evidence_cannot_pass_even_after_rehashing_its_container(tmp_path,corruption):
    root,cell,gate,canonical,approved,deps=zero_evidence(tmp_path)
    if corruption in ['clock','force']:
        with np.load(root/'trajectory.npz') as raw:data={key:raw[key].copy() for key in raw.files}
        if corruption=='clock':data['force_interval_actual_times_s'][15,0]+=.0001
        else:data['force_step_commands_N'][100,0,0]=.01
        np.savez_compressed(root/'trajectory.npz',**data)
        summary=audit.read(root/'summary.json');summary['trajectory_sha256']=audit.sha(root/'trajectory.npz');(root/'summary.json').write_text(json.dumps(summary))
    elif corruption=='residual':
        rows=[json.loads(row) for row in (root/'steps.jsonl').read_text().splitlines()]
        rows[5]['nonlinear_history'][-1]['scaled_generalized_residual_norm_N']=1e-8
        (root/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        summary=audit.read(root/'summary.json');summary['steps_sha256']=audit.sha(root/'steps.jsonl');(root/'summary.json').write_text(json.dumps(summary))
    elif corruption=='hardware':
        p=audit.read(root/'protocol.json');p['hardware'][0]['ordered_patch_nodes']=list(reversed(p['hardware'][0]['ordered_patch_nodes']));(root/'protocol.json').write_text(json.dumps(p))
    elif corruption=='digest':(root/'provenance/campaign_nonlinear_dynamics.py').write_text('corrupted')
    else:approved=set()
    result,data=audit.audit_cell(cell,gate,canonical,approved,deps)
    assert not result['passed'] and result['state']=='invalid' and data is None
