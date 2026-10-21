"""Read-only single D-trajectory diagnostic audit; no old-seal qualification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

try:
    import campaign_authored_force_audit as original_audit
except ImportError:
    from experiments import campaign_authored_force_audit as original_audit


STATE_KEYS={'positions':'positions_m','velocities':'velocities_point_chord_m_s',
    'auxiliary_positions':'auxiliary_positions_m','auxiliary_velocities':'auxiliary_velocities_point_chord_m_s',
    'handle_centers':'handle_centers_m','handle_rotations':'handle_rotations'}


def prefix_agreement(new,partial,last):
    """Physical41-frame accepted prefix, never the failed/unaccepted endpoint."""
    clocks=np.asarray(partial['actual_state_times_s'])
    original_audit.require(len(clocks)==41,'Exactly41 original accepted prefix frames required')
    original_audit.exact(new['actual_state_times_s'][:41],clocks,'Accepted prefix state clocks changed')
    original_audit.require(float(last['time_s'])==float(clocks[-1]),'Last accepted state clock differs from prefix')
    differences={}
    for old_key,new_key in STATE_KEYS.items():
        a=np.asarray(partial[old_key]);b=np.asarray(new[new_key][:41])
        original_audit.require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'Complete finite prefix physical state required '+old_key)
        original_audit.exact(a[-1],last[new_key],'Original last accepted physical state/prefix differ '+old_key)
        differences[new_key]=float(np.max(np.abs(a-b)))
    position=max(differences[k] for k in ['positions_m','auxiliary_positions_m','handle_centers_m'])
    velocity=max(differences[k] for k in ['velocities_point_chord_m_s','auxiliary_velocities_point_chord_m_s'])
    rotation=.24*differences['handle_rotations']
    return {'passed':position<=2e-7 and velocity<=2.4e-5 and rotation<=2e-7,
        'frames':41,'accepted_steps':240,'maximum_differences':differences,
        'maximum_position_difference_m':position,'maximum_point_chord_velocity_difference_m_s':velocity,
        'maximum_rotation_difference_scaled_m':rotation,
        'limits':{'position_m':2e-7,'point_chord_velocity_m_s':2.4e-5,'rotation_difference_times_0p24_m':2e-7}}


def run(protocol_path,protocol_sha256):
    try:
        import campaign_solver_trajectory_damped as diagnostic
    except ImportError:
        from experiments import campaign_solver_trajectory_damped as diagnostic
    declaration,old_cell,new_cell,pins=diagnostic.validate_protocol(protocol_path,protocol_sha256)
    canonical=original_audit.read(original_audit.path(original_audit.FIXTURE_PATH))
    report,data=original_audit.verify_cell(new_cell,pins,canonical)
    result={'schema':'single-metric2-trajectory-diagnostic-audit-v0.1','cell_audit':report,'passed':False,
        'posthoc_auditor_source_sha256':original_audit.sha(Path(__file__)),
        'posthoc_auditor_was_prelaunch_requirement':False,
        'protocol':original_audit.evidence(str(protocol_path),'new_D_full_trajectory_declaration'),
        'original_failure_requalified':False,'actual_force_envelope_qualified':False,'production_neural_dataset_admitted':False,
        'native_reference_qualified':False,'continuum_accuracy_established':False,'prediction_qualified':False,'inference_qualified':False,'active_policy_qualified':False}
    if data is not None:
        prefix_root=old_cell['output']
        for name in ['partial-trajectory.npz','last-accepted-state.npz','steps.jsonl']:
            relative=original_audit.child(prefix_root,name)
            original_audit.require(pins.get(relative)==original_audit.sha(original_audit.path(relative)),'Original accepted-prefix input is not pinned '+name)
        prefix=prefix_agreement(data,original_audit.archive(original_audit.path(original_audit.child(prefix_root,'partial-trajectory.npz'))),
            original_audit.archive(original_audit.path(original_audit.child(prefix_root,'last-accepted-state.npz'))))
        rows=[json.loads(line) for line in original_audit.path(original_audit.child(prefix_root,'steps.jsonl')).read_text().splitlines()]
        original_audit.require(len(rows)==240,'Exactly240 original accepted step ledgers required')
        for i,row in enumerate(rows):
            original_audit.require(row['step_index']==i,'Original accepted steps reordered')
            original_audit.exact(data['force_interval_actual_times_s'][i],np.array([row['start_time_s'],row['end_time_s']]),'Original/new actual step clock differs')
            original_audit.exact(data['force_step_commands_N'][i],np.asarray(row['held_handle_forces_N']),'Original/new held force prefix differs')
        execution=original_audit.read(original_audit.path(original_audit.child(new_cell['output'],'execution.json')))
        original_audit.require(not Path('/proc',str(execution['pid'])).exists(),'New diagnostic worker still live; wait for terminal process')
        result.update(prefix_agreement=prefix,passed=bool(report['passed'] and prefix['passed']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);parser.add_argument('--protocol-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=original_audit.path(args.output)
    if output.exists():raise FileExistsError('Fresh diagnostic audit output required')
    result=run(args.protocol,args.protocol_sha256);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':result['passed'],'cell_state':result['cell_audit']['state']}))
