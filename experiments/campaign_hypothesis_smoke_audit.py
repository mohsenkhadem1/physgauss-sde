"""Read-only single-hypothesis capture audit; no dynamics construction or steps.

Reuses the published pure frame-mechanics reconstruction with an explicit
new-spec adapter. This checks saved evidence, not temporal/continuum accuracy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
try:
    import campaign_hypothesis_dynamics as runner
    import campaign_hypothesis_inputs as inputs
    import campaign_exciting_commands as commands
    import campaign_authored_force_audit as pure
    from campaign_nonlinear_fem import FiniteRigidPatchMap, rotation_matrix, tensor
except ImportError:
    from experiments import campaign_hypothesis_dynamics as runner
    from experiments import campaign_hypothesis_inputs as inputs
    from experiments import campaign_exciting_commands as commands
    from experiments import campaign_authored_force_audit as pure
    from experiments.campaign_nonlinear_fem import FiniteRigidPatchMap, rotation_matrix, tensor

require,close,exact,read,archive = pure.require,pure.close,pure.exact,pure.read,pure.archive


def audit(protocol_path, protocol_sha256):
    protocol=runner.validate_protocol(protocol_path,protocol_sha256)
    root=runner.resolve(protocol['output']);private=root/'private-provenance'
    require(not any(p.is_symlink() for p in root.rglob('*')),'Capture contains symlinks')
    execution=read(root/'execution.json');summary=read(root/'summary.json')
    require(execution['terminal'] and execution['completed'] and not (root/'failure.json').exists(),'Capture not successfully terminal')
    require(not Path('/proc/'+str(execution['pid'])).exists(),'Worker PID still exists')
    require(execution['cpu_threads']==2 and execution['gpu_used'] is False,'Unexpected resources')
    require(read(root/'protocol.json')==protocol,'Saved protocol differs')
    design=read(runner.checked(protocol['input_design']))
    cell=next(c for c in design['cells'] if c['id']==protocol['cell_id'])
    require(cell['program_id']==protocol['program_id'] and cell['split']=='train','Finite TRAIN design membership differs')
    public=read(runner.checked(protocol['public_hardware']));spec=read(runner.checked(protocol['realized_hypothesis']))
    require(next(h for h in design['hypotheses'] if h['id']==cell['hypothesis_id'])['spec']==spec,'Declared hypothesis membership differs')
    require(next(h for h in design['public_hardware'] if h['id']==cell['public_hardware_id'])==public,'Public hardware membership differs')
    hardware=inputs.known_hardware_from_public(public);inputs.validate_realized_hypothesis(hardware,spec)
    require(read(root/'public-hardware.json')==public and read(private/'realized-hypothesis.json')==spec,'Public/private input snapshots differ')
    pins={**protocol['source_sha256'],protocol_path:protocol_sha256,
        **{protocol[k]['path']:protocol[k]['sha256'] for k in ['public_hardware','realized_hypothesis','known_commands']}}
    snapshots=read(private/'snapshots.json')
    require(len(snapshots)==len(pins) and {s['original_path'] for s in snapshots}==set(pins),'Snapshot coverage differs')
    for item in snapshots:
        require(Path(item['saved_path']).name==item['saved_path'] and item['sha256']==pins[item['original_path']],'Unsafe/different snapshot index')
        require(pure.sha(private/item['saved_path'])==item['sha256'],'Saved source snapshot hash differs')
    static=archive(private/'static-reference.npz')
    for key,value in [('rest_points_m',hardware.points),('tetrahedra',hardware.tets),('surface_faces',hardware.faces)]:exact(static[key],value,'Static hardware array differs '+key)
    clock=commands.make_exciting_commands(protocol['program_id'])
    source=archive(runner.checked(protocol['known_commands']))
    for key in ['endpoint_times_s','output_times_s','handle_forces_world_N','handle_torques_world_Nm','nodal_forces_world_N','body_acceleration_world_m_s2']:
        exact(source[key],getattr(clock,key),'Known program bytes differ '+key)
    require(read(root/'public-command-manifest.json')==clock.public_manifest(),'Public command manifest differs')
    public_arrays=archive(root/'public-commands.npz')
    require(set(public_arrays)==set(clock.public_features()),'Extra public conditioning fields')
    for key,value in clock.public_features().items():exact(public_arrays[key],value,'Public commands differ '+key)
    refined=runner.validate_clock(clock,protocol['integration_hz'],len(hardware.points),2)
    saved=archive(private/'integration-commands.npz')
    require(set(saved)==set(refined.private_features()),'Private command schema differs')
    for key,value in refined.private_features().items():exact(saved[key],value,'Private held repeat differs '+key)
    data=archive(root/'trajectory.npz');steps=[json.loads(s) for s in (root/'steps.jsonl').read_text().splitlines()]
    require(all(a.dtype==np.float64 for a in data.values()),'Physical trajectory must remain float64')
    count=len(refined.parent_command_indices);hz=protocol['integration_hz']
    require(len(steps)==count==execution['completed_steps']==execution['planned_steps']==summary['steps'] and summary['frames']==61,'Incomplete trajectory/steps')
    exact(data['times_s'],clock.output_times_s,'Output clock differs');close(data['actual_state_times_s'],clock.output_times_s,1e-12,'Actual output clock differs')
    exact(data['force_step_commands_N'],refined.handle_forces_world_N,'Saved force bytes differ')
    intervals=data['force_interval_actual_times_s'];require(intervals.shape==(count,2),'Complete actual intervals required')
    close(intervals,np.stack([refined.endpoint_times_s[:-1],refined.endpoint_times_s[1:]],1),1e-12,'Actual integration clocks differ')
    exact(intervals[1:,0],intervals[:-1,1],'Actual clock discontinuity')
    oldspec={**spec,'load_nodes':[h.origin_node_id for h in hardware.handles],
        'load_patches':[h.ordered_patch_nodes.tolist() for h in hardware.handles]}
    oldhardware=[dict(mass_kg=h.mass_kg,full_COM_inertia_kg_m2=h.com_inertia_body_kg_m2,COM_offset_m=h.com_offset_body_m) for h in hardware.handles]
    geometry,P,kinetic,elastic=pure.frame_mechanics(data,hardware.points,hardware.tets,oldspec,oldhardware,spec['material'])
    mapping=FiniteRigidPatchMap(hardware.points,spec['fixture_nodes'],oldspec['load_patches'],oldspec['load_nodes'])
    require(data['reduced_position_coordinates'].shape==(61,mapping.size),'Reduced-coordinate shape differs')
    for i,z in enumerate(data['reduced_position_coordinates']):
        # Dynamic stored rotation coordinates reset to zero each step; the
        # accumulated finite rotations are separate saved state variables.
        mapped=hardware.points+(mapping.A@tensor(z)).numpy().reshape(-1,3)
        for j,(patch,node,dofs) in enumerate(zip(oldspec['load_patches'],oldspec['load_nodes'],mapping.handle_dofs)):
            exact(z[dofs[3:]],np.zeros(3),'Stored local rotation increment was not reset')
            center=hardware.points[node]+z[dofs[:3]]
            close(center,data['handle_centers_m'][i,j],1e-12,'Reduced handle translation mismatch')
            mapped[patch]=center+(hardware.points[patch]-hardware.points[node])@data['handle_rotations'][i,j].T
        mapped[spec['fixture_nodes']]=pure.fixture_targets(hardware.points,oldspec,[data['actual_state_times_s'][i]])[0]
        close(mapped,data['positions_m'][i],1e-12,'Reduced coordinate/position mismatch')
    residuals=[];thresholds=[];runtimeJ=[];work_error=[];energy_error=[];momentum_error=[];last=None;lastcenter=data['handle_centers_m'][0]
    for i,row in enumerate(steps):
        pure.finite_tree(row);F=refined.handle_forces_world_N[i]
        require(row['step_index']==i and row['parent_public_command_index']==int(refined.parent_command_indices[i]),'Step/parent index differs')
        exact(row['compiled_interval_s'],refined.endpoint_times_s[i:i+2],'Compiled clock differs')
        exact(row['actual_interval_s'],intervals[i],'Ledger actual clock differs');exact([row['start_time_s'],row['end_time_s']],intervals[i],'Core clock differs')
        require(row['actual_state_start_s']==intervals[i,0] and row['requested_dt_s']==row['step_s']==1/hz,'Requested dt differs')
        exact(row['held_handle_forces_world_N'],F,'Requested held forces differ');exact(row['held_handle_forces_N'],F,'Core held forces differ')
        require(not np.any(row['held_nodal_forces_N']) and not np.any(row['held_nodal_forces_world_N']) and row['held_nodal_work_J']==0,'Undeclared nodal loading')
        threshold=max(1e-10,1e-7*np.linalg.norm(F));close(row['residual_tolerance_N'],threshold,1e-25+threshold*1e-12,'Residual tolerance differs')
        history=row['nonlinear_history'];require(row['converged'] and 1<=len(history)<=251,'Invalid accepted nonlinear history')
        for k,h in enumerate(history):require(h['iteration']==k and h['minimum_J']>0 and h['scaled_generalized_residual_norm_N']>=0,'Invalid iteration history')
        residual=history[-1]['scaled_generalized_residual_norm_N'];require(residual<=threshold*(1+1e-10),'Residual exceeds declared acceptance')
        g=row['geometry'];require(min(g['minimum_volume_ratio'],row['minimum_J'])>=.1,'Endpoint J guard fails')
        require(max(g['fixture_max_error_m'],g['maximum_rigid_pose_map_error_m'],g['maximum_rotation_SO3_error'])<=1e-12,'Exact endpoint constraints fail')
        start=np.asarray(row['handle_centers_start_m']);end=np.asarray(row['handle_centers_end_m']);increment=np.asarray(row['handle_center_step_increment_m'])
        exact(start,lastcenter,'Handle-center discontinuity');close(increment,end-start,1e-12,'Stable center increment differs')
        work,moments=pure.held_force_work_and_moments(F,start,end,increment)
        close(row['held_handle_work_each_J'],work,1e-15,'Handle work differs');close(row['held_handle_work_J'],work.sum(),1e-15,'Core force work differs')
        close(data['handle_work_each_J'][i],work,1e-15,'Archived work differs');close(row['moment_about_world_origin_start_end_Nm'],moments,1e-15,'Moving-origin moments differ');close(data['moment_about_world_origin_start_end_Nm'][i],moments,1e-15,'Archived moments differ')
        targets=pure.fixture_targets(hardware.points,oldspec,intervals[i]);close(row['prescribed_fixture_targets_end_m'],targets[1],1e-12,'Prescribed fixture target differs')
        reaction=np.asarray(row['fixture_reactions_inertia_elastic_drag_minus_external_N']);require(reaction.shape==(len(spec['fixture_nodes']),3),'Reaction shape differs')
        support=np.sum(reaction*(targets[1]-targets[0]));close(row['support_work_J'],support,1e-12,'Support work differs')
        old,new=row['old_invariants'],row['new_invariants']
        if last is not None:require(old==last,'Invariant ledger discontinuity')
        for inv in [old,new]:close(inv['energy_J'],inv['chord_kinetic_energy_J']+inv['elastic_energy_J'],1e-12,'Energy components differ')
        defect=new['energy_J']-old['energy_J']-(work.sum()+support-row['body_viscous_dissipation_J'])
        identity=defect-row['constraint_chord_work_J']+row['implicit_velocity_increment_loss_J']+row['backward_elastic_gradient_defect_J']
        close(identity,row['energy_algebra_identity_residual_J'],1e-12,'Energy identity differs');close(defect,row['energy_balance_time_discretization_defect_J'],1e-12,'Time-discretization defect differs')
        momentum=np.asarray(new['point_linear_momentum_Ns'])-old['point_linear_momentum_Ns']-np.diff(intervals[i])[0]*(F.sum(0)+reaction.sum(0))+row['body_drag_impulse_Ns']
        close(momentum,row['linear_momentum_balance_error_Ns'],1e-12,'Momentum identity differs')
        if i+1 in refined.output_step_indices:
            frame=int(np.flatnonzero(refined.output_step_indices==i+1)[0]);exact(end,data['handle_centers_m'][frame],'Frame/step center differs')
            close([row['minimum_J'],g['minimum_volume_ratio']],np.repeat(geometry['minimum_saved_J_by_frame'][frame],2),1e-10,'Independent sampled J differs')
            close(new['point_linear_momentum_Ns'],P[frame],1e-11,'Independent sampled momentum differs');close(new['chord_kinetic_energy_J'],kinetic[frame],1e-11,'Independent sampled kinetic energy differs');close(new['elastic_energy_J'],elastic[frame],1e-11,'Independent sampled elastic energy differs')
        residuals.append(residual);thresholds.append(threshold);runtimeJ.append(g['minimum_volume_ratio']);work_error.append(float(np.max(abs(work-row['held_handle_work_each_J']))));energy_error.append(abs(identity));momentum_error.append(float(np.linalg.norm(momentum)));last=new;lastcenter=end
    require(summary['trajectory_sha256']==pure.sha(root/'trajectory.npz') and summary['steps_sha256']==pure.sha(root/'steps.jsonl'),'Summary hashes differ')
    require(summary['production_dataset_qualified'] is False and summary['original_gate_requalified'] is False and summary['information_roles']==runner.INFORMATION_ROLES,'Qualification/role overclaim')
    records=[{'path':str(p.relative_to(runner.ROOT)),'sha256':pure.sha(p)} for p in [runner.resolve(protocol_path),runner.checked(protocol['input_design']),root/'trajectory.npz',root/'steps.jsonl',root/'execution.json',root/'summary.json',private/'snapshots.json']]
    return dict(passed=True,cell_id=protocol['cell_id'],frames=61,steps=count,terminal_pid_absent=True,geometry=geometry,
        minimum_runtime_J=min(runtimeJ),maximum_accepted_residual_N=max(residuals),effective_threshold_range_N=[min(thresholds),max(thresholds)],
        maximum_handle_work_error_J=max(work_error),maximum_energy_bookkeeping_identity_J=max(energy_error),maximum_momentum_bookkeeping_error_Ns=max(momentum_error),
        canonical_public_program_exact=True,private_commands_exact_held_repeats=True,full_hardware_and_hypothesis_verified=True,
        public_conditioning_contains_no_material_or_boundary_hypothesis=True,source_snapshot_count=len(snapshots),source_evidence_records=records,
        temporal_accuracy_qualified=False,production_dataset_qualified=False,energy_conservation_claim=False,native_equivalence=False,continuum_accuracy=False,
        limitation='Single engineering smoke; pure saved-frame mechanics and all-step ledger consistency. Per-step gradients/residuals are recorded, not independently re-solved.',
        auditor_source_sha256=pure.sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);parser.add_argument('--protocol-sha256',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    result=audit(args.protocol,args.protocol_sha256);destination=runner.resolve(args.output)
    if destination.exists():raise ValueError('Fresh audit output required')
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['passed','cell_id','frames','steps','minimum_runtime_J','maximum_accepted_residual_N']}))
