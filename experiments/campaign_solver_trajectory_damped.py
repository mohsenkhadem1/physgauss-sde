"""One fresh-rest optimizer-only trajectory diagnostic; no old gate relabeling.

Reuses frozen physical/context/command/endpoint helpers. New protocol guards
this one cell; the old seal establishes lineage only, not new authorization.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

try:
    import campaign_solver_step_damped as damped
except ImportError:
    from experiments import campaign_solver_step_damped as damped

abc=damped.abc
old=abc.old
ROOT=abc.ROOT
SOURCE=Path(__file__).resolve()
OLD_ID='t0000__twist_couple__temporal__hz120'
NEW_ID='diagnostic_t0000_twist120_metric2'
LIMITS={'positions_m':2e-7,'point_chord_velocity_m_s':2.4e-5,'rotation_entry_length_m':2e-7,'rotation_length_scale_m':.24}


def validate_protocol(protocol_path,protocol_sha):
    """Pure provenance/settings guard; also usable for existing-output audit."""
    path=abc.resolve(protocol_path)
    if abc.sha(path)!=protocol_sha:raise ValueError('New trajectory protocol digest mismatch')
    protocol=abc.read(path)
    if (protocol['protocol']!='fresh-rest-t0000-twist120-metric2-diagnostic-v0.1'
            or protocol['status']!='FROZEN_REVIEWED_ONE_TRAJECTORY_DIAGNOSTIC'
            or protocol['original_cell_id']!=OLD_ID or protocol['new_cell_id']!=NEW_ID
            or protocol['metric_scale']!=2. or protocol['prefix_limits']!=LIMITS):
        raise ValueError('Only exact newly reviewed optimizer trajectory scope allowed')
    original_seal=protocol['original_seal'];seal_path=old.checked_record(original_seal)
    _,_,cell,prior_pins=old.validate_seal(original_seal['path'],OLD_ID)
    if (cell['case'],cell['response'],cell['hz'],cell['output_hz'],cell['maximum_iterations'],cell['relative_tolerance'],cell['absolute_tolerance_N'],cell['translation_m'])!=('t0000','twist_couple',120,20,250,1e-7,1e-10,[0,0,0]):
        raise ValueError('Original exact failed physical settings changed')
    pins=protocol['source_and_input_sha256']
    required={*prior_pins,SOURCE.relative_to(ROOT).as_posix(),'experiments/campaign_solver_step_damped.py',
        'experiments/campaign_solver_step_diagnosis.py',
        *[abc.CASE+'/'+name for name in ['partial-trajectory.npz','last-accepted-state.npz','steps.jsonl','compiled-commands.npz','failure.json']]}
    if not required.issubset(pins):raise ValueError('New scope lacks complete source/prefix/auditor pins')
    for rel,digest in pins.items():
        if abc.sha(abc.resolve(rel))!=digest:raise ValueError('Frozen trajectory source/input changed: '+rel)
    if any(pins[rel]!=value for rel,value in prior_pins.items()):raise ValueError('Original C18 lineage differs')
    terminal=abc.read(old.checked_record(protocol['terminal_guard']))
    if terminal['stage_terminal'] is not True or terminal['all_launched_worker_PIDs_missing'] is not True:raise ValueError('Original terminal boundary required')
    pids=[terminal['supervisor_pid']]+[r['pid'] for r in terminal['workers'] if r.get('pid') is not None]
    if len(pids)!=28 or any(Path('/proc',str(pid)).exists() for pid in pids):raise ValueError('Original supervisor/worker still live')
    new_cell={**cell,'id':NEW_ID,'output':protocol['fresh_output'],'stage':'new_optimizer_trajectory_diagnostic'}
    return protocol,cell,new_cell,{**prior_pins,**pins,protocol_path:protocol_sha}


def prefix_error(positions,velocities,rotations,old_positions,old_velocities,old_rotations):
    errors={'positions_max_abs_m':float(np.max(abs(positions-old_positions))),
        'point_chord_velocity_max_abs_m_s':float(np.max(abs(velocities-old_velocities))),
        'rotation_entry_max_abs':float(np.max(abs(rotations-old_rotations)))}
    errors['rotation_scaled_max_abs_m']=.24*errors['rotation_entry_max_abs']
    errors['passed']=bool(errors['positions_max_abs_m']<=2e-7 and errors['point_chord_velocity_max_abs_m_s']<=2.4e-5 and errors['rotation_scaled_max_abs_m']<=2e-7)
    return errors


def run(protocol_path,protocol_sha):
    protocol,source_cell,cell,pins=validate_protocol(protocol_path,protocol_sha)
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('Exactly two CPU threads required')
    output=abc.resolve(cell['output'])
    if output.exists():raise ValueError('Fresh new diagnostic output required')
    case,spec,arrays,_,lineage=old.load_case_inputs('t0000',None,abc.resolve('configs/gaussian_independent_diagnostic_inputs_v01.json'))
    points=arrays['rest_points_m'].astype(np.float64);hardware=old.hardware_descriptor(points,spec);compiled=old.compile_cell(spec,cell)
    with np.load(abc.resolve(abc.CASE+'/compiled-commands.npz')) as raw:
        if any(raw[key].dtype!=value.dtype or raw[key].tobytes()!=value.tobytes() for key,value in compiled.items() if isinstance(value,np.ndarray)):raise ValueError('Exact original compiled force/clock bytes changed')
    with np.load(abc.resolve(abc.CASE+'/partial-trajectory.npz')) as raw:prefix={k:raw[k].copy() for k in raw.files}
    with np.load(abc.resolve(abc.CASE+'/last-accepted-state.npz')) as raw:last={k:raw[k].copy() for k in raw.files}
    prior_rows=[json.loads(line) for line in abc.resolve(abc.CASE+'/steps.jsonl').read_text().splitlines()]
    if len(prefix['actual_state_times_s'])!=41 or len(prior_rows)!=240:raise ValueError('Complete original accepted prefix required')
    known,query,boundary,public,conditioned=old.make_contexts(case,spec,arrays,hardware,compiled,np.zeros(3));old.validate_supported_context(known,query,boundary,spec)
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();torch.set_num_threads(2)
    status={'terminal':False,'completed':False,'pid':os.getpid(),'completed_steps':0,'planned_steps':360,'cpu_threads':2,'gpu_used':False}
    old.write(output/'execution.json',status)
    state=last_valid=model=None;row=request=None;frames=[];intervals=[];works=[];moments=[];prefix_checks=[]
    try:
        private=output/'private-provenance';private.mkdir()
        for index,(rel,digest) in enumerate(pins.items()):
            target=private/(str(index).zfill(2)+'-'+Path(rel).name);target.write_bytes(abc.resolve(rel).read_bytes())
            if abc.sha(target)!=digest:raise ValueError('New diagnostic source snapshot mismatch')
        old.write(private/'pins.json',pins);old.write(private/'snapshots.json',[{'original_path':rel,'saved_path':str(i).zfill(2)+'-'+Path(rel).name,'sha256':digest} for i,(rel,digest) in enumerate(pins.items())])
        np.savez_compressed(private/'static-reference.npz',**arrays);old.write(private/'physical-inputs.json',{'case':case,'spec':spec,'hardware':hardware,'static_lineage':lineage})
        old.write(output/'protocol.json',{'cell':cell,'seconds':3.,'output_hz':20,'frames':61,'admission':old.ENGINEERING_ONLY,
            'physics':'unchanged independent float64 log-J Neo-Hookean, body-onlydrag,zerogravity/nocontact',
            'source_input_sha256':pins,'compiled_force_metadata':compiled['metadata'],'native_equivalence':False,
            'continuum_accuracy_established':False,'native_elasticity_damping_reproduced':False,'actual_force_envelope_qualified':False,
            'new_optimizer_diagnostic':True,'original_seal_lineage_only':True,'optimizer_metric_scale':2.,'surface_crossing_diagnostic':'not_evaluated'})
        old.save_context(output,public,conditioned);np.savez_compressed(output/'compiled-commands.npz',**{k:v for k,v in compiled.items() if isinstance(v,np.ndarray)})
        physical={'case':case,'spec':spec,'hardware':hardware};model=abc.assemble_model(physical,arrays,[0,0,0])
        state=model.initial_state();P0=model.preconditioner(state,1/120).clone();damped.install_half_direction(model)
        initial_metrics=old.endpoint_metrics(model,state);old.endpoint_guard(model,state,initial_metrics);last_valid=state
        def capture():
            frame={'time':state.time_s,**{key:getattr(state,key).numpy().copy() for key in ['positions','velocities','auxiliary_positions','auxiliary_velocities','handle_centers','handle_rotations']}}
            frames.append(frame);index=len(frames)-1
            if index<=40:
                if state.time_s!=prefix['actual_state_times_s'][index]:raise ValueError('Exact original prefix output clock differs')
                check=prefix_error(frame['positions'],frame['velocities'],frame['handle_rotations'],prefix['positions'][index],prefix['velocities'][index],prefix['handle_rotations'][index]);check['frame_index']=index;prefix_checks.append(check)
                if not check['passed']:raise ValueError('Predeclared original prefix overlap bound failed')
        capture()
        with (output/'steps.jsonl').open('x') as stream:
            for index,force in enumerate(compiled['held_handle_forces_world_N']):
                previous=state;row=None;old_coordinates=model.coordinates(state)
                request={'step_index':index,'compiled_interval_s':compiled['force_interval_times_s'][index].tolist(),'compiled_midpoint_s':float(compiled['force_midpoints_s'][index]),'actual_state_start_s':state.time_s,'requested_dt_s':1/120,'held_handle_forces_world_N':force.tolist()}
                wall=time.monotonic();state,row=model.step(state,1/120,handle_forces=force,relative_tolerance=1e-7,absolute_tolerance_N=1e-10,maximum_iterations=250)
                metric=old.endpoint_metrics(model,state,previous);increment=np.array([(state.reduced_position_coordinates[dofs[:3]]-old_coordinates[dofs[:3]]).numpy() for dofs in model.handle_dofs])
                start=previous.handle_centers.numpy();end=state.handle_centers.numpy();work=np.einsum('hc,hc->h',force,increment);moment=np.stack([np.cross(start,force),np.cross(end,force)])
                row.pop('held_nodal_forces_N');row.update(**request,actual_interval_s=[previous.time_s,state.time_s],actual_state_midpoint_s=(previous.time_s+state.time_s)/2,geometry=metric,handle_centers_start_m=start.tolist(),handle_centers_end_m=end.tolist(),handle_center_step_increment_m=increment.tolist(),held_handle_work_each_J=work.tolist(),moment_about_world_origin_start_end_Nm=moment.tolist(),step_wall_seconds=time.monotonic()-wall)
                old.endpoint_guard(model,state,metric,row)
                if not np.array_equal(force.sum(0),np.zeros(3)):raise ValueError('Twist couple lost exact zero resultant')
                if np.max(abs(np.array(row['actual_interval_s'])-compiled['force_interval_times_s'][index]))>1e-12:raise ValueError('Actual versus compiled state clock mismatch')
                if index<240:
                    original=prior_rows[index]
                    if row['start_time_s']!=original['start_time_s'] or row['end_time_s']!=original['end_time_s'] or np.asarray(row['held_handle_forces_N']).tobytes()!=np.asarray(original['held_handle_forces_N']).tobytes():raise ValueError('Exact original prefix command/solverclock differs')
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();intervals.append(row['actual_interval_s']);works.append(work);moments.append(moment);last_valid=state;status['completed_steps']=index+1
                if (index+1)%6==0:capture()
                if index+1==240:
                    last_check=prefix_error(state.positions.numpy(),state.velocities.numpy(),state.handle_rotations.numpy(),last['positions_m'],last['velocities_point_chord_m_s'],last['handle_rotations']);last_check['role']='last_accepted_step240';prefix_checks.append(last_check)
                    if state.time_s!=float(last['time_s']) or not last_check['passed']:raise ValueError('Predeclared last accepted state overlap bound failed')
                if index==0 or (index+1)%30==0:
                    if any(abc.sha(abc.resolve(p))!=digest for p,digest in pins.items()):raise ValueError('Immutable source/input changed during diagnostic')
                    status.update(elapsed_s=time.monotonic()-started,last_iterations=len(row['nonlinear_history'])-1,last_residual_N=row['nonlinear_history'][-1]['scaled_generalized_residual_norm_N']);old.write(output/'execution.json',status);print(json.dumps(status),flush=True)
        times=np.array([f['time'] for f in frames]);nominal=compiled['requested_output_times_s']
        if len(times)!=61 or np.max(abs(times-nominal))>1e-12:raise ValueError('Incomplete actual61-frame trajectory')
        np.savez_compressed(output/'trajectory.npz',times_s=nominal,actual_state_times_s=times,positions_m=[f['positions'] for f in frames],velocities_point_chord_m_s=[f['velocities'] for f in frames],auxiliary_positions_m=[f['auxiliary_positions'] for f in frames],auxiliary_velocities_point_chord_m_s=[f['auxiliary_velocities'] for f in frames],handle_centers_m=[f['handle_centers'] for f in frames],handle_rotations=[f['handle_rotations'] for f in frames],force_interval_actual_times_s=intervals,force_midpoints_compiled_s=compiled['force_midpoints_s'],force_step_commands_N=compiled['held_handle_forces_world_N'],handle_work_each_J=works,moment_about_world_origin_start_end_Nm=moments)
        old.write(output/'prefix-overlap.json',{'limits':LIMITS,'checks':prefix_checks,'passed':len(prefix_checks)==42 and all(r['passed'] for r in prefix_checks),'original_failed_cell_remains_failed':True,'exact_commands_and_clocks':True})
        dt=np.diff(np.array(intervals),axis=1)[:,0];force=compiled['held_handle_forces_world_N']
        summary={'completed':True,'cell_id':NEW_ID,'frames':61,'elapsed_s':time.monotonic()-started,'admission':old.ENGINEERING_ONLY,'actual_force_envelope_qualified':False,'native_reference_qualified':False,'continuum_accuracy_established':False,'prediction_qualified':False,'trajectory_sha256':abc.sha(output/'trajectory.npz'),'steps_sha256':abc.sha(output/'steps.jsonl'),'actual_clock_impulse_Ns':np.einsum('t,thc->hc',dt,force).tolist(),'actual_clock_absolute_impulse_Ns':np.einsum('t,thc->hc',dt,np.abs(force)).tolist(),'actual_clock_first_time_moment_Ns2':np.einsum('t,thc->hc',dt*np.array(intervals).mean(1),force).tolist(),'handle_work_each_total_J':np.sum(works,axis=0).tolist(),'optimizer_metric_scale':2.,'original_failed_cell_requalified':False,'full_new_operating_envelope_qualified':False}
        old.write(output/'summary.json',summary);status.update(terminal=True,completed=True,elapsed_s=summary['elapsed_s']);old.write(output/'execution.json',status);return summary
    except BaseException as error:
        if last_valid is not None:old.save_state(output/'last-accepted-state.npz',last_valid)
        if state is not None:old.save_state(output/'failure-endpoint-state.npz',state)
        if frames:np.savez_compressed(output/'partial-trajectory.npz',actual_state_times_s=[f['time'] for f in frames],**{key:np.array([f[key] for f in frames]) for key in frames[0] if key!='time'})
        if row is not None:old.write(output/'failure-endpoint-ledger.json',old.failure_safe(row))
        old.write(output/'prefix-overlap.json',{'limits':LIMITS,'checks':prefix_checks,'passed':False})
        old.write(output/'failure.json',{'exception':type(error).__name__,'message':str(error),'completed_steps':status['completed_steps'],'step_request':request,'core_failure':old.failure_safe(old.capture_traceback_failure(error,output)),'original_failed_cell_requalified':False})
        status.update(terminal=True,completed=False,elapsed_s=time.monotonic()-started);old.write(output/'execution.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True)
    a=p.parse_args();run(a.protocol,a.protocol_sha256)
