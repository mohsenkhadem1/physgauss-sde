"""Separate finite-dynamics exploration with a latent-independent actuator clock.

Public hardware and known held commands are separate from realized material /
support hypotheses. This adapter changes only a private optimizer metric,
never the physical objective. No old gate or production admission is implied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import types

import numpy as np
import torch

try:
    import campaign_fixed_actuator_clock as clocks
    import campaign_full_hardware_context as context
    import campaign_authored_force_case as guards
    from campaign_nonlinear_dynamics import NonlinearDynamics
except ImportError:
    from experiments import campaign_fixed_actuator_clock as clocks
    from experiments import campaign_full_hardware_context as context
    from experiments import campaign_authored_force_case as guards
    from experiments.campaign_nonlinear_dynamics import NonlinearDynamics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
PHYSICAL_CORE_SHA256 = 'da0db7018b4913020bb228c2fc4df3c240e7dd872f17db4868de147eaaf78fa2'
SETTINGS = dict(actuator_hz=120, output_hz=20, seconds=3, frames=61,
    optimizer_metric_scale=1.25, maximum_iterations=250,
    relative_tolerance=1e-7, absolute_tolerance_N=1e-10)
INFORMATION_ROLES = dict(public_hardware='public_known_hardware',
    known_commands='known_actuator_command',
    realized_hypothesis='known_parameter_training_or_diagnostic',
    support_law='candidate_boundary_hypothesis')
LIMITATION = ('New-distribution engineering exploration only. Independent float64 '
    'log-J Neo-Hookean dynamics; no contacts, native equivalence, continuum '
    'accuracy, original-gate passage, calibrated posterior or production admission.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    value = Path(path)
    if value.is_absolute() or '..' in value.parts or str(path) != value.as_posix() or str(path) == '.':
        raise ValueError('Canonical repository-relative path required')
    result = (ROOT/value).resolve()
    if not result.is_relative_to(ROOT.resolve()):
        raise ValueError('Path escapes repository root')
    return result


def checked(record):
    path = resolve(record['path'])
    if sha(path) != record['sha256']:
        raise ValueError('Pinned artifact changed: '+record['path'])
    return path


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def runtime_sources():
    names = ['campaign_hypothesis_dynamics.py', 'campaign_hypothesis_inputs.py',
        'campaign_physical_prior.py', 'campaign_fixed_actuator_clock.py',
        'campaign_authored_force_case.py', 'campaign_nonlinear_dynamics_case.py',
        'campaign_full_hardware_context.py', 'campaign_nonlinear_dynamics.py',
        'campaign_nonlinear_fem.py', 'campaign_linear_fem.py',
        'campaign_rigid_inertia.py', 'campaign_authored_commands.py', 'campaign_exciting_commands.py',
        'isaac_dataset.py', 'boundary_block.py']
    pins = {f'experiments/{name}':sha(ROOT/'experiments'/name) for name in names}
    if pins['experiments/campaign_nonlinear_dynamics.py'] != PHYSICAL_CORE_SHA256:
        raise ValueError('Original physical core changed')
    return pins


def validate_clock(clock, integration_hz, node_count, handle_count):
    if type(clock) is not clocks.FixedActuatorClock:
        raise TypeError('Explicit known FixedActuatorClock required')
    if clock.actuator_hz != 120 or len(clock.endpoint_times_s) != 361:
        raise ValueError('Exactly three seconds on fixed public120Hz required')
    if not np.array_equal(clock.output_times_s, np.arange(61, dtype=np.float64)/20):
        raise ValueError('Exactly61 common20Hz output times required')
    if type(integration_hz) is not int or integration_hz not in (120,240,480):
        raise ValueError('Only predeclared private120/240/480Hz allowed')
    if clock.handle_forces_world_N.shape != (360,handle_count,3) or clock.nodal_forces_world_N.shape not in [(360,0,3),(360,node_count,3)]:
        raise ValueError('Known commands disagree with public hardware dimensions')
    if np.any(clock.handle_torques_world_Nm) or np.any(clock.body_acceleration_world_m_s2):
        raise ValueError('Unmodified core does not implement pure torques or body acceleration')
    return clocks.repeat_for_integration(clock, integration_hz=integration_hz)


def install_metric(model, metric_scale):
    """Instance-only scaled P0 factor; physical step and line search unchanged."""
    if type(metric_scale) is not float or metric_scale not in (1.25,2.):
        raise ValueError('Only prospectively declared metric1.25 or2 allowed')
    original = model.preconditioner
    def scaled(this, state, h):
        return original(state,h)*math.sqrt(metric_scale)
    model.preconditioner = types.MethodType(scaled,model)


def advance(model, state, refined, index):
    """The actual core call consumes repeated bytes, never a sampled waveform."""
    start,end = refined.endpoint_times_s[index:index+2]
    if abs(state.time_s-start) > 1e-12:
        raise ValueError('Measured solver time disagrees with requested private clock')
    nodal = refined.nodal_forces_world_N[index]
    # Empty known nodal capacity means no nodal actuators, exactly physical zero.
    if len(nodal) == 0: nodal = np.zeros(tuple(model.points.shape),dtype=np.float64)
    new,row = model.step(state,1/refined.integration_hz,
        handle_forces=refined.handle_forces_world_N[index],
        nodal_forces=nodal,
        relative_tolerance=1e-7,absolute_tolerance_N=1e-10,maximum_iterations=250)
    if abs(new.time_s-end) > 1e-12:
        raise ValueError('Measured endpoint disagrees with requested private clock')
    return new,row


def build_model(hardware, hypothesis):
    """Only explicit KnownHardware + realized hypothesis; no episode decoding."""
    try:
        import campaign_hypothesis_inputs as inputs
    except ImportError:
        from experiments import campaign_hypothesis_inputs as inputs
    if type(hardware) is not context.KnownHardware:
        raise TypeError('Complete explicit KnownHardware required')
    spec = inputs.validate_realized_hypothesis(hardware,hypothesis)
    handles = hardware.handles
    if len(handles) != 2 or any(len(h.ordered_patch_nodes) != 9 for h in handles):
        raise ValueError('Current exploration requires two complete nine-node handles')
    if any(h.origin_node_id is None or not np.array_equal(h.initial_rotation_body_to_world,np.eye(3)) for h in handles):
        raise ValueError('Current rest-compatible distribution requires exact origin nodes and identity initial rotations')
    material = spec['material']
    nodes = np.asarray(spec['fixture_nodes'],dtype=np.int64)
    target = velocity = None
    if spec['boundary_kind'] == 'moving_patch':
        target = lambda t: hardware.points[nodes]+inputs.support_translation(spec,t)
        velocity = lambda t: np.broadcast_to(inputs.support_velocity(spec,t),(len(nodes),3)).copy()
    model = NonlinearDynamics(hardware.points,hardware.tets,
        material['youngs_modulus_Pa'],material['poissons_ratio'],material['density_kg_m3'],
        drag=material['body_linear_drag_s_inverse'],fixture_nodes=nodes,
        boundary_kind=spec['boundary_kind'],fixture_target=target,fixture_velocity=velocity,
        load_patches=[h.ordered_patch_nodes.tolist() for h in handles],
        load_nodes=[h.origin_node_id for h in handles],handle_masses=[h.mass_kg for h in handles],
        handle_inertias=[h.com_inertia_body_kg_m2 for h in handles],
        handle_com_offsets=np.stack([h.com_offset_body_m for h in handles]))
    return model,spec


def validate_protocol(path, digest):
    path = resolve(path)
    if sha(path) != digest:
        raise ValueError('Protocol digest mismatch')
    protocol = json.loads(path.read_text())
    if (protocol.get('protocol') != 'independent-hypothesis-fixed-clock-smoke-v0.1'
            or protocol.get('status') != 'FROZEN_REVIEWED_SINGLE_TRAIN_SMOKE'
            or protocol.get('split') != 'TRAIN' or protocol.get('settings') != SETTINGS
            or protocol.get('information_roles') != INFORMATION_ROLES
            or any(type(protocol['settings'][key]) is not type(value) for key,value in SETTINGS.items())):
        raise ValueError('Exactly frozen new one-TRAIN-smoke scope required')
    if type(protocol.get('integration_hz')) is not int or protocol['integration_hz'] not in (120,240,480):
        raise ValueError('Undeclared private integration rate')
    if protocol.get('source_sha256') != runtime_sources():
        raise ValueError('Frozen runtime source/dependency map mismatch')
    for key in ['public_hardware','realized_hypothesis','known_commands']:
        checked(protocol[key])
    resolve(protocol['output'])
    return protocol


def run(protocol_path, protocol_sha256):
    protocol = validate_protocol(protocol_path,protocol_sha256)
    if any(os.environ.get(k) != '2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):
        raise ValueError('Exactly two CPU threads required')
    output = resolve(protocol['output'])
    if output.exists():
        raise ValueError('Fresh output required; failures are never overwritten')
    try:
        import campaign_hypothesis_inputs as inputs
    except ImportError:
        from experiments import campaign_hypothesis_inputs as inputs
    public = json.loads(checked(protocol['public_hardware']).read_text())
    hypothesis = json.loads(checked(protocol['realized_hypothesis']).read_text())
    hardware = inputs.known_hardware_from_public(public)
    with np.load(checked(protocol['known_commands']),allow_pickle=False) as raw:
        clock = clocks.FixedActuatorClock(actuator_hz=120,information_role=clocks.KNOWN_ROLE,
            **{key:raw[key].copy() for key in ['endpoint_times_s','output_times_s',
                'handle_forces_world_N','handle_torques_world_Nm','nodal_forces_world_N','body_acceleration_world_m_s2']})
    refined = validate_clock(clock,protocol['integration_hz'],len(hardware.points),len(hardware.handles))
    # Input validation precedes output creation and all physical construction.
    inputs.validate_realized_hypothesis(hardware,hypothesis)
    output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    status = dict(terminal=False,completed=False,pid=os.getpid(),completed_steps=0,
        planned_steps=len(refined.parent_command_indices),cpu_threads=2,gpu_used=False)
    write(output/'execution.json',status)
    state = last_valid = row = request = None
    frames = []; intervals = []; work_rows = []; moment_rows = []
    try:
        private = output/'private-provenance'; private.mkdir()
        pins = {**protocol['source_sha256'],str(protocol_path):protocol_sha256,
            **{protocol[key]['path']:protocol[key]['sha256'] for key in ['public_hardware','realized_hypothesis','known_commands']}}
        snapshots = []
        for i,(rel,digest) in enumerate(pins.items()):
            saved = f'{i:02d}-'+Path(rel).name
            (private/saved).write_bytes(resolve(rel).read_bytes())
            if sha(private/saved) != digest: raise ValueError('Snapshot hash mismatch')
            snapshots.append(dict(original_path=rel,saved_path=saved,sha256=digest))
        write(private/'snapshots.json',snapshots)
        write(output/'protocol.json',protocol)
        write(output/'public-command-manifest.json',clock.public_manifest())
        write(output/'public-hardware.json',public)
        write(private/'realized-hypothesis.json',hypothesis)
        np.savez_compressed(private/'static-reference.npz',rest_points_m=hardware.points,
            tetrahedra=hardware.tets,surface_faces=hardware.faces)
        np.savez_compressed(output/'public-commands.npz',**clock.public_features())
        np.savez_compressed(private/'integration-commands.npz',**refined.private_features())
        torch.set_num_threads(2)
        model,spec = build_model(hardware,hypothesis)
        state = model.initial_state(handle_rotations=np.stack([h.initial_rotation_body_to_world for h in hardware.handles]))
        P0 = model.preconditioner(state,1/refined.integration_hz).clone()
        install_metric(model,protocol['settings']['optimizer_metric_scale'])
        guards.endpoint_guard(model,state,guards.endpoint_metrics(model,state))
        last_valid = state
        def capture():
            frames.append(dict(time_s=state.time_s,**{key:getattr(state,key).numpy().copy() for key in [
                'positions','velocities','auxiliary_positions','auxiliary_velocities',
                'handle_centers','handle_rotations','reduced_position_coordinates']}))
        # Core's initial state has no reduced coordinates; store exact zero coordinates.
        state.reduced_position_coordinates = model.coordinates(state)
        capture()
        output_indices = set(refined.output_step_indices.tolist())
        with (output/'steps.jsonl').open('x') as stream:
            for index in range(len(refined.parent_command_indices)):
                previous = state; row = None
                F = refined.handle_forces_world_N[index]
                old_q = model.coordinates(previous)
                request = dict(step_index=index,parent_public_command_index=int(refined.parent_command_indices[index]),
                    compiled_interval_s=refined.endpoint_times_s[index:index+2].tolist(),
                    actual_state_start_s=previous.time_s,requested_dt_s=1/refined.integration_hz,
                    held_handle_forces_world_N=F.tolist(),
                    held_nodal_forces_world_N=refined.nodal_forces_world_N[index].tolist())
                wall = time.monotonic(); state,row = advance(model,previous,refined,index)
                metrics = guards.endpoint_metrics(model,state,previous)
                increment = np.stack([(state.reduced_position_coordinates[dofs[:3]]-old_q[dofs[:3]]).numpy() for dofs in model.handle_dofs])
                start,end = previous.handle_centers.numpy(),state.handle_centers.numpy()
                work = np.einsum('hc,hc->h',F,increment)
                moments = np.stack([np.cross(start,F),np.cross(end,F)])
                row.update(**request,actual_interval_s=[previous.time_s,state.time_s],geometry=metrics,
                    handle_centers_start_m=start.tolist(),handle_centers_end_m=end.tolist(),
                    handle_center_step_increment_m=increment.tolist(),held_handle_work_each_J=work.tolist(),
                    moment_about_world_origin_start_end_Nm=moments.tolist(),
                    prescribed_fixture_targets_end_m=model.targets(state.time_s).numpy().tolist(),
                    step_wall_seconds=time.monotonic()-wall)
                guards.endpoint_guard(model,state,metrics,row)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                last_valid = state; intervals.append(row['actual_interval_s']);work_rows.append(work);moment_rows.append(moments)
                status['completed_steps'] = index+1
                if index+1 in output_indices: capture()
                if index == 0 or (index+1) % 30 == 0:
                    if runtime_sources() != protocol['source_sha256']: raise ValueError('Frozen runtime source changed')
                    status.update(elapsed_s=time.monotonic()-started,last_iterations=len(row['nonlinear_history'])-1,
                        last_residual_N=row['nonlinear_history'][-1]['scaled_generalized_residual_norm_N'])
                    write(output/'execution.json',status);print(json.dumps(status),flush=True)
        times = np.array([f['time_s'] for f in frames])
        if len(frames) != 61 or np.max(abs(times-clock.output_times_s)) > 1e-12:
            raise ValueError('Incomplete actual61frame output clock')
        np.savez_compressed(output/'trajectory.npz',times_s=clock.output_times_s,actual_state_times_s=times,
            positions_m=[f['positions'] for f in frames],velocities_point_chord_m_s=[f['velocities'] for f in frames],
            auxiliary_positions_m=[f['auxiliary_positions'] for f in frames],auxiliary_velocities_point_chord_m_s=[f['auxiliary_velocities'] for f in frames],
            handle_centers_m=[f['handle_centers'] for f in frames],handle_rotations=[f['handle_rotations'] for f in frames],
            reduced_position_coordinates=[f['reduced_position_coordinates'] for f in frames],
            force_interval_actual_times_s=intervals,force_step_commands_N=refined.handle_forces_world_N,
            handle_work_each_J=work_rows,moment_about_world_origin_start_end_Nm=moment_rows)
        summary = dict(completed=True,frames=61,steps=status['completed_steps'],elapsed_s=time.monotonic()-started,
            limitation=LIMITATION,production_dataset_qualified=False,original_gate_requalified=False,
            backend='independent_float64_FEM_physics_no_rendering',information_roles=INFORMATION_ROLES,
            empty_public_nodal_capacity_means='No nodal actuators; exact zero[N,3] allocated only at core call.',
            public_command_manifest=clock.public_manifest(),private_integration_hz=refined.integration_hz,
            optimizer_metric_scale=protocol['settings']['optimizer_metric_scale'],original_P0_factor_sha256=hashlib.sha256(P0.numpy().tobytes()).hexdigest(),
            trajectory_sha256=sha(output/'trajectory.npz'),steps_sha256=sha(output/'steps.jsonl'))
        write(output/'summary.json',summary)
        status.update(terminal=True,completed=True,elapsed_s=summary['elapsed_s']);write(output/'execution.json',status)
        return summary
    except BaseException as error:
        if last_valid is not None: guards.save_state(output/'last-accepted-state.npz',last_valid)
        if state is not None: guards.save_state(output/'failure-endpoint-state.npz',state)
        if frames:
            np.savez_compressed(output/'partial-trajectory.npz',**{key:np.array([f[key] for f in frames]) for key in frames[0]})
        if row is not None: write(output/'failure-endpoint-ledger.json',guards.failure_safe(row))
        write(output/'failure.json',dict(exception=type(error).__name__,message=str(error),
            completed_steps=status['completed_steps'],step_request=request,
            core_failure=guards.failure_safe(guards.capture_traceback_failure(error,output)),
            unaccepted_endpoint_semantics=True,limitation=LIMITATION))
        status.update(terminal=True,completed=False,elapsed_s=time.monotonic()-started);write(output/'execution.json',status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--protocol-sha256',required=True)
    args = parser.parse_args();run(args.protocol,args.protocol_sha256)
