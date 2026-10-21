"""Fresh-output TRAIN-only runner for the independent dynamic reference gate.

This file never qualifies a reference from one run. It opens only the four
declared TRAIN recipes/reference meshes, not any predictive held-out response.
No native engine, sensor rendering or learned fitting is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

try:
    from campaign_nonlinear_dynamics import NonlinearDynamics
except ImportError:
    from experiments.campaign_nonlinear_dynamics import NonlinearDynamics


GATE_PATH = Path('configs/gaussian_campaign_independent_dynamic_gate_v01.json')
GATE_SHA256 = '962457955d72a97dd0d8e48a41aff0c6ead821931031343e081cf837c561b2a0'
CASE_BOUNDARIES = dict(zip(['t0000','t0001','t0002','t0003'],
    ['point_pin','fixed_patch','two_patches','moving_patch']))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_request(gate, episode, hz, response, relative_tolerance, translation,
                     temporal_extension_metadata=None):
    if gate['protocol'] != 'independent-finite-strain-dynamic-reference-v0.1':
        raise ValueError('Frozen independent dynamic gate required')
    if episode not in CASE_BOUNDARIES or episode not in gate['diagnostic_cases']:
        raise ValueError('Only the four declared TRAIN cases are allowed')
    if hz not in gate['temporal_refinement_hz']:
        if (temporal_extension_metadata is None
                or hz != 960 or episode != 't0003'
                or temporal_extension_metadata.get('physics_hz') != hz
                or temporal_extension_metadata.get('episode_id') != episode
                or temporal_extension_metadata.get('response') != response
                or temporal_extension_metadata.get('initial_temporal_failure_preserved') is not True):
            raise ValueError('Initial rates or the exactly validated conditional 960 Hz study required')
    if response not in gate['responses'] or relative_tolerance not in gate['nonlinear_relative_residual_tolerances']:
        raise ValueError('Predeclared response and nonlinear tolerance required')
    translation = np.asarray(translation,dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError('Finite declared translation required')
    if not any(np.array_equal(translation,np.asarray(t)) for t in [[0.,0.,0.],gate['coordinate_translation_m']]):
        raise ValueError('Only untranslated or predeclared translated coordinates allowed')
    if gate['seconds'] != 3. or gate['output_hz'] != 10 or hz % gate['output_hz']:
        raise ValueError('Frozen 3 second / 31 common-frame design required')
    return translation


def boundary_motion(spec, time_s):
    """Known sinusoid-squared translated support and its exact initial derivative."""
    seconds = float(spec['seconds']); t = float(time_s)
    if not math.isfinite(t) or seconds <= 0:
        raise ValueError('Finite support clock and positive duration required')
    phase = np.clip(t/seconds,0.,1.)
    amplitude = float(spec['motion_amplitude_m'])
    direction = np.asarray(spec['motion_direction_world'],dtype=float)
    shift = amplitude*np.sin(np.pi*phase)**2*direction
    velocity = amplitude*np.pi/seconds*np.sin(2*np.pi*phase)*direction if 0<t<seconds else np.zeros(3)
    return shift,velocity


def force_command(gate, response, midpoint_s):
    peak = np.array([gate['force']['peak_world_force_N_handle0'],gate['force']['peak_world_force_N_handle1']],dtype=np.float64)
    if response == 'zero_external_force':
        return np.zeros_like(peak)
    if response != 'diagonal_pulse_handle0':
        raise ValueError('Unrecognized predeclared force response')
    phase = (float(midpoint_s)/gate['seconds']-.1)/.45
    return peak*np.sin(np.pi*phase)**2 if 0<phase<1 else np.zeros_like(peak)


def hardware_descriptor(points, spec):
    if len(spec['load_nodes']) != 2 or len(spec['load_patches']) != 2:
        raise ValueError('Both authored rigid handles are required')
    mass = float(spec['handle_mass_kg_each']); handles = []
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('Positive authored rigid mass required')
    for node,patch in zip(spec['load_nodes'],spec['load_patches']):
        if len(patch) != 9 or len(set(patch)) != 9 or node not in patch:
            raise ValueError('Complete ordered nine-node patch required')
        local = points[patch]-points[node]
        # Same authored hardware formula as isaac_batch_dataset_worker.py;
        # the independent kinetic representation does not add another body.
        inertia = mass*np.mean(np.sum(local**2,axis=1)[:,None]-local**2,axis=0)
        inertia = np.maximum(inertia,.4*mass*.003**2)
        handles.append({'reference_origin_node':int(node),'reference_origin_m':points[node].tolist(),
            'ordered_patch_nodes':list(map(int,patch)),'ordered_reference_local_offsets_m':local.tolist(),
            'mass_kg':mass,'COM_offset_m':[0.,0.,0.],
            'full_COM_inertia_kg_m2':np.diag(inertia).tolist(),
            'initial_rotation_matrix':np.eye(3).tolist()})
    return handles


def load_case_inputs(episode, dataset, inputs_json=None):
    """Regenerate public static fixtures or read original TRAIN static files.

    Original archive-file hashes in the fixture are lineage only: this route
    independently verifies canonical arrays and regenerated specs, not bytes
    of an unavailable original archive. No motion/observation array is read.
    """
    if episode not in CASE_BOUNDARIES:
        raise ValueError('Only declared TRAIN static inputs may be opened')
    if inputs_json is None:
        directory=Path(dataset)/'episodes'/episode
        inputs=[directory/name for name in ['reference.npz','truth/recipe.json','truth/boundaries.json']]
        case=json.loads(inputs[1].read_text());spec=json.loads(inputs[2].read_text())
        with np.load(inputs[0]) as archive:
            arrays={name:archive[name].copy() for name in ['rest_points_m','tetrahedra','surface_faces']}
        return case,spec,arrays,inputs,{'mode':'original TRAIN static files; whole-file hashes verified'}
    path=Path(inputs_json);fixture=json.loads(path.read_text())
    cases=fixture['cases']
    if len(cases)!=4 or {c['episode_id'] for c in cases}!=set(CASE_BOUNDARIES):
        raise ValueError('Public static fixture must contain exactly the four declared TRAIN cases')
    selected=next(c for c in cases if c['episode_id']==episode)
    case=selected['recipe'];spec=selected['boundaries']
    if any(c['recipe']['split']!='train' or c['recipe']['episode_id']!=c['episode_id'] for c in cases):
        raise ValueError('Public diagnostic fixture contains a non-TRAIN identity')
    geometry_path=Path(__file__).with_name('isaac_dataset.py')
    block_path=Path(__file__).with_name('boundary_block.py')
    if sha(geometry_path)!=fixture['geometry_source_sha256'] or sha(block_path)!=fixture['block_mesh_source_sha256']:
        raise ValueError('Frozen public geometry source hash mismatch')
    sys.path.insert(0,str(geometry_path.parent))
    try:
        import isaac_dataset as static_geometry
    finally:
        sys.path.pop(0)
    if sha(static_geometry.__file__)!=fixture['geometry_source_sha256']:
        raise ValueError('Imported geometry module does not match pinned source')
    if static_geometry.digest_json({k:v for k,v in case.items() if k!='recipe_sha256'})!=case['recipe_sha256']:
        raise ValueError('Public TRAIN recipe digest mismatch')
    points,tets,faces,generated_spec=static_geometry.geometry(case)
    if generated_spec!=spec:
        raise ValueError('Regenerated boundary specification differs from pinned public fixture')
    arrays=dict(rest_points_m=points,tetrahedra=tets,surface_faces=faces)
    if {row['key'] for row in selected['reference_static_arrays']}!=set(arrays):
        raise ValueError('Incomplete canonical static-array fixture')
    for row in selected['reference_static_arrays']:
        array=np.ascontiguousarray(arrays[row['key']])
        if array.dtype!=np.dtype(row['dtype']) or list(array.shape)!=row['shape']:
            raise ValueError('Regenerated static array dtype/shape mismatch: '+row['key'])
        if hashlib.sha256(array.tobytes(order='C')).hexdigest()!=row['sha256']:
            raise ValueError('Regenerated canonical static array hash mismatch: '+row['key'])
    return case,spec,arrays,[path,geometry_path,block_path],{
        'mode':'public privileged TRAIN static-input fixture; no bulk data needed',
        'canonical_static_arrays_verified':selected['reference_static_arrays'],
        'boundary_spec_exactly_regenerated':True,
        'original_archive_file_hashes_lineage_only':selected['original_input_files_sha256']}


def geometry_metrics(model, state):
    displacement = state.positions-model.points
    ids = torch.as_tensor(model.tets,dtype=torch.long)
    G = torch.einsum('eic,eid->ecd',displacement[ids],model.gradients)
    F = G+torch.eye(3,dtype=torch.float64)
    green = .5*(F.transpose(-1,-2)@F-torch.eye(3,dtype=torch.float64))
    volume = torch.linalg.det(F)
    rigid_error = max((float((torch.cdist(state.positions[p],state.positions[p])-
        torch.cdist(model.points[p],model.points[p])).abs().max()) for p in model.patches),default=0.)
    fixture_error = float((state.positions[model.fixture_nodes]-model.targets(state.time_s)).abs().max()) if len(model.fixture_nodes) else 0.
    return {'coordinate_rms_from_reference_m':float(displacement.square().mean().sqrt()),
        'maximum_nodal_displacement_m':float(torch.linalg.vector_norm(displacement,dim=-1).max()),
        'maximum_point_chord_speed_m_s':float(torch.linalg.vector_norm(state.velocities,dim=-1).max()),
        'minimum_volume_ratio':float(volume.min()),'maximum_volume_ratio':float(volume.max()),
        'maximum_absolute_Green_principal_strain':float(torch.linalg.eigvalsh(green).abs().max()),
        'maximum_rigid_patch_distance_error_m':rigid_error,'fixture_max_error_m':fixture_error}


def save_state(path, state):
    np.savez_compressed(path,time_s=state.time_s,positions_m=state.positions.numpy(),
        velocities_point_chord_m_s=state.velocities.numpy(),auxiliary_positions_m=state.auxiliary_positions.numpy(),
        auxiliary_velocities_point_chord_m_s=state.auxiliary_velocities.numpy(),
        handle_centers_m=state.handle_centers.numpy(),handle_rotations=state.handle_rotations.numpy(),
        reduced_position_coordinates=state.reduced_position_coordinates.numpy() if state.reduced_position_coordinates is not None else np.array([]))


def run(output, *, episode='t0001', hz=120, response='diagonal_pulse_handle0',
        relative_tolerance=1e-7, translation=(0.,0.,0.), dataset='runs/gaussian-boundary-neural-v01',
        absolute_tolerance_N=1e-10, maximum_iterations=250, gate_path=GATE_PATH,inputs_json=None,
        supplemental_floor_protocol=None,temporal_extension_protocol=None):
    torch.set_num_threads(2)
    gate_path = Path(gate_path)
    if sha(gate_path) != GATE_SHA256:
        raise ValueError('Predeclared gate hash changed; no implicit protocol replacement')
    gate = json.loads(gate_path.read_text())
    if supplemental_floor_protocol is not None and temporal_extension_protocol is not None:
        raise ValueError('Floor and temporal studies are separately declared; no combined request')
    temporal_metadata = None
    temporal_dependencies = []
    if temporal_extension_protocol is not None:
        try:
            from campaign_dynamic_temporal_protocol import validate_temporal_extension
        except ImportError:
            from experiments.campaign_dynamic_temporal_protocol import validate_temporal_extension
        temporal_metadata = validate_temporal_extension(temporal_extension_protocol,
            episode=episode,hz=hz,response=response,relative_tolerance=relative_tolerance,
            absolute_tolerance_N=absolute_tolerance_N,maximum_iterations=maximum_iterations,
            translation=translation,inputs_json=inputs_json,
            core_path=Path(__file__).with_name('campaign_nonlinear_dynamics.py'),gate_path=gate_path)
        temporal_dependencies = [Path(temporal_extension_protocol),
            Path(__file__).with_name('campaign_dynamic_temporal_protocol.py'),
            Path(temporal_metadata['trigger_audit_path'])]
    translation = validate_request(gate,episode,hz,response,relative_tolerance,translation,temporal_metadata)
    supplemental_metadata = None
    supplemental_dependencies = []
    if supplemental_floor_protocol is None:
        if not math.isfinite(absolute_tolerance_N) or absolute_tolerance_N != 1e-10 or maximum_iterations != 250:
            raise ValueError('Frozen baseline requires 1e-10 N absolute floor and 250-iteration cap')
    else:
        try:
            from campaign_dynamic_floor_protocol import validate_supplemental_floor
        except ImportError:
            from experiments.campaign_dynamic_floor_protocol import validate_supplemental_floor
        supplemental_metadata = validate_supplemental_floor(supplemental_floor_protocol,
            episode=episode,hz=hz,response=response,relative_tolerance=relative_tolerance,
            absolute_tolerance_N=absolute_tolerance_N,maximum_iterations=maximum_iterations,
            translation=translation,inputs_json=inputs_json,
            core_path=Path(__file__).with_name('campaign_nonlinear_dynamics.py'),gate_path=gate_path)
        supplemental_dependencies = [Path(supplemental_floor_protocol),
            Path(__file__).with_name('campaign_dynamic_floor_protocol.py')]
    case,spec,arrays,inputs,input_lineage=load_case_inputs(episode,dataset,inputs_json)
    if case['split'] != 'train' or case['episode_id'] != episode or case['boundary_type'] != CASE_BOUNDARIES[episode]:
        raise ValueError('TRAIN role, identity or known boundary family mismatch')
    if spec['boundary_type'] != case['boundary_type'] or case['seconds'] != gate['seconds'] or spec['seconds'] != gate['seconds']:
        raise ValueError('Source boundary/duration mismatch')
    points = arrays['rest_points_m'].astype(np.float64)+translation
    tets = arrays['tetrahedra'].copy()
    hardware = hardware_descriptor(points,spec)
    folder = Path(output)
    folder.mkdir(parents=True,exist_ok=False)
    dependencies = [Path(__file__),Path(__file__).with_name('campaign_nonlinear_dynamics.py'),
        Path(__file__).with_name('campaign_nonlinear_fem.py'),Path(__file__).with_name('campaign_linear_fem.py'),
        Path(__file__).with_name('campaign_rigid_inertia.py'),*supplemental_dependencies,*temporal_dependencies]
    hashes = {str(p):sha(p) for p in [*dependencies,*inputs,gate_path]}
    for p in [*dependencies,*inputs,gate_path]:
        target = folder/'provenance'/p.name;target.parent.mkdir(exist_ok=True)
        if target.exists():
            raise FileExistsError('Conflicting provenance basenames')
        target.write_bytes(p.read_bytes())
    np.savez_compressed(folder/'provenance/static-reference.npz',**arrays)
    protocol = {'role':'One TRAIN-only independent dynamic smoke/refinement cell; no reference qualification from this cell.',
        'episode_id':episode,'boundary_kind':case['boundary_type'],'physics_hz':hz,'output_hz':gate['output_hz'],
        'seconds':gate['seconds'],'response':response,'translation_m':translation.tolist(),
        'source_input_dependency_and_gate_sha256':hashes,'declared_material':case['material'],
        'static_input_lineage':input_lineage,
        'saved_regenerated_static_reference_sha256':sha(folder/'provenance/static-reference.npz'),
        'body_only_viscous_damping_s_inverse':case['material']['linear_damping_s_inverse'],
        'native_elasticity_damping_reproduced':False,'hardware':hardware,
        'relative_nonlinear_residual_tolerance':relative_tolerance,'absolute_residual_floor_N':absolute_tolerance_N,
        'supplemental_accuracy_protocol':supplemental_metadata,
        'conditional_temporal_protocol':temporal_metadata,
        'nonlinear_iteration_cap':maximum_iterations,'velocity_semantics':'Unprojected finite pose-map point chords, not instantaneous rigid tangent velocity.',
        'boundary_law':'amplitude*sin(pi*clip(t/seconds,0,1))^2*known_world_direction; exact initial velocity zero.',
        'force_semantics':'Every actual float64 world force is midpoint-evaluated and held through its logged step. Zero applied handle torque and zero external nodal force.',
        'native_equivalence':False,'continuum_accuracy_established':False,'dynamic_reference_gate_passed':False,
        'predictive_reserved_responses_opened':False,'cpu_threads':2,'gpu_used':False}
    write_json(folder/'protocol.json',protocol)
    write_json(folder/'hardware.json',hardware)
    started = time.monotonic(); state = None; completed_steps = 0
    frames,velocities,centers,rotations,times = [],[],[],[],[]
    actual_intervals,midpoints,commands,iterations,step_wall = [],[],[],[],[]
    status = {'pid':os.getpid(),'terminal':False,'completed':False,'completed_steps':0,'planned_steps':round(hz*gate['seconds'])}
    write_json(folder/'execution.json',status)
    try:
        fixed = np.asarray(spec['fixture_nodes'],dtype=int)
        targets = (lambda t: points[fixed]+boundary_motion(spec,t)[0]) if case['boundary_type']=='moving_patch' else None
        target_v = (lambda t: np.tile(boundary_motion(spec,t)[1],(len(fixed),1))) if targets is not None else None
        material = case['material']
        model = NonlinearDynamics(points,tets,material['youngs_modulus_Pa'],material['poissons_ratio'],material['density_kg_m3'],
            fixture_nodes=fixed,load_patches=spec['load_patches'],load_nodes=spec['load_nodes'],
            handle_masses=[h['mass_kg'] for h in hardware],handle_inertias=[h['full_COM_inertia_kg_m2'] for h in hardware],
            drag=material['linear_damping_s_inverse'],boundary_kind=case['boundary_type'],
            fixture_target=targets,fixture_velocity=target_v)
        state = model.initial_state()
        def capture():
            frames.append(state.positions.numpy().copy());velocities.append(state.velocities.numpy().copy())
            centers.append(state.handle_centers.numpy().copy());rotations.append(state.handle_rotations.numpy().copy());times.append(state.time_s)
        capture()
        with (folder/'steps.jsonl').open('x') as stream:
            for index in range(status['planned_steps']):
                h = 1./hz;midpoint = state.time_s+h/2
                force = force_command(gate,response,midpoint)
                old_centers = state.handle_centers.numpy().copy()
                old_coordinates = model.coordinates(state)
                wall = time.monotonic(); state,row = model.step(state,h,handle_forces=force,
                    relative_tolerance=relative_tolerance,absolute_tolerance_N=absolute_tolerance_N,maximum_iterations=maximum_iterations)
                seconds = time.monotonic()-wall; step_wall.append(seconds)
                count = len(row['nonlinear_history'])-1;iterations.append(count)
                actual_intervals.append([row['start_time_s'],row['end_time_s']]);midpoints.append(midpoint);commands.append(force)
                row.pop('held_nodal_forces_N');row.update(step_index=index,nonlinear_iterations=count,step_wall_seconds=seconds,
                    nodal_force_command='all_zero',geometry=geometry_metrics(model,state),
                    handle_centers_start_m=old_centers.tolist(),handle_centers_end_m=state.handle_centers.tolist(),
                    handle_center_step_increment_m=[(state.reduced_position_coordinates[dofs[:3]]-
                        old_coordinates[dofs[:3]]).tolist() for dofs in model.handle_dofs])
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                completed_steps = index+1
                if completed_steps % (hz//gate['output_hz']) == 0:
                    capture()
                if completed_steps == 1 or completed_steps % 30 == 0 or completed_steps == status['planned_steps']:
                    if any(sha(p)!=expected for p,expected in hashes.items()):
                        raise ValueError('Source/input/gate changed during run; preserve incomplete cell')
                    status.update(completed_steps=completed_steps,elapsed_s=time.monotonic()-started,
                        last_step_iterations=count,last_step_seconds=seconds,
                        last_physical_residual_N=row['nonlinear_history'][-1]['scaled_generalized_residual_norm_N'])
                    write_json(folder/'execution.json',status)
                    print(json.dumps({key:status[key] for key in ['completed_steps','planned_steps','elapsed_s','last_step_iterations','last_step_seconds','last_physical_residual_N']}),flush=True)
        intervals = np.asarray(actual_intervals,dtype=np.float64);force_array = np.asarray(commands,dtype=np.float64)
        nominal_times = np.arange(round(gate['seconds']*gate['output_hz'])+1,dtype=np.float64)/gate['output_hz']
        if len(times) != 31 or not np.allclose(times,nominal_times,atol=1e-12,rtol=0):
            raise ValueError('Incomplete or misaligned common-time trajectory')
        np.savez_compressed(folder/'trajectory.npz',times_s=nominal_times,actual_state_times_s=times,
            positions_m=frames,velocities_point_chord_m_s=velocities,handle_centers_m=centers,
            handle_rotations=rotations,force_interval_actual_times_s=intervals,
            force_midpoints_actual_s=midpoints,force_step_commands_N=force_array,
            requested_step_s=np.full(len(intervals),1./hz),nonlinear_iterations=iterations)
        impulse=np.sum(force_array*np.diff(intervals,axis=1)[:,:,None],axis=0)
        analytic=np.zeros((2,3)) if response=='zero_external_force' else .45*gate['seconds']/2*np.array([
            gate['force']['peak_world_force_N_handle0'],gate['force']['peak_world_force_N_handle1']])
        summary={'completed':True,'case':episode,'physics_hz':hz,'response':response,'frames':len(times),
            'elapsed_s':time.monotonic()-started,'nonlinear_iterations':{'minimum':min(iterations),'maximum':max(iterations),'mean':float(np.mean(iterations))},
            'mean_step_seconds':float(np.mean(step_wall)),'maximum_step_seconds':max(step_wall),
            'maximum_clock_error_s':float(np.max(np.abs(np.asarray(times)-nominal_times))),
            'command_impulse_Ns':impulse.tolist(),'analytic_pulse_impulse_Ns':analytic.tolist(),
            'maximum_impulse_error_Ns':float(abs(impulse-analytic).max()),
            'trajectory_sha256':sha(folder/'trajectory.npz'),'steps_sha256':sha(folder/'steps.jsonl'),
            'final_geometry':geometry_metrics(model,state),'total_declared_mass_kg':model.total_mass,
            'reset_replay_performed':False,'dynamic_reference_gate_passed':False,
            'qualification':'Single independent TRAIN diagnostic; no timestep/tolerance/four-BC/reset qualification, native equivalence, continuum accuracy, prediction or uncertainty claim.'}
        write_json(folder/'summary.json',summary)
        status.update(terminal=True,completed=True,completed_steps=completed_steps,elapsed_s=summary['elapsed_s'])
        write_json(folder/'execution.json',status)
        return summary
    except BaseException as error:
        if state is not None:
            save_state(folder/'partial-state.npz',state)
        write_json(folder/'failure.json',{'exception':type(error).__name__,'message':str(error),
            'completed_steps':completed_steps,'elapsed_s':time.monotonic()-started,'completed':False,
            'qualification':'Original failed/incomplete cell preserved; no replacement data generated.'})
        status.update(terminal=True,completed=False,completed_steps=completed_steps,elapsed_s=time.monotonic()-started)
        write_json(folder/'execution.json',status)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--episode',choices=list(CASE_BOUNDARIES),default='t0001')
    parser.add_argument('--hz',type=int,choices=[120,240,480,960],default=120)
    parser.add_argument('--response',choices=['zero_external_force','diagonal_pulse_handle0'],default='diagonal_pulse_handle0')
    parser.add_argument('--relative-tolerance',type=float,choices=[1e-5,1e-7,1e-9],default=1e-7)
    parser.add_argument('--translation',nargs=3,type=float,default=[0.,0.,0.])
    parser.add_argument('--dataset',default='runs/gaussian-boundary-neural-v01')
    parser.add_argument('--absolute-tolerance-N',type=float,default=1e-10)
    parser.add_argument('--supplemental-floor-protocol',type=Path,
        help='Exact separately predeclared moving-zero absolute-floor study only.')
    parser.add_argument('--temporal-extension-protocol',type=Path,
        help='Exact conditional t0003 zero/diagonal 960 Hz temporal study only.')
    parser.add_argument('--inputs-json',type=Path,help='Public privileged static fixture; verifies regenerated arrays and needs no bulk dataset.')
    args=parser.parse_args()
    result=run(args.output,episode=args.episode,hz=args.hz,response=args.response,
        relative_tolerance=args.relative_tolerance,translation=args.translation,dataset=args.dataset,inputs_json=args.inputs_json,
        absolute_tolerance_N=args.absolute_tolerance_N,supplemental_floor_protocol=args.supplemental_floor_protocol,
        temporal_extension_protocol=args.temporal_extension_protocol)
    print(json.dumps({key:result[key] for key in ['completed','case','physics_hz','response','frames','elapsed_s','nonlinear_iterations','maximum_impulse_error_Ns','dynamic_reference_gate_passed']}),flush=True)
