"""Separate guarded TRAIN authored-force episode runner (draft, no launch seal).

Engineering diagnostics only: BC-dependent solver command clocks prohibit this
format's admission as production NN inputs. No existing runner/core is edited.
A reviewed, published, hash-pinned protocol/manifest/seal is mandatory for run().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

try:
    from campaign_nonlinear_dynamics import NonlinearDynamics
    from campaign_nonlinear_dynamics_case import (CASE_BOUNDARIES, boundary_motion,
        geometry_metrics, hardware_descriptor, load_case_inputs, save_state)
    import campaign_full_hardware_context as context
    import campaign_authored_commands as compiler
except ImportError:
    from experiments.campaign_nonlinear_dynamics import NonlinearDynamics
    from experiments.campaign_nonlinear_dynamics_case import (CASE_BOUNDARIES, boundary_motion,
        geometry_metrics, hardware_descriptor, load_case_inputs, save_state)
    from experiments import campaign_full_hardware_context as context
    from experiments import campaign_authored_commands as compiler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
PROTOCOL_PATH = ROOT/'configs/gaussian_campaign_actual_force_envelope_v01.json'
AUDITOR_PATH = ROOT/'experiments/campaign_authored_force_audit.py'
PROTOCOL_SHA256 = 'dd078bd8542cd99015d009e9806e2e09590ee855e6bdb8bafd1cc483ce3d8070'
PINNED = {
    'experiments/campaign_full_hardware_context.py': '47cc0047f2cce6e88c4b76650c9358773eaa233fd670ce2223e4b1f29d7bb958',
    'experiments/campaign_nonlinear_dynamics.py': 'da0db7018b4913020bb228c2fc4df3c240e7dd872f17db4868de147eaaf78fa2',
    'experiments/campaign_nonlinear_fem.py': '468daaafa9d02c9db27553e2cfb154190248500677c212d9dde25bffdf14e1ff',
    'experiments/campaign_linear_fem.py': 'c4a94946b8d0edd2246a889dddcf89b58e0a8c660d75250a89f95de85b31ca55',
    'experiments/campaign_rigid_inertia.py': 'e420d9cadaa97d1b1648227b7b28af1803952caa87432f819baf0827deecabf7',
    'experiments/campaign_nonlinear_dynamics_case.py': 'e7b6b785c262f43d4b1acc503b14feaf1cad659c171024346fc9e70a4dffc342',
    'experiments/campaign_authored_commands.py': 'fdbf333b9531ce11700b6be05ab916fbdb7b30c2fb96cec3762369ae05a3c475',
    'experiments/isaac_dataset.py': '197174b0ddc85c67ed5179ab07a6e48d6c24b8bfa587495bb249c8a19e2f2f0c',
    'experiments/boundary_block.py': '54b0261c9f5b1518a0e071df482d3e781db11874cd3282dda727ef5505a0ab97',
    'configs/gaussian_independent_diagnostic_inputs_v01.json': '073380f72559c50ccc674d5f69fdff1be7e4d0b30484ad66f0233695a813847f',
}
ENGINEERING_ONLY = 'Engineering-only BC-dependent solver command clock; NOT production NN input or admitted dataset.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def resolve_repo_path(relative_record, repo_root=None):
    """Resolve canonical repo-relative records, never CWD or original-root fallback."""
    root = Path(ROOT if repo_root is None else repo_root).resolve()
    value = str(relative_record); relative = Path(value)
    if (relative.is_absolute() or '..' in relative.parts or not relative.parts
            or value != relative.as_posix() or value == '.'):
        raise ValueError('Canonical repository-relative path required: '+value)
    resolved = (root/relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Repository-relative path escapes root: '+value)
    return resolved


def repo_record(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def checked_record(record):
    path = resolve_repo_path(record['path'])
    if sha(path) != record['sha256']:
        raise ValueError('Pinned proof/source mismatch: '+str(path))
    return path


def dependencies():
    paths = dict(PINNED)
    paths[repo_record(SOURCE)] = sha(SOURCE)
    for path, expected in paths.items():
        if sha(resolve_repo_path(path)) != expected:
            raise ValueError('Frozen dependency changed: '+path)
    return paths


def validate_candidate(protocol, proof, scenario_id, hz):
    """Verify a complete reviewed auditor declaration; never infer from success alone."""
    if scenario_id.endswith('__zero_external_force'):
        zero = proof['accepted_zero_candidates'].get(scenario_id, {}).get(str(hz))
        if zero is None or not zero.get('passing_family_ids'):
            raise ValueError('Moving-zero control requires accepted families at this exact rate')
        candidates = []
        for family_id in zero['passing_family_ids']:
            candidate = validate_candidate(protocol, proof, family_id, hz)
            if not family_id.startswith('t0003__'):
                raise ValueError('Moving-zero proof cannot be shared with another layout')
            candidates.append(candidate)
        entry = zero
    else:
        entry = proof['accepted_candidates'].get(scenario_id)
        if entry is None:
            raise ValueError('Missing scenario candidate proof')
        scenario = next((s for s in protocol['scenarios'] if s['id'] == scenario_id), None)
        if scenario is None:
            raise ValueError('Unknown driven scenario')
        rates = entry['all_actual_rates_hz']
        initial = scenario['initial_rates_hz']
        allowed = initial+protocol['conditional_refinement']['allowed_next_rates_hz'][scenario['boundary_type']]
        if (not isinstance(rates,list) or len(rates)<3 or rates != allowed[:len(rates)]
                or any(isinstance(r,bool) or not isinstance(r,int) for r in rates) or hz != rates[-1]):
            raise ValueError('Complete allowed temporal rate history and finest accepted rate required')
        signal = entry['force_induced_signal_m']; tolerance = entry['tolerance_m']
        if not np.isfinite(signal) or signal < 0 or tolerance != max(2e-6,.05*signal):
            raise ValueError('Exact family-specific temporal tolerance required')
        for role in ['driven','zero']:
            metrics = entry['full_'+role+'_temporal_metrics']
            errors = np.asarray(metrics['adjacent_coordinate_rmse_m'],dtype=float)
            if errors.shape != (len(rates)-1,) or not np.isfinite(errors).all() or np.any(errors<0):
                raise ValueError('Complete finite adjacent '+role+' errors required')
            if not np.array_equal(errors,np.asarray(entry['all_adjacent_errors_m'][role])):
                raise ValueError('Temporal error records disagree')
            contract = errors[-1] < tolerance and (errors[-1] < errors[-2] or np.all(errors[-2:] < 2e-6))
            if metrics.get('passed') is not True or not contract:
                raise ValueError('Declared '+role+' temporal contraction gate failed')
    if (entry.get('passed') is not True or entry.get('physics_hz') != hz
            or entry.get('output_hz') != 20 or entry.get('frames') != 61
            or entry.get('geometry_residual_checks_passed') is not True):
        raise ValueError('Complete 61-frame geometry/residual candidate proof required')
    evidence = entry.get('source_evidence_records')
    if not isinstance(evidence,list) or not evidence:
        raise ValueError('Candidate source and response evidence records required')
    for record in evidence:
        if not isinstance(record.get('role'),str) or not record['role']:
            raise ValueError('Evidence information role required')
        checked_record(record)
    return entry


def validate_cell(protocol, manifest, cell_id):
    """Exact resolved rows, never CLI numerical overrides or default matched Hz."""
    if manifest.get('schema') != 'actual-authored-force-resolved-stage-v0.1' or manifest.get('status') != 'FROZEN_PUBLISHED_RESOLVED_EXECUTION_MANIFEST':
        raise ValueError('Exact reviewed resolved-stage manifest schema/status required')
    if protocol['protocol'] != 'actual-authored-force-envelope-TRAIN-v0.1':
        raise ValueError('Declared authored-force protocol required')
    if (protocol['seconds'], protocol['output_hz'], protocol['frames']) != (3., 20, 61):
        raise ValueError('New exact three-second / 61-frame design required')
    templates = {r['id']: r for r in protocol['initial_cell_plan']}
    if len(templates) != len(protocol['initial_cell_plan']):
        raise ValueError('Duplicate protocol cells')
    rows = manifest['cells']
    if len({r['id'] for r in rows}) != len(rows) or len({str(resolve_repo_path(r['output'])) for r in rows}) != len(rows):
        raise ValueError('Resolved IDs/output paths must be unique')
    selected = [r for r in rows if r['id'] == cell_id]
    if len(selected) != 1:
        raise ValueError('Exactly one declared resolved cell required')
    stage = manifest['stage']
    if stage not in ['initial_temporal', 'matched_after_temporal']:
        raise ValueError('Conditional refinement needs a separate reviewed runner declaration')
    if stage == 'initial_temporal':
        expected = {r['id'] for r in templates.values() if r['stage'] == stage}
        if {r['template_id'] for r in rows} != expected or len(rows) != len(expected):
            raise ValueError('Initial stage requires all exact 51 temporal rows')
    for row in rows:
        template = templates.get(row['template_id'])
        if template is None or template['stage'] != stage:
            raise ValueError('Unknown or wrong-stage template')
        for name in ['case', 'scenario_id', 'response', 'output_hz', 'relative_tolerance',
                     'absolute_tolerance_N', 'translation_m', 'maximum_iterations', 'kind']:
            if row[name] != template[name]:
                raise ValueError('Resolved cell changed '+name)
        for name in ['output_hz', 'maximum_iterations']:
            if isinstance(row[name],bool) or not isinstance(row[name],int):
                raise ValueError('Exact integer '+name+' required')
        hz = row['hz']
        if isinstance(hz, bool) or not isinstance(hz, int) or hz <= 0 or hz % 20:
            raise ValueError('Positive integer common 20 Hz rate required')
        if stage == 'initial_temporal':
            if hz != template['hz'] or row['id'] != template['id']:
                raise ValueError('Initial temporal settings/ID changed')
        elif row['id'] != template['id']+'__hz'+str(hz):
            raise ValueError('Matched output ID must include resolved Hz')
        if stage == 'matched_after_temporal':
            proof = read(checked_record(row['temporal_candidate_proof']))
            if proof['protocol_sha256'] != manifest['protocol_sha256']:
                raise ValueError('Matched proof/protocol mismatch')
            validate_candidate(protocol, proof, row['scenario_id'], hz)
    return dict(selected[0])


def validate_seal(seal_path, cell_id):
    """Internal provenance contract, not an additional user approval workflow."""
    seal_path = resolve_repo_path(seal_path); seal = read(seal_path)
    if (seal.get('schema') != 'actual-authored-force-prelaunch-seal-v0.1'
            or seal.get('status') != 'REVIEWED_PUBLISHED_EXECUTION_SEAL'
            or seal.get('physical_launch_authorized') is not True or seal.get('sources_and_stage_manifest_published') is not True):
        raise ValueError('Draft alone cannot launch physical episodes')
    protocol_path = checked_record(seal['protocol']); protocol = read(protocol_path)
    if protocol_path != PROTOCOL_PATH or sha(protocol_path) != PROTOCOL_SHA256:
        raise ValueError('Exact frozen authored-force protocol required')
    manifest_path = checked_record(seal['resolved_manifest']); manifest = read(manifest_path)
    if manifest['protocol_sha256'] != sha(protocol_path):
        raise ValueError('Resolved stage/protocol mismatch')
    expected = dependencies()
    if seal['source_sha256'] != expected or manifest['source_sha256'] != expected:
        raise ValueError('Exact reviewed runner/compiler/context/dependency pins required')
    auditor_path = checked_record(seal['authored_auditor_source'])
    if auditor_path != AUDITOR_PATH or manifest.get('authored_auditor_source_sha256') != sha(auditor_path):
        raise ValueError('Exact reviewed authored auditor path/source pin required')
    prerequisite_path = checked_record(seal['small_force_complete60_audit'])
    prior = read(prerequisite_path)
    if sha(prerequisite_path) != protocol['proof_pin_pending_before_launch']['small_force_complete60_audit_sha256']:
        raise ValueError('Exact completed prerequisite audit pin required')
    if (prior.get('required_unique_cells') != 60 or prior.get('indexed_unique_cells') != 60
            or prior.get('verified_unique_cells') != 60 or len(prior.get('cells', [])) != 60
            or not all(row.get('passed') is True for row in prior['cells'])
            or prior.get('independent_fixed_discretization_small_force_reference_qualified') is not True):
        raise ValueError('Complete passing 60-cell small-force prerequisite required')
    resources_path = checked_record(seal['resource_declaration']); resources = read(resources_path)
    workers = resources.get('max_parallel_workers'); threads = resources.get('threads_per_worker')
    if (isinstance(threads,bool) or not isinstance(threads,int) or threads != 2
            or isinstance(workers,bool) or not isinstance(workers,int) or not 1 <= workers <= 8
            or resources.get('gpu_used') is not False):
        raise ValueError('Separate bounded CPU resource declaration required')
    cell = validate_cell(protocol, manifest, cell_id)
    paths = {**expected, **{repo_record(p): sha(p) for p in [seal_path, protocol_path, manifest_path,
        auditor_path, prerequisite_path, resources_path]}}
    for row in manifest['cells']:
        if 'temporal_candidate_proof' in row:
            p = checked_record(row['temporal_candidate_proof']); paths[repo_record(p)] = sha(p)
            if read(p).get('auditor_source_sha256') != sha(auditor_path):
                raise ValueError('Candidate proof must come from the reviewed authored auditor')
    return protocol, manifest, cell, paths


def validate_prelaunch(seal_path, cell_id, output):
    protocol, manifest, cell, paths = validate_seal(seal_path, cell_id)
    if any(os.environ.get(name) != '2' for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']):
        raise ValueError('Worker BLAS/OpenMP thread environment must be explicitly capped at two')
    if resolve_repo_path(cell['output']) != resolve_repo_path(output) or resolve_repo_path(output).exists():
        raise ValueError('Exact fresh resolved output required')
    return protocol, cell, paths


def compile_cell(spec, cell):
    family = cell['response']
    zero = family == 'zero_external_force'
    compiled = compiler.compile_authored_commands(spec, force_family=spec['force_type'] if zero else family,
        physics_hz=cell['hz'], output_hz=20, information_role='known_actuator_command')
    if zero:
        compiled['held_handle_forces_world_N'][:] = 0.
        compiled['metadata'] = {**compiled['metadata'], 'requested_force_family': family,
            'explicit_zero_force_control': True,
            'zero_override': 'All external commands exactly zero; authored support/hardware remain.'}
        for key in list(compiled['metadata']):
            if 'impulse' in key or 'first_time_moment' in key:
                value = compiled['metadata'][key]
                compiled['metadata'][key] = np.zeros_like(value).tolist() if isinstance(value, list) else 0.
    return compiled


def make_contexts(case, spec, arrays, hardware, compiled, translation):
    """No truth hashes/BC labels in public features; diagnostics are separate."""
    points = arrays['rest_points_m'].astype(np.float64)+translation
    handles = [context.KnownHandle(origin_world_m=h['reference_origin_m'], ordered_patch_nodes=np.asarray(h['ordered_patch_nodes']),
        mass_kg=h['mass_kg'], com_inertia_body_kg_m2=h['full_COM_inertia_kg_m2'],
        com_offset_body_m=h['COM_offset_m'], initial_rotation_body_to_world=h['initial_rotation_matrix'],
        origin_node_id=h['reference_origin_node']) for h in hardware]
    known = context.KnownHardware(rest_points_world_m=points, tetrahedra=arrays['tetrahedra'],
        surface_faces=arrays['surface_faces'], handles=handles, coordinate_origin_world_m=points.mean(0), source_sha256={})
    T = len(compiled['force_midpoints_s']); N = len(points)
    program = context.ForceProgram(held_handle_forces_world_N=compiled['held_handle_forces_world_N'],
        held_handle_torques_world_Nm=np.zeros((T, 2, 3)), held_nodal_forces_world_N=np.zeros((T, N, 3)),
        held_body_acceleration_world_m_s2=np.zeros((T, 3)), information_role='known_actuator_command', source_sha256={})
    support = np.zeros((T+1, 2, 3))
    if case['boundary_type'] == 'moving_patch':
        support[:, 0] = np.stack([boundary_motion(spec, t)[0] for t in compiled['command_boundary_times_s']])
    query = context.HeldCommandQuery(boundary_times_s=compiled['command_boundary_times_s'], force_programs=[program],
        support_translations_world_m=support, support_rotations_world=np.broadcast_to(np.eye(3), (T+1, 2, 3, 3)),
        support_law_role='candidate_boundary_hypothesis', support_law_source_sha256={},
        requested_output_times_s=compiled['requested_output_times_s'], source_sha256={})
    mask = np.zeros((2, N, 3), dtype=bool); mask[0, spec['fixture_nodes']] = True
    boundary = context.BoundaryHypothesis(constraint_axis_masks=mask,
        support_origins_relative_m=np.array([points[spec['fixture_nodes']].mean(0)-known.origin, np.zeros(3)]))
    material = case['material']
    hypothesis = context.MaterialHypothesis(youngs_modulus_Pa=material['youngs_modulus_Pa'],
        poissons_ratio=material['poissons_ratio'], density_kg_m3=material['density_kg_m3'],
        body_linear_drag_s_inverse=material['linear_damping_s_inverse'], constitutive_model='log_J_neo_hookean')
    public = context.inference_public_context(known_hardware=known, command_query=query)
    conditioned = context.conditioned_forward_context(known_hardware=known, command_query=query,
        material_hypothesis=hypothesis, boundary_hypothesis=boundary,
        hypothesis_role='known_parameter_training_or_diagnostic', hypothesis_source_sha256={})
    return known, query, boundary, public, conditioned


def validate_supported_context(known, query, boundary, spec):
    """Reject broader schema inputs the frozen case adapter cannot execute."""
    for h in known.handles:
        if np.any(h.com_offset_body_m != 0) or not np.array_equal(h.initial_rotation_body_to_world, np.eye(3)):
            raise ValueError('This frozen fixture adapter requires zero COM and identity initial rotations')
    masks = boundary.masks
    if masks.shape != (2, len(known.points), 3) or not np.array_equal(masks[:, :, 0], masks[:, :, 1]) or not np.array_equal(masks[:, :, 1], masks[:, :, 2]):
        raise ValueError('Only full three-axis fixture masks are supported')
    if not np.array_equal(np.flatnonzero(masks[:, :, 0].any(0)), np.sort(spec['fixture_nodes'])):
        raise ValueError('Fixture membership mismatch')
    if np.any(query.support_rotation != np.eye(3)) or len(known.handles) != 2:
        raise ValueError('Two handles and translation-only support required')
    forcing = context.combined_forward_forcing(query)['features']
    for name in ['held_handle_torques_world_Nm', 'held_nodal_forces_world_N', 'held_body_acceleration_world_m_s2']:
        if np.any(forcing[name] != 0):
            raise ValueError('Unsupported nonzero forcing: '+name)


def save_context(folder, public, conditioned):
    for role, data in [('engineering-public', public), ('known-parameter-diagnostic', conditioned)]:
        group = folder/role; group.mkdir()
        arrays = {name+'/'+key: value for name, fields in data['feature_groups'].items() for key, value in fields.items()}
        np.savez_compressed(group/'features.npz', **arrays)
        write(group/'manifest.json', {'role': data['role'], 'hypothesis_role': data.get('hypothesis_role'),
            'admission': ENGINEERING_ONLY, 'feature_groups': {name: {key: context.descriptor(v) for key, v in fields.items()}
                for name, fields in data['feature_groups'].items()}, 'feature_archive_sha256': sha(group/'features.npz')})


def endpoint_metrics(model, state, previous=None):
    metrics = geometry_metrics(model, state)
    ids = torch.as_tensor(model.tets, dtype=torch.long)
    F = torch.einsum('eic,eid->ecd', (state.positions-model.points)[ids], model.gradients)+torch.eye(3, dtype=torch.float64)
    eigenvalues = torch.linalg.eigvalsh(F.transpose(-1,-2)@F)
    metrics.update(minimum_principal_stretch=float(eigenvalues.min().clamp_min(0).sqrt()),
        maximum_principal_stretch=float(eigenvalues.max().sqrt()),
        minimum_Green_principal_strain=float((eigenvalues.min()-1)/2),
        maximum_Green_principal_strain=float((eigenvalues.max()-1)/2))
    patch_error = 0.
    for j, patch in enumerate(model.patches):
        expected = state.handle_centers[j]+(model.points[patch]-model.points[model.origins[j]])@state.handle_rotations[j].T
        patch_error = max(patch_error, float((state.positions[patch]-expected).abs().max()))
    R = state.handle_rotations
    proper = max(float((R.transpose(-1,-2)@R-torch.eye(3, dtype=torch.float64)).abs().max()), float((torch.linalg.det(R)-1).abs().max()))
    angle = lambda r: float(torch.acos(((r.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)).max())
    metrics.update(maximum_rigid_pose_map_error_m=patch_error, maximum_rotation_SO3_error=proper,
        maximum_handle_rotation_rad=angle(R), maximum_handle_step_rotation_rad=0. if previous is None else angle(R@previous.handle_rotations.transpose(-1,-2)))
    return metrics


def endpoint_guard(model, state, metrics, row=None):
    model.validate_state(state)
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError('Nonfinite accepted-endpoint geometry')
    if metrics['minimum_volume_ratio'] < .1:
        raise ValueError('Accepted endpoint J below declared 0.1')
    if max(metrics['fixture_max_error_m'], metrics['maximum_rigid_pose_map_error_m'], metrics['maximum_rotation_SO3_error']) > 1e-12:
        raise ValueError('Accepted endpoint exact fixture/rigid transform guard failed')
    if row is not None:
        if not row['converged'] or row['nonlinear_history'][-1]['scaled_generalized_residual_norm_N'] > row['residual_tolerance_N']:
            raise ValueError('Original discrete residual gate failed')
        for entry in row['nonlinear_history']:
            if not all(not isinstance(v, (int, float)) or math.isfinite(v) for v in entry.values()) or entry['minimum_J'] <= 0:
                raise ValueError('Nonfinite/nonpositive-J nonlinear history')


def failure_safe(value):
    """Preserve nonfinite failure evidence explicitly without invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k:failure_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [failure_safe(v) for v in value]
    return value


def capture_traceback_failure(error, folder):
    """Copy bounded named locals only; never retain frames/models/autograd graphs."""
    proof = {}; arrays = {}; trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        if frame.f_code.co_name == 'step' and Path(frame.f_code.co_filename).resolve() == ROOT/'experiments/campaign_nonlinear_dynamics.py':
            local = frame.f_locals
            history = local.get('history')
            if isinstance(history, list):
                proof['nonlinear_history'] = history[:251]
            for name in ['q', 'trial', 'target', 'F', 'f']:
                value = local.get(name)
                if isinstance(value, torch.Tensor):
                    arrays[name] = value.detach().cpu().numpy().copy()
            configuration = local.get('configuration')
            if isinstance(configuration, tuple):
                for name, value in zip(['positions', 'auxiliary_positions', 'handle_centers', 'handle_rotations'], configuration):
                    if isinstance(value, torch.Tensor):
                        arrays['unaccepted_'+name] = value.detach().cpu().numpy().copy()
            for name in ['candidate_metrics', 'metrics', 'minJ', 'alpha']:
                value = local.get(name)
                if isinstance(value,(dict,float,int)):
                    proof[name] = value
            final_candidate = local.get('_')
            if isinstance(final_candidate,tuple) and len(final_candidate)==4:
                for name,value in zip(['positions','auxiliary_positions','handle_centers','handle_rotations'],final_candidate):
                    if isinstance(value,torch.Tensor):
                        arrays['last_evaluated_trial_'+name] = value.detach().cpu().numpy().copy()
                proof['trial_configuration_caveat'] = 'Last successfully evaluated trial, possibly preceding later invalid-J trials; trial vector is the final proposed reduced candidate, alpha is the post-loop scalar.'
            proof['semantics'] = 'UNACCEPTED internal iteration/candidate locals; not an accepted physical endpoint.'
        trace = trace.tb_next
    if arrays:
        np.savez_compressed(folder/'unaccepted-core-locals.npz', **arrays)
    return proof


def run(output, *, prelaunch_seal, cell_id):
    # Every proof/source/request is validated BEFORE mkdir or a dynamics call.
    protocol, cell, pins = validate_prelaunch(prelaunch_seal, cell_id, output)
    case, spec, arrays, _, lineage = load_case_inputs(cell['case'], None, resolve_repo_path(protocol['source_fixture']))
    if protocol['source_fixture_sha256'] != PINNED[protocol['source_fixture']]:
        raise ValueError('Protocol fixture pin mismatch')
    for path, digest in protocol['frozen_source_sha256'].items():
        if PINNED.get(path) != digest:
            raise ValueError('Protocol physical/compiler dependency mismatch')
    if spec['seconds'] != 3. or case['seconds'] != 3.:
        raise ValueError('Exact source duration required')
    if cell['response'] != 'zero_external_force':
        scenario = next(s for s in protocol['scenarios'] if s['id'] == cell['scenario_id'])
        if (scenario['source_recipe_sha256'] != case['recipe_sha256']
                or scenario['source_spec_canonical_sha256'] != context.digest(spec)
                or scenario['exact_authored_amplitude_coefficient_N'] != spec['peak_force_N']
                or scenario['requested_force_family'] != cell['response']):
            raise ValueError('Exact source force scenario mismatch')
    if case['split'] != 'train' or case['boundary_type'] != CASE_BOUNDARIES[cell['case']]:
        raise ValueError('Four declared TRAIN static fixtures only')
    translation = np.asarray(cell['translation_m'], dtype=float)
    points = arrays['rest_points_m'].astype(np.float64)+translation
    hardware = hardware_descriptor(points, spec); compiled = compile_cell(spec, cell)
    known, query, boundary, public, conditioned = make_contexts(case, spec, arrays, hardware, compiled, translation)
    validate_supported_context(known, query, boundary, spec)
    for path, expected in pins.items():
        if sha(resolve_repo_path(path)) != expected:
            raise ValueError('Source changed during preflight')
    folder = resolve_repo_path(output); folder.mkdir(parents=True, exist_ok=False)
    status = {'pid': os.getpid(), 'terminal': False, 'completed': False, 'completed_steps': 0, 'planned_steps': len(compiled['force_midpoints_s']), 'cpu_threads': 2, 'gpu_used': False}
    state = previous = model = last_accepted = None; row = None; request = None; started = time.monotonic()
    frames = []; intervals = []; work = []; moments = []; step_seconds = []; extrema = []
    write(folder/'execution.json', status)
    try:
        provenance = folder/'private-provenance'; provenance.mkdir()
        for index, (path, digest) in enumerate(pins.items()):
            target = provenance/(str(index).zfill(2)+'-'+Path(path).name); target.write_bytes(resolve_repo_path(path).read_bytes())
            if sha(target) != digest:
                raise ValueError('Source snapshot verification failed')
        write(provenance/'pins.json', pins)
        write(provenance/'snapshots.json', [{'original_path':path, 'saved_path':str(index).zfill(2)+'-'+Path(path).name, 'sha256':digest}
            for index,(path,digest) in enumerate(pins.items())])
        np.savez_compressed(provenance/'static-reference.npz', **arrays)
        write(provenance/'physical-inputs.json', {'case': case, 'spec': spec, 'hardware': hardware, 'static_lineage': lineage})
        write(folder/'protocol.json', {'cell': cell, 'seconds': 3., 'output_hz': 20, 'frames': 61,
            'admission': ENGINEERING_ONLY, 'physics': 'independent float64 log-J Neo-Hookean, body-only drag, zero gravity/no contact',
            'native_elasticity_damping_reproduced': False, 'surface_crossing_diagnostic': 'not_evaluated',
            'source_input_sha256': pins, 'compiled_force_metadata': compiled['metadata'],
            'native_equivalence': False, 'continuum_accuracy_established': False, 'actual_force_envelope_qualified': False})
        save_context(folder, public, conditioned)
        np.savez_compressed(folder/'compiled-commands.npz', **{k:v for k,v in compiled.items() if isinstance(v,np.ndarray)})
        torch.set_num_threads(2)
        fixed = np.asarray(spec['fixture_nodes'], dtype=int)
        moving = case['boundary_type'] == 'moving_patch'
        material = case['material']
        model = NonlinearDynamics(points, arrays['tetrahedra'], material['youngs_modulus_Pa'], material['poissons_ratio'], material['density_kg_m3'],
            fixture_nodes=fixed, load_patches=spec['load_patches'], load_nodes=spec['load_nodes'],
            handle_masses=[h.mass_kg for h in known.handles], handle_inertias=[h.com_inertia_body_kg_m2 for h in known.handles],
            handle_com_offsets=[h.com_offset_body_m for h in known.handles], drag=material['linear_damping_s_inverse'], boundary_kind=case['boundary_type'],
            fixture_target=(lambda t: points[fixed]+boundary_motion(spec,t)[0]) if moving else None,
            fixture_velocity=(lambda t: np.tile(boundary_motion(spec,t)[1],(len(fixed),1))) if moving else None)
        state = model.initial_state(); metrics = endpoint_metrics(model, state); endpoint_guard(model, state, metrics); extrema.append(metrics); last_accepted = state
        def capture():
            frames.append({'time':state.time_s, **{key:getattr(state,key).numpy().copy() for key in [
                'positions','velocities','auxiliary_positions','auxiliary_velocities','handle_centers','handle_rotations']}})
        capture()
        with (folder/'steps.jsonl').open('x') as stream:
            for index, force in enumerate(compiled['held_handle_forces_world_N']):
                previous = state; row = None
                request = {'step_index': index, 'compiled_interval_s': compiled['force_interval_times_s'][index].tolist(),
                    'compiled_midpoint_s': float(compiled['force_midpoints_s'][index]), 'actual_state_start_s': previous.time_s,
                    'requested_dt_s': 1./cell['hz'], 'held_handle_forces_world_N': force.tolist()}
                old_coordinates = model.coordinates(previous)
                wall = time.monotonic()
                state, row = model.step(previous, 1./cell['hz'], handle_forces=force,
                    relative_tolerance=cell['relative_tolerance'], absolute_tolerance_N=cell['absolute_tolerance_N'], maximum_iterations=cell['maximum_iterations'])
                step_seconds.append(time.monotonic()-wall)
                metrics = endpoint_metrics(model, state, previous); extrema.append(metrics)
                increments = np.array([(state.reduced_position_coordinates[dofs[:3]]-old_coordinates[dofs[:3]]).numpy() for dofs in model.handle_dofs])
                per_work = np.einsum('hc,hc->h', force, increments)
                start_centers = previous.handle_centers.numpy(); end_centers = state.handle_centers.numpy()
                moment = np.stack([np.cross(start_centers, force),np.cross(end_centers,force)])
                row.pop('held_nodal_forces_N'); row.update(**request, actual_interval_s=[previous.time_s,state.time_s],
                    actual_state_midpoint_s=(previous.time_s+state.time_s)/2, geometry=metrics,
                    handle_centers_start_m=start_centers.tolist(), handle_centers_end_m=end_centers.tolist(),
                    handle_center_step_increment_m=increments.tolist(), held_handle_work_each_J=per_work.tolist(),
                    moment_about_world_origin_start_end_Nm=moment.tolist(), step_wall_seconds=step_seconds[-1])
                endpoint_guard(model, state, metrics, row)
                if row['start_time_s'] != previous.time_s or row['end_time_s'] != state.time_s or np.max(np.abs(np.array([previous.time_s,state.time_s])-compiled['force_interval_times_s'][index])) > 1e-12:
                    raise ValueError('Measured state clock disagrees with compiled command interval')
                if cell['response'] == 'twist_couple' and np.any(force.sum(0) != 0):
                    raise ValueError('Twist couple lost exact zero resultant')
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                intervals.append([previous.time_s,state.time_s]);work.append(per_work);moments.append(moment)
                status['completed_steps'] = index+1; last_accepted = state
                if (index+1) % (cell['hz']//20) == 0:
                    capture()
                if index == 0 or (index+1) % 30 == 0 or index+1 == status['planned_steps']:
                    if any(sha(resolve_repo_path(path))!=digest for path,digest in pins.items()):
                        raise ValueError('Frozen dependency changed during cell')
                    status.update(elapsed_s=time.monotonic()-started, last_state_time_s=state.time_s,
                        last_residual_N=row['nonlinear_history'][-1]['scaled_generalized_residual_norm_N'])
                    write(folder/'execution.json',status)
                    print(json.dumps(status),flush=True)
        times=np.array([f['time'] for f in frames]);nominal=compiled['requested_output_times_s']
        if len(frames)!=61 or np.max(abs(times-nominal))>1e-12:
            raise ValueError('Incomplete actual 61-frame trajectory')
        dt=np.diff(np.array(intervals),axis=1)[:,0]; commands=compiled['held_handle_forces_world_N']
        np.savez_compressed(folder/'trajectory.npz',times_s=nominal,actual_state_times_s=times,
            positions_m=[f['positions'] for f in frames],velocities_point_chord_m_s=[f['velocities'] for f in frames],
            auxiliary_positions_m=[f['auxiliary_positions'] for f in frames],auxiliary_velocities_point_chord_m_s=[f['auxiliary_velocities'] for f in frames],
            handle_centers_m=[f['handle_centers'] for f in frames],handle_rotations=[f['handle_rotations'] for f in frames],
            force_interval_actual_times_s=intervals,force_midpoints_compiled_s=compiled['force_midpoints_s'],force_step_commands_N=commands,
            handle_work_each_J=work,moment_about_world_origin_start_end_Nm=moments)
        summary={'completed':True,'frames':61,'cell_id':cell['id'],'elapsed_s':time.monotonic()-started,
            'maximum_frame_clock_error_s':float(abs(times-nominal).max()),'actual_clock_impulse_Ns':np.einsum('t,thc->hc',dt,commands).tolist(),
            'actual_clock_absolute_impulse_Ns':np.einsum('t,thc->hc',dt,np.abs(commands)).tolist(),
            'actual_clock_first_time_moment_Ns2':np.einsum('t,thc->hc',dt*np.array(intervals).mean(1),commands).tolist(),
            'handle_work_each_total_J':np.sum(work,axis=0).tolist(), 'endpoint_metric_ranges':{k:[min(e[k] for e in extrema),max(e[k] for e in extrema)] for k in extrema[0]},
            'trajectory_sha256':sha(folder/'trajectory.npz'),'steps_sha256':sha(folder/'steps.jsonl'),
            'admission':ENGINEERING_ONLY,'actual_force_envelope_qualified':False,'native_reference_qualified':False,
            'continuum_accuracy_established':False,'prediction_qualified':False}
        write(folder/'summary.json',summary);status.update(terminal=True,completed=True,elapsed_s=summary['elapsed_s']);write(folder/'execution.json',status)
        return summary
    except BaseException as error:
        # Last accepted state is previous if this step returned an invalid candidate.
        if last_accepted is not None: save_state(folder/'last-accepted-state.npz',last_accepted)
        if state is not None: save_state(folder/'failure-endpoint-state.npz',state)
        if frames:
            np.savez_compressed(folder/'partial-trajectory.npz',actual_state_times_s=[f['time'] for f in frames],
                **{key:np.array([f[key] for f in frames]) for key in frames[0] if key!='time'})
        locals_proof=capture_traceback_failure(error,folder)
        if row is not None:
            write(folder/'failure-endpoint-ledger.json', failure_safe({'row':row, 'endpoint_accepted':state is last_accepted}))
        # Error messages are bounded; full named iteration history is separately retained.
        write(folder/'failure.json',{'completed':False,'exception':type(error).__name__,'message':str(error)[:16000],
            'completed_steps':status['completed_steps'],'step_request':request,'returned_candidate_is_not_accepted':state is not last_accepted,
            'core_failure':failure_safe(locals_proof),'elapsed_s':time.monotonic()-started})
        status.update(terminal=True,completed=False,elapsed_s=time.monotonic()-started);write(folder/'execution.json',status)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--prelaunch-seal',required=True,type=Path);p.add_argument('--cell-id',required=True)
    a=p.parse_args();result=run(a.output,prelaunch_seal=a.prelaunch_seal,cell_id=a.cell_id)
    print(json.dumps({k:result[k] for k in ['completed','frames','cell_id','elapsed_s']}),flush=True)
