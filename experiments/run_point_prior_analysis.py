"""Role-gated episodes for the frozen point-prior posterior/view experiment.

Development and prepare-execution need no cohort response arrays or labels.
Calibration opens only the first16 complete A/B pairs. Reserved access needs
a separate hash-bound calibration freeze AND explicit evaluation release.
Each episode freezes action choice before A24, and all predictive states before
future motion/material/support truth. NPZ storage may decompress a complete
positions member; only explicitly allowed frames leave the early-read adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

try:
    from campaign_point_prior_posterior import (METHOD_PATH, METHOD_SHA256, SUPPORT_PRIOR,
        load_reference_libraries, condition_reference)
    from campaign_point_active_views import select_point_active_view, update_after_view
    from campaign_simultaneous_bands import object_score
    from censored_range_observation import CensoredRangeModel, CensoredRanges
except ImportError:
    from experiments.campaign_point_prior_posterior import (METHOD_PATH, METHOD_SHA256, SUPPORT_PRIOR,
        load_reference_libraries, condition_reference)
    from experiments.campaign_point_active_views import select_point_active_view, update_after_view
    from experiments.campaign_simultaneous_bands import object_score
    from experiments.censored_range_observation import CensoredRangeModel, CensoredRanges

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'point-prior-analysis-execution-v0.1'
STATE_NAMES = ('no_additional_view', *(f'fixed_view_{i}' for i in range(4)))
ACQUIRED_QUERY_SHA = '9699169b51e02184f6eb41372d4c00dcc8261eda844f8d712b2b2452f2b677e1'
SOURCE_NAMES = ('run_point_prior_analysis.py', 'campaign_point_prior_posterior.py',
    'campaign_point_active_views.py', 'campaign_simultaneous_bands.py',
    'campaign_point_material_reference.py', 'campaign_material_marginal.py',
    'campaign_sparse_depth_queries.py', 'campaign_ray_sensor.py',
    'censored_range_observation.py', 'campaign_support_basis_cache.py',
    'campaign_packed_boundary_forward.py', 'campaign_exciting_commands.py',
    'summarize_point_prior_analysis.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve(path, root=ROOT):
    root = Path(root).resolve()
    p = Path(path)
    if p.is_absolute() or '..' in p.parts or p.as_posix() != str(path):
        raise ValueError('Canonical repository-relative path required')
    result = (root/p).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Repository escape')
    return result


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def record(path, root=ROOT):
    path = Path(path).resolve()
    return dict(path=path.relative_to(Path(root).resolve()).as_posix(), sha256=sha(path))


def checked(rec, root=ROOT):
    path = resolve(rec['path'], root)
    if sha(path) != rec['sha256']:
        raise ValueError('Frozen artifact changed: '+rec['path'])
    return path


def eligible_objects(plan, role, *, execution_sha256, method_sha256,
                     release=None, calibration_freeze=None, calibration_freeze_record=None):
    """Metadata-only role barrier, evaluated BEFORE capture/truth file access."""
    objects = plan['objects']
    if (len(objects) != 48 or sorted(o['index'] for o in objects) != list(range(48))
            or len({o['object_id'] for o in objects}) != 48
            or any(o['object_id'] != f"point_prior_{o['index']:03d}" for o in objects)
            or any(o['role'] != ('CALIBRATION' if o['index'] < 16 else 'RESERVED_EVALUATION') for o in objects)
            or plan['roles'] != dict(CALIBRATION=list(range(16)), RESERVED_EVALUATION=list(range(16,48)))):
        raise ValueError('Exact independent48 object identities and frozen16/32 roles required')
    if role == 'CALIBRATION':
        if release is not None or calibration_freeze is not None:
            raise ValueError('Calibration must not consume a reserved release')
        indices = list(range(16))
    elif role == 'RESERVED_EVALUATION':
        if (not isinstance(release, dict) or not isinstance(calibration_freeze, dict)
                or release.get('schema') != 'point-prior-reserved-release-v0.1'
                or release.get('status') != 'FROZEN_EXPLICIT_RESERVED_EVALUATION_RELEASE'
                or release.get('method_sha256') != method_sha256
                or release.get('execution_protocol_sha256') != execution_sha256
                or release.get('object_indices') != list(range(16,48))
                or release.get('calibration_freeze') != calibration_freeze_record
                or calibration_freeze.get('completed') is not True
                or calibration_freeze.get('method_sha256') != method_sha256
                or calibration_freeze.get('execution_protocol_sha256') != execution_sha256
                or calibration_freeze.get('calibration_object_indices') != list(range(16))):
            raise ValueError('Hash-bound complete calibration freeze and explicit reserved release required')
        indices = list(range(16,48))
    else:
        raise ValueError('Only explicit CALIBRATION or RESERVED_EVALUATION role allowed')
    return [next(o for o in objects if o['index'] == i) for i in indices]


def capture_pairs(plan, objects, *, root=ROOT):
    """Require ALL selected pairs complete; only selected-role metadata is read.

    No NPZ member is opened here. Byte hashes verify immutable trajectories.
    A missing/failed slot stops the entire stage before any episode is analyzed.
    """
    pairs = {}
    for obj in objects:
        cells = [c for c in plan['cells'] if c['object_id'] == obj['object_id']]
        expected = [obj['object_id']+'__'+label for label in ('A','B')]
        if (len(cells) != 2 or sorted(c['id'] for c in cells) != expected
                or any(c['role'] != obj['role'] for c in cells)):
            raise ValueError('Exact two complete role-matched A/B cells required')
        pair = {}
        for label, cell in zip(('A','B'), sorted(cells, key=lambda c: c['id'])):
            protocol = read(checked(cell['protocol'], root))
            if (protocol['cell_id'] != cell['id'] or protocol['output'] != cell['output']
                    or protocol['evaluation_role'] != obj['role']
                    or protocol['integration_hz'] != 120):
                raise ValueError('Capture identity/role/common120Hz mismatch')
            directory = resolve(cell['output'], root)
            execution, summary = read(directory/'execution.json'), read(directory/'summary.json')
            trajectory = directory/'trajectory.npz'
            if (execution.get('terminal') is not True or execution.get('completed') is not True
                    or summary.get('completed') is not True or summary.get('frames') != 61
                    or summary.get('steps') != 360 or sha(trajectory) != summary['trajectory_sha256']):
                raise ValueError('ALL selected-role finite pairs must be complete without replacement')
            pair[label] = dict(trajectory=record(trajectory, root),
                execution=record(directory/'execution.json', root), summary=record(directory/'summary.json', root),
                protocol=cell['protocol'])
        pairs[obj['index']] = pair
    return pairs


def weighted_displacement_moments(displacements, weights, *, chunk=512):
    """Stable two-pass moments of DISPLACEMENTS, never E[x²]-E[x]².

    No shared world rest coordinates enter this calculation. Subtract a fixed
    particle first, then center about the computed weighted displacement mean.
    Returns (mean_displacement, latent_std), with target shape [time,nodes,3].
    """
    u, w = np.asarray(displacements), np.asarray(weights, dtype=np.float64)
    if (u.ndim != 4 or u.shape[-1] != 3 or u.dtype != np.dtype('float64')
            or w.shape != (len(u),) or not len(u) or not np.isfinite(w).all()
            or (w < 0).any() or abs(w.sum()-1.) > 64*np.finfo(float).eps
            or type(chunk) is not int or chunk < 1):
        raise ValueError('Finite float64 displacement particles and normalized weights required')
    anchor = u[0].copy()
    delta_mean = np.zeros_like(anchor)
    for first in range(0,len(u),chunk):
        block = u[first:first+chunk]-anchor
        if not np.isfinite(block).all():
            raise ValueError('Nonfinite displacement particles')
        delta_mean += np.einsum('p,ptnc->tnc', w[first:first+chunk], block, optimize=True)
    variance = np.zeros_like(anchor)
    for first in range(0,len(u),chunk):
        centered = (u[first:first+chunk]-anchor)-delta_mean
        variance += np.einsum('p,ptnc->tnc', w[first:first+chunk], centered*centered, optimize=True)
    return anchor+delta_mean, np.sqrt(variance)


def target_metrics(truth_a, truth_b, mean_a, mean_b, std_a, std_b, *, after_view):
    """One state's declared A/B union, with both A errors explicitly reported."""
    start = 31 if after_view else 21
    score = max(object_score(truth_a[start:], mean_a[start:], std_a[start:]),
                object_score(truth_b, mean_b, std_b))
    # Width summaries use the SAME action-fair window for all five states,
    # while the no-action simultaneous score additionally covers A21..30.
    scale = np.concatenate((std_a[31:].reshape(-1), std_b.reshape(-1)))
    return dict(A_after1s_MSE_m2=float(np.mean((truth_a[21:]-mean_a[21:])**2)),
        A_after1p5s_MSE_m2=float(np.mean((truth_a[31:]-mean_a[31:])**2)),
        B_all_MSE_m2=float(np.mean((truth_b-mean_b)**2)), individual_band_score=score,
        mean_target_scale_m=float(np.mean(np.maximum(scale, 1e-9))),
        latent_target_std_RMS_m=float(np.sqrt(np.mean(scale**2))),
        band_target_A_frames=list(range(start,61)), band_target_B_frames=list(range(61)))


def _expected_source_pins(plan, cache):
    sources = {**plan['source_sha256'], **cache['source_sha256']}
    for name in SOURCE_NAMES:
        sources['experiments/'+name] = sha(ROOT/'experiments'/name)
    return sources


def prepare_execution(cache_path, cache_sha256, protocol_output, analysis_output, *, device):
    method = read(checked(dict(path=METHOD_PATH, sha256=METHOD_SHA256)))
    plan = read(checked(method['cohort_plan']))  # metadata/record paths only
    cache = read(checked(dict(path=cache_path, sha256=cache_sha256)))
    if not cache.get('completed') or not str(device).startswith('cuda:'):
        raise ValueError('Complete fixed cache and explicit selected CUDA device required')
    if resolve(analysis_output).exists() or resolve(protocol_output).exists():
        raise ValueError('Fresh protocol and analysis output required')
    # Every source affecting analysis or its fixed aggregate/release is bound.
    sources = _expected_source_pins(plan,cache)
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    for rec in method['base_inference']['library_references']:
        checked(rec)  # bytes only; no NPZ arrays, old LL or weights
    checked(SUPPORT_PRIOR)
    for rec in (plan['public_manifest'], plan['public_hardware'], plan['public_acquired_queries']):
        checked(rec)
    if plan['public_acquired_queries']['sha256'] != ACQUIRED_QUERY_SHA:
        raise ValueError('Original public rest72ray acquired recipe required')
    eligible_objects(plan, 'CALIBRATION', execution_sha256='', method_sha256=METHOD_SHA256)
    result = dict(schema=SCHEMA, status='FROZEN_POINT_PRIOR_ANALYSIS_EXECUTION',
        method=dict(path=METHOD_PATH, sha256=METHOD_SHA256), cohort_plan=method['cohort_plan'],
        cache=dict(path=cache_path, sha256=cache_sha256), support_prior=SUPPORT_PRIOR,
        prior_libraries=method['base_inference']['library_references'], source_sha256=sources,
        public_manifest=plan['public_manifest'], analysis_output=analysis_output,
        device=device, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cpu_threads=2, forward_particle_chunk=512, particle_count=2048,
        roles=plan['roles'], raw_artifacts_LOCAL=True,
        cohort_response_arrays_or_truth_opened=False,
        reserved_requires_separate_hash_bound_calibration_freeze_and_release=True)
    path = resolve(protocol_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, result)
    return record(path)


def validate_execution(path, digest, *, check_environment=True):
    """Metadata/hash validator, also usable by aggregate/release code.

    ``check_environment=False`` allows a CPU-only report process to validate
    the SAME complete dependency closure without inheriting the GPU mapping.
    """
    protocol = read(checked(dict(path=path, sha256=digest)))
    if (protocol.get('schema') != SCHEMA or protocol.get('status') != 'FROZEN_POINT_PRIOR_ANALYSIS_EXECUTION'
            or protocol.get('method') != dict(path=METHOD_PATH, sha256=METHOD_SHA256)
            or protocol.get('cpu_threads') != 2 or protocol.get('forward_particle_chunk') != 512
            or protocol.get('particle_count') != 2048 or protocol.get('support_prior') != SUPPORT_PRIOR
            or (check_environment and protocol.get('CUDA_VISIBLE_DEVICES') != os.environ.get('CUDA_VISIBLE_DEVICES'))):
        raise ValueError('Frozen execution settings/environment mismatch')
    method = read(checked(protocol['method']))
    if (protocol['cohort_plan'] != method['cohort_plan']
            or protocol['prior_libraries'] != method['base_inference']['library_references']):
        raise ValueError('Execution does not bind the frozen method inputs')
    cache = read(checked(protocol['cache']))
    plan = read(checked(protocol['cohort_plan']))
    if (cache.get('completed') is not True
            or protocol.get('source_sha256') != _expected_source_pins(plan,cache)
            or protocol.get('public_manifest') != plan['public_manifest']
            or protocol.get('roles') != plan['roles']):
        raise ValueError('Complete exact source dependency closure and public role inputs required')
    resolve(protocol['analysis_output'])
    if not isinstance(protocol.get('device'),str) or not protocol['device'].startswith('cuda:'):
        raise ValueError('Explicit frozen CUDA device required')
    for name, source_sha in protocol['source_sha256'].items():
        checked(dict(path=name, sha256=source_sha))
    return protocol, method, plan


def _load_positions_subset(trajectory, frames, rest):
    """Only requested frames leave this adapter; no future frame is returned."""
    if any(type(i) is not int or not 0 <= i <= 60 for i in frames):
        raise ValueError('Explicit allowed frame indices required')
    with np.load(checked(trajectory), allow_pickle=False) as data:
        # NPZ positions member decompresses whole; select immediately, do not
        # hand a full early trajectory to a sensor, inference or active policy.
        selected = data['positions_m'][frames].copy()
        times = data['times_s']
    if (selected.shape != (len(frames),len(rest),3) or not np.isfinite(selected).all()
            or not np.array_equal(times, np.arange(61)/20)):
        raise ValueError('Finite61-frame common20Hz capture required')
    positions = np.broadcast_to(rest,(1,61,*rest.shape)).copy()
    positions[0,frames] = selected
    return positions


def _sensor(method, dimension):
    s = method['working_sensor']
    return CensoredRangeModel(np.eye(dimension)*s['range_std_m']**2,
        s['visibility_error_probability'], s['background_std_m'], s['outlier_probability'])


def _actual_observation(trajectory, frames, rest, faces, queries, model, seed, device):
    import torch
    try:
        from campaign_sparse_depth_queries import predict_ranges
    except ImportError:
        from experiments.campaign_sparse_depth_queries import predict_ranges
    positions = _load_positions_subset(trajectory, frames, rest)
    with torch.no_grad():
        predicted = predict_ranges(positions, faces, queries, device=device, particle_chunk=64)
    noisy, _ = model.fantasies(predicted, count=1, seed=seed)
    return CensoredRanges(noisy.visible[0], noisy.residual_m[0])


def _assert_sources(protocol):
    for name, digest in protocol['source_sha256'].items():
        checked(dict(path=name, sha256=digest))


def _save_observation(path, observation):
    np.savez_compressed(path, visible=observation.visible, residual_m=observation.residual_m)


def analyze_episode(obj, pair, context, protocol, execution_sha256, destination):
    """Execute one role-authorized episode; never called before pair preflight."""
    import torch
    try:
        from campaign_exciting_commands import make_exciting_commands, PROGRAM_IDS
        from campaign_sparse_depth_queries import predict_ranges
    except ImportError:
        from experiments.campaign_exciting_commands import make_exciting_commands, PROGRAM_IDS
        from experiments.campaign_sparse_depth_queries import predict_ranges
    forward, public, libraries, acquired_queries, candidates, method = context
    device = protocol['device']
    rest, faces = np.asarray(public['rest_points_m']), np.asarray(public['surface_faces'])
    index = obj['index']
    started = time.monotonic()
    destination.mkdir(parents=True, exist_ok=False)
    observation = _actual_observation(pair['A']['trajectory'], [10,20], rest, faces,
        acquired_queries, _sensor(method,72), 2026113000+index, device)
    _save_observation(destination/'early-observation.npz', observation)
    posterior = condition_reference(libraries, observation)
    empirical = posterior['empirical_posterior']
    np.savez_compressed(destination/'empirical-posterior.npz', **{
        k: empirical[k] for k in ('reference_indices','geometry_ids','physical_coordinates','weights')})
    raw = {}
    for seed in posterior['seeds']:
        raw[f"seed{seed['seed']}_new_log_likelihood"] = seed['log_likelihood']
        for prefix in seed['prefixes']:
            label = f"seed{seed['seed']}_count{prefix['count_per_mask']}"
            raw[label+'_indices'], raw[label+'_weights'] = prefix['indices'], prefix['weights']
    np.savez_compressed(destination/'reference-new-likelihood-weights.npz', **raw)
    write(destination/'reference-diagnostics.json', dict(summaries=posterior['summaries'],
        stability=posterior['stability'], old_likelihood_or_posterior_reused=False))
    displacements = {}
    # Each program/particle is predicted exactly once, in fixed512 chunks.
    for label, program in zip(('A','B'), PROGRAM_IDS):
        q = np.empty((2048,61,len(rest),3), dtype=np.float64)
        clock = make_exciting_commands(program)
        for first in range(0,2048,512):
            with torch.no_grad():
                output = forward.predict(clock,
                    torch.tensor(empirical['geometry_ids'][first:first+512].copy(), dtype=torch.int64),
                    torch.tensor(empirical['physical_coordinates'][first:first+512].copy(), dtype=torch.float64), device=device)
            q[first:first+512] = output['positions_m'].cpu().numpy()-rest
            del output
        displacements[label] = q
    # Candidate rays are forward predictions only; no actual A24 is read yet.
    predicted_candidates = []
    for query in candidates:
        visible, residual = [], []
        for first in range(0,2048,512):
            positions = np.broadcast_to(rest,(min(512,2048-first),61,*rest.shape)).copy()
            positions[:,24] += displacements['A'][first:first+512,24]
            with torch.no_grad():
                rays = predict_ranges(positions, faces, [query], device=device, particle_chunk=64)
            visible.append(rays.visible)
            residual.append(rays.residual_m)
        predicted_candidates.append(CensoredRanges(np.concatenate(visible), np.concatenate(residual)))
    target = displacements['A'][:,31:].reshape(2048,-1)
    choice = select_point_active_view(target, predicted_candidates, _sensor(method,36), device=device)
    np.savez_compressed(destination/'candidate-predicted-rays.npz',
        visible=np.stack([p.visible for p in predicted_candidates]),
        residual_m=np.stack([p.residual_m for p in predicted_candidates]))
    _assert_sources(protocol)
    choice.update(object_index=index, role=obj['role'], method_sha256=METHOD_SHA256,
        execution_protocol_sha256=execution_sha256,
        empirical_posterior=record(destination/'empirical-posterior.npz'),
        early_observation=record(destination/'early-observation.npz'),
        source_sha256=protocol['source_sha256'], actual_frame24_opened=False,
        selected_random_view=int(np.random.default_rng(2026117000+index).integers(4)))
    write(destination/'active-choice-frozen.json', choice)
    # ONLY NOW read A24, producing four potential observations for fixed-camera
    # counterfactuals. The selected policy receives only its chosen view update.
    weights = [np.full(2048,1./2048)]
    view_diagnostics = []
    for j, (query, prediction) in enumerate(zip(candidates,predicted_candidates)):
        actual = _actual_observation(pair['A']['trajectory'], [24], rest, faces, [query],
            _sensor(method,36), 2026119000+4*index+j, device)
        _save_observation(destination/f'view{j}-observation.npz', actual)
        w, diagnostic = update_after_view(prediction, actual, _sensor(method,36))
        weights.append(w)
        view_diagnostics.append(diagnostic)
    states, payload = {}, {}
    for name, w in zip(STATE_NAMES,weights):
        states[name] = {}
        for program in ('A','B'):
            mean, std = weighted_displacement_moments(displacements[program], w)
            states[name][program] = (mean, std)
            payload[name+'__'+program+'__mean_displacement_m'] = mean
            payload[name+'__'+program+'__latent_std_m'] = std
        payload[name+'__weights'] = w
    np.savez_compressed(destination/'predictive-states.npz', rest_points_m=rest,
        times_s=np.arange(61)/20, **payload)
    _assert_sources(protocol)
    frozen = dict(object_index=index, role=obj['role'], method_sha256=METHOD_SHA256,
        execution_protocol_sha256=execution_sha256, source_sha256=protocol['source_sha256'],
        active_choice=record(destination/'active-choice-frozen.json'),
        states=record(destination/'predictive-states.npz'), state_names=list(STATE_NAMES),
        future_A_or_B_truth_opened=False, material_or_support_truth_opened=False,
        state_target='No action A21..60/B0..60; each extra view A31..60/B0..60')
    write(destination/'posterior-states-frozen.json', frozen)
    del displacements, target
    # Logical truth gate: all five fixed predictive states and action are saved
    # and source-verified BEFORE future targets/parameters are supplied here.
    truth = {}
    for program in ('A','B'):
        truth[program] = _load_positions_subset(pair[program]['trajectory'], list(range(61)), rest)[0]-rest
    spec = read(checked(obj['realized_hypothesis']))
    node = int(spec['fixture_nodes'][0])
    values = [spec['material'][k] for k in ('youngs_modulus_Pa','density_kg_m3',
        'poissons_ratio','body_linear_drag_s_inverse')]
    summary = next(r for r in posterior['summaries']
        if r['seed'] == libraries.primary_seed and r['count_per_mask'] == max(libraries.counts_per_mask))
    category = next(c for c in summary['support_categories'] if c['node_id'] == node)
    metrics = {}
    for j, name in enumerate(STATE_NAMES):
        (ma,sa), (mb,sb) = states[name]['A'], states[name]['B']
        metrics[name] = target_metrics(truth['A'],truth['B'],ma,mb,sa,sb,after_view=j>0)
        if j:
            metrics[name]['view_update_ESS'] = view_diagnostics[j-1]['effective_sample_size']
    result = dict(schema='point-prior-analysis-object-v0.1', object_index=index,
        object_id=obj['object_id'], role=obj['role'], completed=True,
        method_sha256=METHOD_SHA256, execution_protocol_sha256=execution_sha256,
        posterior_frozen_before_future_truth=True, posterior_freeze=record(destination/'posterior-states-frozen.json'),
        capture_pairs=pair, states=metrics, joint_band_score=max(m['individual_band_score'] for m in metrics.values()),
        selected_active_view=choice['selected_index'], selected_random_view=choice['selected_random_view'],
        active_choice=record(destination/'active-choice-frozen.json'), view_update_diagnostics=view_diagnostics,
        reference_stability=posterior['stability'], base_posterior_summary=summary,
        rawmaterialintervalchecks=[dict(name=r['name'], truth=float(v),
            in_90_percent_interval=bool(r['quantile05'] <= v <= r['quantile95'])) for r,v in zip(summary['material'],values)],
        supporttruthprobability=category['posterior_probability'],
        supporttruthpriorprobability=category['prior_probability'], true_support_node=node,
        supporttruthrank=1+sorted(summary['support_categories'],key=lambda c:-c['posterior_probability']).index(category),
        elapsed_s=time.monotonic()-started, raw_outputs_LOCAL=True,
        logical_gate_not_literal_NPZ_decompression_isolation=True,
        scope='Fixed H0/known point-pin type; unknown node/material; working approximate forward/sensor model')
    write(destination/'result.json', result)
    return result


def run_analysis(execution_path, execution_sha256, *, role,
                 release_path=None, release_sha256=None):
    protocol, method, plan = validate_execution(execution_path, execution_sha256)
    release, freeze, freeze_record = None, None, None
    if role == 'RESERVED_EVALUATION':
        if release_path is None or release_sha256 is None:
            raise ValueError('Reserved evaluation requires explicit release path AND hash')
        release = read(checked(dict(path=release_path,sha256=release_sha256)))
        freeze_record = release.get('calibration_freeze')
        if not isinstance(freeze_record,dict):
            raise ValueError('Hash-bound calibration freeze missing')
        freeze = read(checked(freeze_record))
        if len(freeze.get('calibration_results',[])) != 16:
            raise ValueError('All16 hash-bound calibration result records required')
        for rec in freeze['calibration_results']:
            checked(rec)
        for name,digest in freeze.get('source_sha256',{}).items():
            checked(dict(path=name,sha256=digest))
    elif release_path is not None or release_sha256 is not None:
        raise ValueError('Calibration cannot open a reserved release')
    objects = eligible_objects(plan,role,execution_sha256=execution_sha256,
        method_sha256=METHOD_SHA256,release=release,calibration_freeze=freeze,
        calibration_freeze_record=freeze_record)
    pairs = capture_pairs(plan,objects)  # ALL role pairs before ANY analysis
    import torch
    try:
        from campaign_support_basis_cache import load_support_basis_cache
        from campaign_sparse_depth_queries import make_queries
        from campaign_exciting_commands import make_exciting_commands, PROGRAM_IDS
    except ImportError:
        from experiments.campaign_support_basis_cache import load_support_basis_cache
        from experiments.campaign_sparse_depth_queries import make_queries
        from experiments.campaign_exciting_commands import make_exciting_commands, PROGRAM_IDS
    if any(os.environ.get(k) != '2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('CPU2 thread environment required')
    torch.set_num_threads(2)
    output = resolve(protocol['analysis_output'])
    status_path = output/('calibration-execution.json' if role == 'CALIBRATION' else 'reserved-execution.json')
    if status_path.exists() or any((output/'objects'/o['object_id']).exists() for o in objects):
        raise ValueError('Existing attempt retained; no overwrite/resume/retry')
    public_manifest = read(checked(protocol['public_manifest']))
    if protocol['public_manifest'] != plan['public_manifest']:
        raise ValueError('Public context not from the frozen cohort')
    forward, public, ids, distribution = load_support_basis_cache(protocol['cache']['path'],protocol['cache']['sha256'])
    if public != read(checked(plan['public_hardware'])):
        raise ValueError('Cache/cohort full public hardware differs')
    if distribution != read(checked(public_manifest['support_prior'])):
        raise ValueError('Cache/cohort exact prior differs')
    if plan['public_acquired_queries']['sha256'] != ACQUIRED_QUERY_SHA:
        raise ValueError('Original acquired72 rays required')
    queries = read(checked(plan['public_acquired_queries']))
    libraries = load_reference_libraries()
    if set(ids) != set(libraries.probabilities):
        raise ValueError('Complete all55 same prior categories required')
    # Public clock bytes, never true material/support, determine both controls.
    for program in PROGRAM_IDS:
        clock = make_exciting_commands(program)
        with np.load(checked(public_manifest['known_commands'][program]),allow_pickle=False) as data:
            names = ('endpoint_times_s','output_times_s','handle_forces_world_N',
                'handle_torques_world_Nm','nodal_forces_world_N','body_acceleration_world_m_s2')
            if set(data.files) != set(names) or any(not np.array_equal(data[name],getattr(clock,name)) for name in names):
                raise ValueError('Known public command bytes differ from frozen library controls')
    candidates = make_queries(public['rest_points_m'],public['surface_faces'],
        frame_indices=(24,),offsets_deg=((-60.,10.),(-20.,10.),(20.,10.),(60.,10.)),grid_size=6)
    context = (forward,public,libraries,queries,candidates,method)
    output.mkdir(parents=True,exist_ok=True)
    status = dict(pid=os.getpid(),role=role,terminal=False,completed=False,
        method_sha256=METHOD_SHA256,execution_protocol_sha256=execution_sha256,
        planned_object_indices=[o['index'] for o in objects], completed_object_indices=[],
        unlaunched_object_indices=[o['index'] for o in objects], failures=[])
    write(status_path,status)
    started = time.monotonic()
    for obj in objects:
        status['active_object_index'] = obj['index']
        status['unlaunched_object_indices'].remove(obj['index'])
        write(status_path,status)
        destination = output/'objects'/obj['object_id']
        try:
            _assert_sources(protocol)
            result = analyze_episode(obj,pairs[obj['index']],context,protocol,execution_sha256,destination)
            status['completed_object_indices'].append(obj['index'])
            print(json.dumps(dict(object_index=obj['index'],role=role,completed=True,
                elapsed_s=result['elapsed_s'],selected_active_view=result['selected_active_view'])),flush=True)
        except BaseException as exc:
            failure = dict(object_index=obj['index'],role=role,completed=False,
                error=type(exc).__name__,message=str(exc),method_sha256=METHOD_SHA256,
                execution_protocol_sha256=execution_sha256)
            if destination.exists() and not (destination/'result.json').exists():
                write(destination/'failure.json',failure)
            status['failures'].append(failure)
            status.update(terminal=True,completed=False,elapsed_s=time.monotonic()-started)
            write(status_path,status)
            raise
        status['elapsed_s'] = time.monotonic()-started
        write(status_path,status)
    status.update(terminal=True,completed=True,active_object_index=None,elapsed_s=time.monotonic()-started)
    write(status_path,status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode',required=True)
    prep = sub.add_parser('prepare-execution')
    for name in ('cache','cache-sha256','protocol-output','analysis-output'):
        prep.add_argument('--'+name,required=True)
    prep.add_argument('--device',required=True)
    for name in ('run-calibration','run-reserved'):
        p = sub.add_parser(name)
        p.add_argument('--execution',required=True)
        p.add_argument('--execution-sha256',required=True)
        if name == 'run-reserved':
            p.add_argument('--release',required=True)
            p.add_argument('--release-sha256',required=True)
    args = parser.parse_args()
    if args.mode == 'prepare-execution':
        result = prepare_execution(args.cache,args.cache_sha256,args.protocol_output,args.analysis_output,device=args.device)
    else:
        result = run_analysis(args.execution,args.execution_sha256,
            role='CALIBRATION' if args.mode == 'run-calibration' else 'RESERVED_EVALUATION',
            release_path=getattr(args,'release',None),release_sha256=getattr(args,'release_sha256',None))
    print(json.dumps(result))
