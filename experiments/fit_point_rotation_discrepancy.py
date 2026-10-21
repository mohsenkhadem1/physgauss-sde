"""Fit a joint working discrepancy from OLD calibration objects, now development.

No NEW-cohort responses/labels, inverse fitting, observation selection or finite
runs. Known-parameter forward residuals define shared range/target factors.
Missing ranges are not measurements: available-case centered factor rows use
only paired foreground hits, with their counts retained. This PSD working
covariance is approximate under missingness/nonlinear rendering. It is not a
calibrated physical posterior; independent new objects must test it.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
try:
    import run_point_prior_analysis as old
    from campaign_batched_point_rotation import retract_positions
    from campaign_support_basis_cache import load_support_basis_cache
    from campaign_exciting_commands import make_exciting_commands, PROGRAM_IDS
    from campaign_sparse_depth_queries import make_queries, predict_ranges
except ImportError:
    from experiments import run_point_prior_analysis as old
    from experiments.campaign_batched_point_rotation import retract_positions
    from experiments.campaign_support_basis_cache import load_support_basis_cache
    from experiments.campaign_exciting_commands import make_exciting_commands, PROGRAM_IDS
    from experiments.campaign_sparse_depth_queries import make_queries, predict_ranges

OLD_EXECUTION = 'runs/operator-dev/point-prior-analysis-v01/execution-protocol.json'
OLD_EXECUTION_SHA = 'fabad32cb79a94bde93890506f8862f0ea00d9535f001bfad026236aaa08242c'
OLD_SEAL = dict(path='runs/gaussian-campaign-v01/point-prior-analysis-v01/reserved-evaluation.json',
    sha256='51fcbd0e93e92e2a6f80ab996c45c30b21e73c57c797cd436bca4549b733240d')
TARGET_SHAPE = (2, 61, 455, 3)


def joint_factors(target_errors, range_errors, paired_hits):
    """Shared object columns, no per-channel eigenvalue repair or fabricated hit.

    Target: sample mean/factor with1/sqrt(n-1). Range: available-case mean and
    1/sqrt(n_valid-1) loading. Their concatenation defines one PSD Gaussian
    covariance even with missing entries; it is an explicit working estimator.
    Unknown range channels get zero learned mean/loading, NOT a coverage claim.
    """
    t, r, h = np.asarray(target_errors), np.asarray(range_errors), np.asarray(paired_hits)
    if (t.dtype != np.float64 or r.dtype != np.float64 or h.dtype != bool
        or t.ndim != 2 or r.ndim != 2 or h.shape != r.shape or len(t) != len(r)
        or len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(r[h]).all()):
        raise ValueError('Matching float64 object-by-feature errors and boolean paired-hit mask required')
    tm = t.mean(0); tf = ((t-tm)/np.sqrt(len(t)-1)).T
    counts = h.sum(0); safe = np.where(h, r, 0.)
    rm = np.divide(safe.sum(0), counts, out=np.zeros(r.shape[1]), where=counts > 0)
    centered = np.where(h, safe-rm, 0.)
    rf = (centered/np.sqrt(np.maximum(counts-1, 1))).T
    return dict(target_mean_m=tm, target_factor_m=tf, range_mean_m=rm,
        range_factor_m=rf, paired_hit_count=counts)


def prepare(output):
    old.checked(OLD_SEAL)  # Old evaluation must already be sealed, before repurposing training data.
    execution, _, plan = old.validate_execution(OLD_EXECUTION, OLD_EXECUTION_SHA, check_environment=False)
    objects = old.eligible_objects(plan, 'CALIBRATION', execution_sha256=OLD_EXECUTION_SHA,
        method_sha256=execution['method']['sha256'])
    pairs = old.capture_pairs(plan, objects)  # ALL16 pairs, metadata/digests only.
    sources = dict(execution['source_sha256'])
    for name in ['fit_point_rotation_discrepancy.py', 'campaign_batched_point_rotation.py']:
        path = old.ROOT/'experiments'/name; sources[path.relative_to(old.ROOT).as_posix()] = old.sha(path)
    protocol = dict(schema='point-rotation-discrepancy-fit-v0.2',
        status='FROZEN_BEFORE_KNOWN_PARAMETER_FORWARD_QUERIES', old_seal=OLD_SEAL,
        old_execution=dict(path=OLD_EXECUTION, sha256=OLD_EXECUTION_SHA),
        training_role='Previous CALIBRATION objects0..15 repurposed as DEVELOPMENT_DISCREPANCY_FIT',
        object_indices=list(range(16)), source_sha256=sources, cache=execution['cache'],
        targets=dict(shape=list(TARGET_SHAPE), order='A/B,all61times,455nodes,xyz'),
        range_order=dict(base_rows=[0,72], view_rows=[[72+36*j,108+36*j] for j in range(4)],
            frames=[10,20,24], candidate_offsets_deg=[[-60,10],[-20,10],[20,10],[60,10]]),
        estimator='Empirical shared object-column Gaussian factors; available-case range rows',
        target_constraint='Future caller projects target mean/factors to exact zero at each candidate point pin',
        sensor_std_m=.0002, visibility_error_probability=.005, background_std_m=.05, outlier_probability=0.,
        CPU_threads=2, CUDA_VISIBLE_DEVICES='1', device='cuda:0',
        output=output, fit_objects=objects, capture_pairs=pairs,
        new_cohort_truth_or_response_values_allowed=False, inference_or_finite_runs=0,
        limitation='Approximate fixed joint discrepancy, independent of material/support except target pin projection; needs fresh calibration and reserved evaluation.')
    out = old.resolve(output); out.mkdir(parents=True, exist_ok=False)
    old.write(out/'protocol.json', protocol)
    return old.record(out/'protocol.json')


def fit(protocol_path, digest):
    p = old.read(old.checked(dict(path=protocol_path, sha256=digest)))
    if p['schema'] != 'point-rotation-discrepancy-fit-v0.2' or p['object_indices'] != list(range(16)):
        raise ValueError('Exact declared old-development fit required')
    old.checked(p['old_seal'])
    for path, sha in p['source_sha256'].items(): old.checked(dict(path=path, sha256=sha))
    execution, _, plan = old.validate_execution(p['old_execution']['path'], p['old_execution']['sha256'], check_environment=False)
    objects = old.eligible_objects(plan, 'CALIBRATION', execution_sha256=OLD_EXECUTION_SHA,
        method_sha256=execution['method']['sha256'])
    pairs = old.capture_pairs(plan, objects)
    recorded_pairs = {int(k): v for k, v in p['capture_pairs'].items()}
    if objects != p['fit_objects'] or pairs != recorded_pairs:
        raise ValueError('Complete old-development object/pair binding changed')
    if any(os.environ.get(k) != '2' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']) or os.environ.get('CUDA_VISIBLE_DEVICES') != p['CUDA_VISIBLE_DEVICES']:
        raise ValueError('Declared CPU2/GPU1 environment required')
    out = old.resolve(p['output'])
    if (out/'execution.json').exists(): raise ValueError('Existing fit attempt retained; no overwrite/retry')
    status = dict(pid=os.getpid(), terminal=False, completed=False)
    old.write(out/'execution.json', status); started = time.monotonic(); torch.set_num_threads(2)
    try:
        forward, public, ids, _ = load_support_basis_cache(p['cache']['path'], p['cache']['sha256'])
        if public != old.read(old.checked(plan['public_hardware'])): raise ValueError('Public H0 differs')
        rest = np.asarray(public['rest_points_m']); faces = np.asarray(public['surface_faces'])
        base = old.read(old.checked(plan['public_acquired_queries']))
        views = make_queries(rest, faces, frame_indices=(24,),
            offsets_deg=((-60.,10.),(-20.,10.),(20.,10.),(60.,10.)), grid_size=6)
        # First private labels and arrays occur only AFTER complete source/role preflight.
        specs = [old.read(old.checked(o['realized_hypothesis'])) for o in objects]
        pins = np.asarray([s['fixture_nodes'][0] for s in specs], dtype=np.int64)
        if any(s['boundary_kind'] != 'point_pin' for s in specs) or not set(pins).issubset(set(ids)):
            raise ValueError('Only supplied true point-pin hypotheses from complete catalogue')
        keys = ['youngs_modulus_Pa', 'density_kg_m3', 'poissons_ratio', 'body_linear_drag_s_inverse']
        theta = np.asarray([[s['material'][k] for k in keys] for s in specs], dtype=np.float64)
        target_error = []; range_prediction = range_truth = None; metrics = []
        for program_index, program in enumerate(PROGRAM_IDS):
            with torch.no_grad():
                raw = forward.predict(make_exciting_commands(program), torch.tensor(pins), torch.tensor(theta), device=p['device'])['positions_m']
                corrected = retract_positions(torch.tensor(rest, device=p['device']), raw,
                    torch.tensor(pins, device=p['device']))['positions_m']
                if program_index == 0:
                    range_prediction = predict_ranges(corrected, faces, base+views, device=p['device'], particle_chunk=64)
                predicted = corrected.cpu().numpy()
            label = 'AB'[program_index]; truth = []
            for obj in objects:
                with np.load(old.checked(pairs[obj['index']][label]['trajectory']), allow_pickle=False) as f:
                    q = f['positions_m']; times = f['times_s']
                if q.shape != (61,455,3) or not np.isfinite(q).all() or not np.array_equal(times, np.arange(61)/20):
                    raise ValueError('Full unchanged old-development target required')
                truth.append(q)
            truth = np.asarray(truth); error = truth-predicted; target_error.append(error)
            metrics.append(dict(program=program, objects=16,
                supplied_true_corrected_coordinate_RMSE_m=float(np.sqrt(np.mean(error**2))),
                maximum_coordinate_error_m=float(np.abs(error).max())))
            if program_index == 0:
                range_truth = predict_ranges(truth, faces, base+views, device=p['device'], particle_chunk=64)
        t = np.stack(target_error, axis=1).reshape(16, -1)
        hits = range_prediction.visible & range_truth.visible
        errors = range_truth.residual_m-range_prediction.residual_m
        factors = joint_factors(t, errors, hits)
        archive = out/'joint-discrepancy.npz'; np.savez_compressed(archive, **factors)
        freeze = dict(completed=True, protocol=dict(path=protocol_path, sha256=digest),
            source_sha256=p['source_sha256'], archive=old.record(archive), objects=16, rank_columns=16,
            target_shape=list(TARGET_SHAPE), observation_dimensions=216, known_parameter_controls=32,
            controls=metrics, range_paired_counts=factors['paired_hit_count'].tolist(),
            range_no_pairs=int((factors['paired_hit_count']==0).sum()),
            visibility_mismatches=int((range_prediction.visible != range_truth.visible).sum()),
            range_pair_total=int(hits.sum()),
            range_paired_RMS_m=float(np.sqrt(np.mean(errors[hits]**2))),
            new_cohort_response_or_truth_read=False, fitted_using_noisy_sensor_observations=False,
            fitted_from_supplied_true_physics_only=True, raw_archive_LOCAL=True,
            elapsed_s=time.monotonic()-started, limitation=p['limitation'])
        old.write(out/'fit-frozen.json', freeze)
        status.update(terminal=True, completed=True, elapsed_s=time.monotonic()-started)
        old.write(out/'execution.json', status)
        return old.record(out/'fit-frozen.json')
    except BaseException as exc:
        status.update(terminal=True, completed=False, error=repr(exc), elapsed_s=time.monotonic()-started)
        old.write(out/'execution.json', status); raise


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__); sub = a.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', required=True)
    p = sub.add_parser('fit'); p.add_argument('--protocol', required=True); p.add_argument('--protocol-sha256', required=True)
    x = a.parse_args()
    print(json.dumps(prepare(x.output) if x.action == 'prepare' else fit(x.protocol, x.protocol_sha256)))
