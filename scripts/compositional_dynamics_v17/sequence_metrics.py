"""Matched-time dynamic geometry metrics; no truth alignment in world errors.

Rigid alignment is a separate shape diagnostic, never an adjustment to rendering
or world-position errors. Missing prediction tails are counted, not filled.
"""
import numpy as np


def _cloud(x, name):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError(f'{name} must be a finite (N,3) cloud')
    return x


def proper_alignment(source, target, weights):
    source = _cloud(source, 'source'); target = _cloud(target, 'target')
    w = np.asarray(weights, dtype=float)
    if source.shape != target.shape or w.shape != (len(source),) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Matched material points and positive finite mass weights required')
    w = w / w.sum(); a = w @ source; b = w @ target
    U, _, Vt = np.linalg.svd((source-a).T @ ((target-b)*w[:, None]))
    correction = np.eye(3); correction[-1, -1] = np.linalg.det(U @ Vt)
    rotation = U @ correction @ Vt
    return (source-a) @ rotation+b, rotation, b-a @ rotation


def matched_sequence_metrics(rest, reference_times, reference_positions,
                             prediction_times, prediction_positions, masses,
                             *, characteristic_length_m):
    rest = _cloud(rest, 'rest')
    t = np.asarray(reference_times, dtype=float); pt = np.asarray(prediction_times, dtype=float)
    ref = np.asarray(reference_positions, dtype=float); pred = np.asarray(prediction_positions, dtype=float)
    m = np.asarray(masses, dtype=float); L = float(characteristic_length_m)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError('Increasing finite reference times required')
    if ref.shape != (len(t), len(rest), 3) or not np.isfinite(ref).all():
        raise ValueError('Complete finite material-corresponded reference required')
    if pt.ndim != 1 or pred.shape != (len(pt), len(rest), 3) or len(pt) > len(t) or not np.isfinite(pred).all():
        raise ValueError('Prediction must be an explicitly finite prefix, not NaN-filled')
    if not np.array_equal(pt, t[:len(pt)]):
        raise ValueError('Exactly matched sample times required; no interpolation or time fitting')
    if m.shape != (len(rest),) or not np.isfinite(m).all() or (m <= 0).any() or not np.isfinite(L) or L <= 0:
        raise ValueError('Positive masses and undeformed characteristic length required')
    w = m/m.sum(); count = len(pt)
    rows = []
    for i in range(count):
        aligned, _, _ = proper_alignment(pred[i], ref[i], w)
        rigid_ref, _, _ = proper_alignment(rest, ref[i], w)
        rigid_pred, _, _ = proper_alignment(rest, pred[i], w)
        rms = lambda d: float(np.sqrt(np.sum(w[:, None]*d*d)))
        rows.append(dict(time_s=float(t[i]), world_point_rms_m=rms(pred[i]-ref[i]),
                         shape_point_rms_m=rms(aligned-ref[i]),
                         com_error_m=float(np.linalg.norm(w @ (pred[i]-ref[i]))),
                         reference_nonrigid_rms_m=rms(ref[i]-rigid_ref),
                         predicted_nonrigid_rms_m=rms(pred[i]-rigid_pred)))
    result = dict(expected_frames=len(t), predicted_frames=count,
                  complete=count == len(t), frame_coverage=count/len(t),
                  first_missing_time_s=float(t[count]) if count < len(t) else None,
                  characteristic_length_m=L, rows=rows,
                  rms_definition='Positive-point-weighted vector RMS, not coordinate RMS or full consistent-FEM mass norm; uniform recorded-frame aggregation including initial frame.',
                  weight_scope='Caller-supplied positive point weights; with nodal mass row sums this is a lumped-node diagnostic. COM is computed from the same specified weights.',
                  alignment_scope='Proper rigid alignment only for shape error and deformation census; world errors unchanged.')
    if count:
        for field in ('world_point_rms_m', 'shape_point_rms_m', 'com_error_m',
                      'reference_nonrigid_rms_m', 'predicted_nonrigid_rms_m'):
            result[field] = float(np.sqrt(np.mean([r[field]**2 for r in rows])))
        result['world_coordinate_rms_m'] = result['world_point_rms_m']/np.sqrt(3.)
        result['shape_coordinate_rms_m'] = result['shape_point_rms_m']/np.sqrt(3.)
        result['world_error_over_length'] = result['world_point_rms_m']/L
        result['reference_nonrigid_over_length'] = result['reference_nonrigid_rms_m']/L
        denominator = result['reference_nonrigid_rms_m']
        result['world_error_over_reference_deformation'] = result['world_point_rms_m']/denominator if denominator > 1e-10*L else None
        result['aggregation_scope'] = 'Entire trajectory' if result['complete'] else 'Covered prefix only; incomplete trajectory remains a failure'
    return result
