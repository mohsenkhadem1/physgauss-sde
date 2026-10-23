"""Pure PRIVATE evaluation, separate from public inference and action design.

This module performs no I/O. The caller releases only labels and the canonical
target after predictions AND action choices have been sealed. Probe outcomes
are deliberately absent from the accepted truth snapshot. Pointwise intervals
are model conditional, not joint field or population confidence statements.
"""
from __future__ import annotations
import numpy as np
from experiments.campaign_c38_table_validation import POINTS
from experiments.campaign_c38_controlled_observations import POINT_IDS

TRUTH_KEYS = frozenset(('object_id', 'private_configuration_index', 'repeat_index',
                        'young_Pa', 'support_category', 'canonical_target_Fz_N',
                        'canonical_target_point_ids', 'canonical_target_positions_m'))
CATEGORIES = ('both_ends', 'separated_bottom_patches')
PRIMARY_METHOD = 'adaptive'


def truth_snapshot(value):
    if not isinstance(value, dict) or set(value) != TRUTH_KEYS:
        raise ValueError('Labels/canonical target only; no probe outcomes allowed')
    cfg, repeat = value['private_configuration_index'], value['repeat_index']
    if type(cfg) is not int or cfg not in range(4) or type(repeat) is not int or repeat not in range(16):
        raise ValueError('Four fixed configurations, sixteen noise repetitions')
    if value['support_category'] not in CATEGORIES or not 75000 <= value['young_Pa'] <= 150000:
        raise ValueError('Fixed material/support scope')
    if value['canonical_target_Fz_N'] != 1. or not np.array_equal(value['canonical_target_point_ids'], np.arange(63)):
        raise ValueError('Canonical +1N exact63 point target required')
    q = np.asarray(value['canonical_target_positions_m'], dtype=float)
    if q.shape != (63, 3) or not np.isfinite(q).all():
        raise ValueError('Finite canonical target required')
    return q


def position_error(predicted, actual):
    a, b = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Matched finite point arrays required')
    d = a - b
    return dict(coordinate_RMSE_m=float(np.sqrt(np.mean(d*d))),
                maximum_point_error_m=float(np.linalg.norm(d, axis=1).max()),
                coordinate_squared_error_sum_m2=float(np.sum(d*d)),
                coordinate_count=d.size)


def interval_metrics(lower, upper, actual):
    lo, hi, y = (np.asarray(v, dtype=float) for v in (lower, upper, actual))
    if lo.shape != (9,3) or hi.shape != lo.shape or y.shape != lo.shape or not np.isfinite([lo,hi,y]).all() or np.any(lo > hi):
        raise ValueError('Finite ordered pointwise intervals required')
    covered = (lo <= y) & (y <= hi)
    miss = np.maximum(np.maximum(lo-y, y-hi), 0.)
    return dict(covered=covered, covered_count=int(covered.sum()), coordinate_count=27,
                z_covered_count=int(covered[:,2].sum()), z_coordinate_count=9,
                covered_count_by_axis=covered.sum(axis=0), coordinate_count_per_axis=9,
                mean_width_by_axis_m=np.mean(hi-lo,axis=0), maximum_miss_by_axis_m=miss.max(axis=0),
                mean_width_m=float(np.mean(hi-lo)), maximum_miss_m=float(miss.max()),
                miss_m=miss, widths_m=hi-lo,
                scope='Pointwise model-conditional intervals; coordinates correlated; no joint-field or population coverage claim')


def score_prediction(prediction, private, public):
    """Descriptive metrics retain failures; admission requires numerical checks.

    Quantile midpoint intervals and numerical root-enclosure intervals are
    distinct outputs. No truth-dependent widening, noise floors or clipping.
    """
    target = truth_snapshot(private)
    if public['object_id'] != private['object_id'] or prediction.get('object_id', public['object_id']) != public['object_id']:
        raise ValueError('Exact opaque object identity required')
    result = dict(object_id=public['object_id'], private_configuration_index=private['private_configuration_index'],
                  repeat_index=private['repeat_index'], admitted=bool(prediction.get('completed') is True and prediction.get('numerical_controls_passed') is True),
                  numerical_failures=prediction.get('failures', []), primary_method=PRIMARY_METHOD,
                  independent_population_object=False)
    result['rest63'] = position_error(POINTS, target)
    measured = np.asarray(public['observed_positions_m'])[0]
    result['noisy_gravity_persistence9'] = position_error(measured, target[POINT_IDS])
    row = prediction.get('methods', {}).get(PRIMARY_METHOD)
    if not row or not row.get('completed'):
        result['primary_prediction_available'] = False
        result['admitted'] = False
        return result
    result['primary_prediction_available'] = True
    q = np.asarray(row['canonical_mean_positions_m'])
    result['canonical63'] = position_error(q, target)
    result['canonical9'] = position_error(q[POINT_IDS], target[POINT_IDS])
    prob = np.asarray(row['category_probabilities'], dtype=float)
    if prob.shape != (2,) or not np.isfinite(prob).all() or np.any(prob < 0) or abs(prob.sum()-1) > 1e-10:
        raise ValueError('Normalized category probabilities required')
    category = CATEGORIES.index(private['support_category']); true_p = float(prob[category])
    correct = np.eye(2)[category]
    result['support'] = dict(true_category_probability=true_p, categorical_Brier=float(np.sum((prob-correct)**2)),
                             negative_log_probability=float(-np.log(true_p)) if true_p > 0 else None,
                             infinite_negative_log_probability=bool(true_p == 0),
                             predicted_category=CATEGORIES[int(np.argmax(prob))], correct=bool(np.argmax(prob) == category))
    E = float(private['young_Pa']); quant = np.asarray(row['material_equal_tail_Pa'], dtype=float)
    if quant.shape != (3,) or not np.isfinite(quant).all() or np.any(np.diff(quant) < 0):
        raise ValueError('Ordered material 2.5/50/97.5 percentiles required')
    details = row['material_quantile_details']
    root_lo = 75000*np.exp(np.log(2)*details[0]['unit_bracket']['lower'])
    root_hi = 75000*np.exp(np.log(2)*details[2]['unit_bracket']['upper'])
    result['material'] = dict(median_Pa=float(quant[1]), absolute_log_median_error=float(abs(np.log(quant[1]/E))),
                              relative_median_error=float(abs(quant[1]-E)/E), equal_tail_Pa=quant,
                              covered95=bool(quant[0] <= E <= quant[2]),
                              numerical_quantile_enclosure95_Pa=[float(root_lo),float(root_hi)],
                              covered95_numerical_enclosure=bool(root_lo <= E <= root_hi))
    intervals = np.asarray(row['canonical_position_equal_tail_m'])
    enclosure = np.asarray(row['canonical_position_interval95_numerical_enclosure_m'])
    result['canonical_pointwise95'] = interval_metrics(intervals[:,:,0], intervals[:,:,2], target[POINT_IDS])
    result['canonical_pointwise95_numerical_quantile_enclosure'] = interval_metrics(enclosure[:,:,0], enclosure[:,:,1], target[POINT_IDS])
    variance = np.asarray(row['canonical_coordinate_variance_m2'])
    resolution = np.asarray(row['canonical_variance_integration_resolution_estimate_m2'])
    if variance.shape != (63,3) or resolution.shape != variance.shape or np.any(variance < 0) or not np.isfinite([variance,resolution]).all():
        raise ValueError('Finite nonnegative latent-field variance required')
    result['predicted_mean_coordinate_variance_m2'] = float(variance.mean())
    result['variance_at_or_below_estimated_integration_resolution_count'] = int(np.sum(variance <= resolution))
    result['zero_variance_with_positive_error_count'] = int(np.sum((variance == 0) & (q != target)))
    return result


def aggregate(rows):
    """Exactly64 records, retaining every failed denominator and four groups."""
    if len(rows) != 64 or len({r['object_id'] for r in rows}) != 64:
        raise ValueError('All64 unique records required; no filtering')
    if {(r['private_configuration_index'],r['repeat_index']) for r in rows} != {(c,r) for c in range(4) for r in range(16)}:
        raise ValueError('Four complete fixed configurations times sixteen repetitions')
    from scipy.stats import beta
    groups = []
    for c in range(4):
        group = [r for r in rows if r['private_configuration_index'] == c]
        available = [r for r in group if r['primary_prediction_available']]
        admitted = [r for r in available if r['admitted']]
        k = sum(r['material']['covered95'] for r in admitted)
        # Bounds describe the observed success count including unavailable as
        # failures. The interval is noise-repeat conditional, never population.
        lo = 0. if k == 0 else float(beta.ppf(.025,k,17-k))
        hi = 1. if k == 16 else float(beta.ppf(.975,k+1,16-k))
        groups.append(dict(configuration=c, noise_repeats=16, available=len(available),
                           numerically_admitted=sum(r['admitted'] for r in group),
                           material95_covered_count=k, denominator_including_failures=16,
                           material95_success_fraction_conditional_binomial_interval=[lo,hi],
                           canonical_pointwise_covered_count=sum(r['canonical_pointwise95']['covered_count'] for r in admitted),
                           canonical_pointwise_z_covered_count=sum(r['canonical_pointwise95']['z_covered_count'] for r in admitted),
                           canonical_z_denominator_including_failures=16*9,
                           numerical_quantile_enclosure_covered_count=sum(r['canonical_pointwise95_numerical_quantile_enclosure']['covered_count'] for r in admitted),
                           canonical_coordinate_denominator_including_failures=16*27))
    available = [r for r in rows if r['primary_prediction_available']]
    ncoord = sum(r['canonical63']['coordinate_count'] for r in available)
    mse = sum(r['canonical63']['coordinate_squared_error_sum_m2'] for r in available)
    return dict(record_count=64, physical_configurations=4, noise_repeats_per_configuration=16,
                independent_population_objects=False, primary_method=PRIMARY_METHOD,
                available_predictions=len(available), numerically_admitted=sum(r['admitted'] for r in rows),
                pooled_available_coordinate_RMSE_m=float(np.sqrt(mse/ncoord)) if ncoord else None,
                pooled_RMSE_scope='Descriptive available predictions only; unavailable counts reported, not silently excluded from coverage denominators',
                groups=groups)
