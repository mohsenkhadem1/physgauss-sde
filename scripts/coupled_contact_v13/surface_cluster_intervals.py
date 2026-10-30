"""Descriptive object-cluster bootstrap for fixed primary paired comparisons.

Eight objects, not their repeated interactions, are resampling units. Missing
method predictions stay missing; estimates are conditional on paired completion.
This is a post-fit summary and cannot alter any case, prediction or selection.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/coupled-surface-v13'
SEED = 130013
DRAWS = 10000


def interval(gains_by_object, draws=DRAWS, seed=SEED):
    sums = np.array([sum(values) for values in gains_by_object], dtype=float)
    counts = np.array([len(values) for values in gains_by_object], dtype=int)
    if not len(counts) or not counts.sum():
        return dict(pairs=0, mean_gain_mm=None, percentile_95_mm=None,
                    valid_bootstrap_draws=0, draws=draws, seed=seed)
    if not np.isfinite(sums).all():
        raise ValueError('Nonfinite paired gain')
    indices = np.random.default_rng(seed).integers(0, len(counts), size=(draws, len(counts)))
    totals = counts[indices].sum(axis=1)
    valid = totals > 0
    samples = sums[indices].sum(axis=1)[valid]/totals[valid]
    return dict(pairs=int(counts.sum()), mean_gain_mm=float(sums.sum()/counts.sum()),
                percentile_95_mm=np.percentile(samples, [2.5, 97.5]).tolist(),
                valid_bootstrap_draws=int(valid.sum()), draws=draws, seed=seed)


def records(rows):
    out = {(row['object'], row['scenario']): row for row in rows}
    if len(out) != len(rows):
        raise ValueError('Duplicate forecast records')
    return out


def main():
    names = ['protocol.json', 'reserved-results.json', 'ablation-results.json']
    protocol, main_results, supplement = [json.loads((REPORT/name).read_text()) for name in names]
    objects = [obj['id'] for obj in protocol['objects'] if obj['split'] == 'reserved']
    if len(objects) != 8 or len(set(objects)) != 8:
        raise ValueError('This declared study requires eight distinct reserved objects')
    scenes = protocol['primary_forecast_scenarios']
    expected = {(obj, scene) for obj in objects for scene in scenes}
    primary = records([row for row in main_results['cases'] if row['scenario'] in scenes])
    secondary = records([row for row in supplement['cases'] if row['split'] == 'reserved' and row['scenario'] in scenes])
    if set(primary) != expected or set(secondary) != expected:
        raise ValueError('Every fixed primary record, including failed predictions, is required')
    comparisons = {}
    for method in ['rigid', 'native', 'position_ridge', 'nominal', 'vertex']:
        clusters = []
        coverage = []
        for obj in objects:
            gains = []
            for scene in scenes:
                key = (obj, scene)
                ours = primary[key]['results'].get('fem', {})
                other = (secondary if method in ['nominal', 'vertex'] else primary)[key]['results'].get(method, {})
                if ours.get('complete') and other.get('complete'):
                    values = [other['coordinate_rmse_m'], ours['coordinate_rmse_m']]
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite completed score')
                    gains.append((values[0]-values[1])*1000)
            clusters.append(gains)
            coverage.append(dict(object=obj, pairs=len(gains), expected=len(scenes)))
        comparisons[method] = dict(interval(clusters), expected_pairs=len(expected),
                                   contributing_objects=sum(bool(values) for values in clusters),
                                   object_coverage=coverage)
    output = dict(scope='Descriptive 95% percentile intervals from resampling all eight objects with replacement. '
        'Repeated interactions stay together. Conditional on paired completion; failed predictions remain '
        'in coverage. No imputation, per-case quality filtering, multiple-comparison correction or population guarantee.',
        primary_scenarios=scenes, objects=objects, comparisons=comparisons,
        source_sha256={name: hashlib.sha256((REPORT/name).read_bytes()).hexdigest() for name in names},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (REPORT/'primary-cluster-intervals.json').write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
