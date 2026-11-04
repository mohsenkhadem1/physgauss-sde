"""Post-hoc ranking diagnostic on the prospectively selected refinement subset.

Requires all twelve reference comparisons terminal. Never refits predictions.
Unavailable refined references remain explicit and exclude paired worlds.
Run from repository root with PYTHONPATH=. and the repository environment.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.complex_scenes_v18.held_appearance import closed_census
from scripts.complex_scenes_v18.world_aggregation import paired


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positions(path):
    with np.load(path) as data:
        x, t = data['surface_positions_m'].copy(), data['times_s'].copy()
    assert np.isfinite(x).all() and np.isfinite(t).all()
    if np.iscomplexobj(x):
        assert not np.any(x.imag)
        x = x.real
    return x, t


def main():
    campaign = Path('runs/complex-scenes-v18/held-campaign-01')
    sensitivity_path = campaign / 'reference-sensitivity.json'
    sensitivity = json.loads(sensitivity_path.read_text())
    assert len(sensitivity['rows']) == 12
    assert all(r['state'] in ('complete', 'reference_failed') for r in sensitivity['rows'])
    sensitivity_rows = {(r['world_id'], r['query_index']): r for r in sensitivity['rows']}
    physical_path = Path('reports/complex-scenes-v18/held-physical-results.json')
    physical = json.loads(physical_path.read_text())
    key = lambda r: (r['world_id'], r['query_index'], r['method'])
    primary = {key(r): r for r in physical['rows']}
    selected = {(r['world_id'], r['query_index']) for r in sensitivity['rows']}
    rows = []
    cache = {}
    for original in closed_census(campaign):
        identity = (original['world_id'], original['query_index'])
        if identity not in selected:
            continue
        row = {k: original[k] for k in ('world_id', 'query_index', 'family', 'method', 'state')}
        rows.append(row)
        if original['state'] != 'sealed':
            continue
        if sensitivity_rows[identity]['state'] != 'complete':
            row.update(state='reference_unavailable', prediction_state=original['state'],
                       reference_errors=sensitivity_rows[identity].get('errors'))
            continue
        if identity not in cache:
            seals = list((campaign / 'references' / identity[0] / f'{identity[1]}-refined').glob('*/complete.json'))
            assert len(seals) == 1
            seal = json.loads(seals[0].read_text())
            ref = seals[0].parent / 'trajectory.npz'
            assert seal['completed'] and seal['n'] == 6
            assert seal['public_query'] == original['public_query']
            assert seal['private_world_sha256'] == original['world_material_commitment']
            assert digest(ref) == seal['trajectory_sha256']
            cache[identity] = (*positions(ref), digest(ref))
        actual, times, reference_hash = cache[identity]
        prediction_path = Path(original['physical_output']) / 'trajectory.npz'
        predicted, clock = positions(prediction_path)
        assert actual.shape == predicted.shape and np.allclose(times, clock, atol=1e-9, rtol=0)
        error = predicted-actual
        rms = float(1000*np.linalg.norm(error.ravel())/np.sqrt(np.prod(error.shape[:-1])))
        row.update(state='complete', metrics={'vector_rms_mm': rms},
                   primary_vector_rms_mm=primary[key(original)]['metrics']['vector_rms_mm'],
                   refined_reference_sha256=reference_hash, prediction_sha256=digest(prediction_path))
    aggregates = []
    for baseline in ('analytic-control', 'spring-gaus', 'phystwin', 'uniphy'):
        refined = paired(rows, 'ours', baseline)
        primary_rows = [dict(r, metrics={'vector_rms_mm': r['primary_vector_rms_mm']}) if r['state'] == 'complete' else r for r in rows]
        base = paired(primary_rows, 'ours', baseline)
        aggregates.append(dict(baseline=baseline, primary=base, refined=refined))
    result = dict(scope='Post-hoc ranking sensitivity of sealed predictions on the prospectively selected one-world-per-family refinement subset; not new training, population convergence, or a new held test',
                  sensitivity_report_sha256=digest(sensitivity_path), physical_report_sha256=digest(physical_path),
                  reference_comparisons=sensitivity['rows'], rows=rows, paired_world_aggregates=aggregates)
    path = Path('reports/complex-scenes-v18/held-refined-prediction-audit.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
