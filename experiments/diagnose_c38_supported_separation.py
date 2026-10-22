"""Read-only finite-corner response separation; no posterior or observations.

Unqualified uniform fields remain unqualified. Noise is an illustrative scale,
not calibrated sensor noise. No interpolation over unknown material is used.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / 'runs/operator-dev/c38-supported-separation-v02'
    base.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'runs/operator-dev/c38-static-prequalification-v01/completed-summary.json'
    matrix = json.loads(source.read_text())
    if matrix['completed'] is not True or matrix['cases_completed'] != 14:
        raise ValueError('Completed fixed development matrix required')
    fields, inputs, indicators = {}, [], {}
    points = None
    for entry in matrix['rows']:
        case = entry['case']
        if case['kind'] != 'domain_corner':
            continue
        result_path = ROOT / entry['result']['path']
        if digest(result_path) != entry['result']['sha256']:
            raise ValueError('Changed matrix-bound result')
        result = json.loads(result_path.read_text())
        capture = result['meshes'][-1]['capture']
        path = ROOT / capture['path']
        if digest(path) != capture['sha256']:
            raise ValueError('Changed matrix-bound capture')
        with np.load(path, allow_pickle=False) as data:
            x = data['material_probes_m'].copy()
            q = data['probe_positions_m'].copy()
        if x.shape != (63, 3) or not np.isfinite(q).all():
            raise ValueError('Finite original physical queries required')
        if points is None:
            points = x
        if not np.array_equal(points, x):
            raise ValueError('Different physical query set')
        key = (case['support'], case['young_Pa'], case['Fz_N'])
        fields[key] = q
        indicators[key] = result['conditional_spatial_indicators_m'][-1]
        inputs.append(dict(case=case, result=entry['result'], capture=capture,
                           spatial_indicator_passed=result['passed']))
    # Nine top-face points, fixed geometrically; exclude fixed end faces and
    # include the known middle force patch. These are ideal material
    # positions, not rendered/occluded camera observations.
    selected = np.flatnonzero(np.isclose(points[:, 2], .04) &
                              np.isin(np.round(points[:, 0], 8), [.04, .12, .20]))
    if len(selected) != 9:
        raise ValueError('Declared nine-point geometry changed')
    sigma = 200e-6

    def compare(a, b):
        delta = (fields[a] - fields[b])[selected]
        return dict(coordinate_RMS_m=float(np.sqrt(np.mean(delta**2))),
                    maximum_point_distance_m=float(np.linalg.norm(delta, axis=1).max()),
                    illustrative_whitened_distance=float(np.linalg.norm(delta) / sigma),
                    sum_conditional_mesh_indicators_m=indicators[a] + indicators[b])

    supports = ['both_ends', 'separated_bottom_patches']
    materials = [75000., 150000.]
    rows = []
    for force in [-2., 0., 2.]:
        for support in supports:
            a, b = (support, materials[0], force), (support, materials[1], force)
            rows.append(dict(kind='material_endpoint_separation', force_N=force,
                             support=support, **compare(a, b)))
        cross = []
        for e1 in materials:
            for e2 in materials:
                a, b = (supports[0], e1, force), (supports[1], e2, force)
                cross.append(dict(young_both_ends_Pa=e1, young_bottom_Pa=e2,
                                  **compare(a, b)))
        rows.append(dict(kind='cross_support_four_endpoint_pairs', force_N=force,
                         pairs=cross,
                         minimum_sampled_whitened_distance=min(
                             v['illustrative_whitened_distance'] for v in cross)))
    out = dict(schema='C38-fixed-development-separation-v02', completed=True,
               matrix=dict(path=str(source.relative_to(ROOT)), sha256=digest(source)),
               source_sha256=digest(Path(__file__)), inputs=inputs,
               ideal_top_material_points_m=points[selected].tolist(),
               illustrative_independent_coordinate_noise_scale_m=sigma, rows=rows,
               observations_drawn=False, posterior_computed=False, dataset_admitted=False,
               limits=['Finite selected development corners only; no continuous minimization over E',
                       'A minimum over four endpoint pairs is NOT a lower bound over the material interval',
                       'Uniform matrix contains failed spatial indicators; diagnostic only',
                       'Summed engineering indicators are not rigorous response error bounds',
                       'Ideal point coordinates do not establish camera visibility or RGB-D performance',
                       'Illustrative whitened distances do not establish calibrated inference or useful probing'])
    (base / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(completed=True, comparisons=len(rows), observations_drawn=False)))


if __name__ == '__main__':
    main()
