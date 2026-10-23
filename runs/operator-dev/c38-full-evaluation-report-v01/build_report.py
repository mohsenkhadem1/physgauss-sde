"""Aggregate sealed evaluation scalars; never publish per-record coordinates."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
DATA = ROOT / 'runs/gaussian-campaign-v01/c38-full-evaluation-v01'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
summary = json.loads((DATA / 'summary.json').read_text())
assert sha(DATA / 'summary.json') == 'c37ce28a038293b7eda67e1e1f9bf542a33edcfa3bf1962f42ca562dbe14e82f'
inputs = {'summary': {'path': str((DATA / 'summary.json').relative_to(ROOT)), 'sha256': sha(DATA / 'summary.json')}}

def aggregate_rows(path):
    rows = json.loads(path.read_text())
    assert len(rows) == 64 and all(r['admitted'] for r in rows)
    inputs[str(path.relative_to(ROOT))] = sha(path)
    raw = 'canonical_pointwise95'
    enc = 'canonical_pointwise95_numerical_quantile_enclosure'
    misses = [float(m) for r in rows for m, covered in zip(np.asarray(r[raw]['miss_m']).flat, np.asarray(r[enc]['covered']).flat) if m > 0 and covered]
    return {
        'records': 64,
        'numerically_admitted': 64,
        'mean_case_RMSE_um': float(np.mean([r['canonical63']['coordinate_RMSE_m'] for r in rows]) * 1e6),
        'mean_material_relative_median_error_percent': float(np.mean([r['material']['relative_median_error'] for r in rows]) * 100),
        'material95_covered': sum(r['material']['covered95'] for r in rows),
        'support_correct': sum(r['support']['correct'] for r in rows),
        'raw_pointwise95_covered': sum(r[raw]['covered_count'] for r in rows),
        'raw_z95_covered': sum(r[raw]['z_covered_count'] for r in rows),
        'coordinate_denominator': 1728,
        'z_coordinate_denominator': 576,
        'numerical_enclosure_covered': sum(r[enc]['covered_count'] for r in rows),
        'raw_covered_by_axis': np.sum([r[raw]['covered_count_by_axis'] for r in rows], axis=0).tolist(),
        'numerical_enclosure_covered_by_axis': np.sum([r[enc]['covered_count_by_axis'] for r in rows], axis=0).tolist(),
        'raw_misses_inside_numerical_enclosure_count': len(misses),
        'maximum_such_miss_nm': max(misses, default=0) * 1e9,
        'maximum_raw_interval_miss_um': max(r[raw]['maximum_miss_m'] for r in rows) * 1e6,
        'mean_raw_interval_width_um': float(np.mean([r[raw]['mean_width_m'] for r in rows]) * 1e6),
    }

models = {}
for model in ('nonlinear_table', 'linear_rest_tangent'):
    models[model] = {'before': aggregate_rows(DATA / 'initial' / f'{model}-private-metrics.json')}
    for force in ('down', 'up'):
        models[model][force] = aggregate_rows(DATA / f'{model}-{force}-private-metrics.json')
    models[model]['pooled_coordinate_RMSE_um'] = {
        'before': summary['initial'][model]['pooled_available_coordinate_RMSE_m'] * 1e6,
        **{p: v['aggregate']['pooled_available_coordinate_RMSE_m'] * 1e6 for p, v in summary['postprobe'][model]['policies'].items()},
    }
    paired = summary['postprobe'][model]['policies']['fixed_down']['canonical_RMSE_reduction_m']
    models[model]['paired_mean_down_improvement_um'] = paired['mean_improvement'] * 1e6
    models[model]['conditional_noise_bootstrap95_um'] = [v * 1e6 for v in paired['conditional_noise_bootstrap95']]

compact = dict(schema='C38-sealed-controlled-inverse-result-v01', inputs=inputs, models=models,
               independent_physical_configurations=4, noise_repeats_per_configuration=16,
               independent_population_objects=False, observed_coordinates_per_acquisition=27,
               initial_and_choice_and_postprediction_sealed_before_private_evaluation=True,
               all_information_and_risk_actions_equal_fixed_down=True,
               active_advantage_over_fixed_down=False,
               RGBD_or_temporal_forecasting=False, calibrated_full3D_uncertainty_demonstrated=False,
               numerical_enclosures_are_not_new_credible_intervals=True,
               raw_fields_uploaded=False, campaign_complete=False)
(OUT / 'compact-result.json').write_text(json.dumps(compact, indent=2) + '\n')
files = [Path(__file__), OUT / 'compact-result.json', DATA / 'summary.json',
         ROOT / 'runs/operator-dev/c38-full-evaluation-v01/protocol.json',
         ROOT / 'docs/GAUSSIAN_C38_SEALED_INVERSE_RESULT.md']
inventory = dict(schema='C38-completed-evaluation-publication-v01', files=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in files], raw_fields_or_private_rows_included=False)
(OUT / 'publication-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
print(json.dumps({'models': {m: r['pooled_coordinate_RMSE_um'] for m, r in models.items()}, 'inventory_sha256': sha(OUT / 'publication-inventory.json')}))
