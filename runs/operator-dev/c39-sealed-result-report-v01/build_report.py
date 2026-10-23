"""Compact scored C39 results and an explicitly posthoc public-rule control."""
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments import run_campaign_c38_prediction_evaluation as e

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
DATA = ROOT / 'runs/gaussian-campaign-v01/c39-full-evaluation-v01'
SUMMARY = DATA / 'summary.json'
assert e.io.sha(SUMMARY) == 'a6aa1a8838d27cd39c7c87984ad92d9b821c0315311caae02a12b6bcdb3c447c'
s = json.loads(SUMMARY.read_text())
choices = e.io.read_json(ROOT, s['choices_seal'])
reference = e.io.read_json(ROOT, s['frozen_reference'])
initial = e.io.read_json(ROOT, reference['initial_prediction_seal'])
models = {}
selected_by_model = {}
for name, result in s['models'].items():
    choices_by_id = {r['object_id']: r['choices'] for r in choices['model_choices'][name]}
    matched = 0
    for record in initial['objects']:
        state = e.io.read_json(ROOT, record['result'])
        pred = e.restore_prediction(ROOT, state['models'][name]['files'])
        p = pred['methods']['adaptive']['category_probabilities']
        # Frozen public fine-linear design ranked DOWN for ends and SIDE for
        # bottom patches. This check was added after scores: it is a diagnostic,
        # not another prospectively registered policy or a new outcome trial.
        rule = 'middle_down_2N' if int(np.argmax(p)) == 0 else 'middle_side_2N'
        c = choices_by_id[record['object_id']]
        matched += int(c['information'] == c['target_risk'] == rule)
    old = reference['old_metric_rows'][name]
    before = e.io.read_json(ROOT, old['initial'])
    down = {r['object_id']: r for r in e.io.read_json(ROOT, old['down'])}
    side = {r['object_id']: r for r in json.loads((DATA / (name + '-SIDE-private-metrics.json')).read_text())}
    selected = [down[token] if c['target_risk'] == 'middle_down_2N' else side[token] for token, c in choices_by_id.items()]
    selected_by_model[name] = {r['object_id']: r for r in selected}
    def scalar(rows):
        return dict(records=len(rows), admitted=sum(r['admitted'] for r in rows),
                    mean_case_RMSE_um=float(np.mean([r['canonical63']['coordinate_RMSE_m'] for r in rows])*1e6),
                    material_relative_median_error_mean_percent=float(np.mean([r['material']['relative_median_error'] for r in rows])*100),
                    material95_covered=sum(r['material']['covered95'] for r in rows),
                    support_correct=sum(r['support']['correct'] for r in rows),
                    raw_pointwise95_covered=sum(r['canonical_pointwise95']['covered_count'] for r in rows),
                    raw_z95_covered=sum(r['canonical_pointwise95']['z_covered_count'] for r in rows),
                    numerical_quantile_enclosure_covered=sum(r['canonical_pointwise95_numerical_quantile_enclosure']['covered_count'] for r in rows),
                    coordinate_denominator=1728, z_coordinate_denominator=576)
    models[name] = dict(prefix=scalar(before), selected_one_probe=scalar(selected),
                        pooled_RMSE_um={p: v['aggregate']['pooled_available_coordinate_RMSE_m']*1e6 for p, v in result['policies'].items()},
                        paired_advantage=result['canonical_RMSE_advantage_over_baseline_m'],
                        posthoc_public_MAP_support_Fisher_rule_matches=matched,
                        uncertainty_policy_advantage_over_this_rule=0. if matched == 64 else None)
receipt = dict(schema='C39-sealed-paired-development-result-v01', summary=e.io.record(ROOT, SUMMARY),
               models=models, fixed_physical_configurations=4, noise_repeats_per_configuration=16,
               independent_population_objects=False, multiple_comparison_adjustment_applied=False,
               old_10nm_coordinate_quantile_limitation_retained=True,
               raw_95_coverage_is_not_proven_resolved_for_tiny_coordinates=True,
               posthoc_rule_uses_public_prefix_predictions_and_preexisting_public_linear_rankings=True,
               posthoc_rule_control_not_prospectively_registered=True,
               no_added_observations_or_prediction_fits_for_posthoc_rule=True,
               active_policy_novelty_or_ambiguous_boundary_resolution_demonstrated=False,
               calibrated_full3D_uncertainty_demonstrated=False, is_temporal_forecast=False,
               raw_fields_or_private_metric_rows_uploaded=False, campaign_complete=False)
from experiments.campaign_c38_policy_metrics import paired_noise_summary
nl = selected_by_model['nonlinear_table']
linear = selected_by_model['linear_rest_tangent']
tokens = sorted(nl)
receipt['selected_nonlinear_mean_RMSE_advantage_over_linear_m'] = paired_noise_summary(
    [linear[t]['canonical63']['coordinate_RMSE_m']-nl[t]['canonical63']['coordinate_RMSE_m'] for t in tokens],
    [(nl[t]['private_configuration_index'], nl[t]['repeat_index']) for t in tokens])
(OUT / 'compact-result.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({name: {k: v for k, v in result.items() if k != 'paired_advantage'} for name, result in models.items()}))
