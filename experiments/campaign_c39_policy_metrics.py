"""Score sealed typed DOWN/SIDE choices, never select actions or fit a model."""
from experiments.campaign_c38_policy_metrics import index_rows, paired_noise_summary
from experiments.campaign_c38_prediction_metrics import aggregate

DOWN = 'middle_down_2N'
SIDE = 'middle_side_2N'
POLICIES = ('information', 'target_risk', 'fixed_down', 'fixed_side', 'random')


def evaluate_policies(initial_rows, post_rows_by_action, choice_rows):
    """One acquired27-coordinate action plus original27-coordinate prefix.

    Reuse the exact C38 initial and downward results; new SIDE results are
    scored only after their full seal. All64 records and failures are retained.
    The conditional bootstrap describes noise in four fixed configurations.
    """
    before = index_rows(initial_rows)
    if set(post_rows_by_action) != {DOWN, SIDE}:
        raise ValueError('Both typed fixed-action baselines required')
    after = {a: index_rows(rows) for a, rows in post_rows_by_action.items()}
    if any(set(rows) != set(before) for rows in after.values()):
        raise ValueError('Matched original prefix and both action identities')
    choices = {r['object_id']: r['choices'] for r in choice_rows}
    if len(choice_rows) != 64 or set(choices) != set(before):
        raise ValueError('All64 sealed choices required')
    for token, c in choices.items():
        if (set(c) != set(POLICIES) or c['fixed_down'] != DOWN or
                c['fixed_side'] != SIDE or any(v not in (DOWN, SIDE) for v in c.values())):
            raise ValueError('Exact five typed one-probe policies; no scalar Fz alias')
        for action in (DOWN, SIDE):
            if any(before[token][k] != after[action][token][k]
                   for k in ('private_configuration_index', 'repeat_index')):
                raise ValueError('Same fixed physical/noise repetition required')
    tokens = sorted(before)
    identities = [(before[t]['private_configuration_index'], before[t]['repeat_index']) for t in tokens]
    selected = {p: [after[choices[t][p]][t] for t in tokens] for p in POLICIES}
    definitions = {
        'canonical_RMSE_reduction_m': lambda r: r['canonical63']['coordinate_RMSE_m'],
        'material_absolute_log_error_reduction': lambda r: r['material']['absolute_log_median_error'],
        'support_Brier_reduction': lambda r: r['support']['categorical_Brier'],
        'predicted_variance_reduction_m2': lambda r: r['predicted_mean_coordinate_variance_m2'],
    }
    results = {}
    for name, rows in selected.items():
        admitted = all(before[t]['admitted'] and r['admitted'] for t, r in zip(tokens, rows))
        result = dict(aggregate=aggregate(rows), one_probe_coordinates=27,
                      total_acquired_coordinates=54, all64_pairs_numerically_admitted=admitted,
                      action_counts={a: sum(choices[t][name] == a for t in tokens) for a in (DOWN, SIDE)})
        if admitted:
            for key, value in definitions.items():
                result[key] = paired_noise_summary([value(before[t]) - value(r) for t, r in zip(tokens, rows)], identities)
        else:
            result['paired_interval_unavailable_reason'] = 'Missing/numerically unqualified pair; no selected-case confidence interval'
        results[name] = result
    comparisons = {}
    for policy in ('information', 'target_risk', 'random'):
        comparisons[policy] = {}
        for fixed in ('fixed_down', 'fixed_side', 'random'):
            if fixed == policy:
                continue
            if not all(r['admitted'] for r in selected[policy] + selected[fixed]):
                comparisons[policy][fixed] = dict(available=False, reason='Numerical/missing failures retained')
                continue
            delta = [b['canonical63']['coordinate_RMSE_m'] - a['canonical63']['coordinate_RMSE_m']
                     for a, b in zip(selected[policy], selected[fixed])]
            comparisons[policy][fixed] = paired_noise_summary(delta, identities)
            comparisons[policy][fixed]['positive_favors'] = policy
    return dict(completed=True, policies=results, canonical_RMSE_advantage_over_baseline_m=comparisons,
                four_fixed_physical_configurations=True, independent_population_objects=False,
                paired_development_extension_of_C38=True, all_policy_choices_supplied_not_optimized_here=True,
                one_acquired_action_per_policy=True, equal_force_magnitude_N=2.,
                equal_mechanical_work_or_energy_cost_claim=False,
                reused_original_initial_and_DOWN_results=True)
