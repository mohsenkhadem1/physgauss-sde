import copy
import pytest
from experiments.campaign_c39_policy_metrics import DOWN, SIDE, evaluate_policies


def fixture():
    def rows(action):
        result = []
        for c in range(4):
            for n in range(16):
                correct_action = DOWN if c < 2 else SIDE
                error = 3. if action == 'before' else (1. if action == correct_action else 2.)
                result.append(dict(object_id=f'{c*16+n:032x}', private_configuration_index=c, repeat_index=n,
                                   primary_prediction_available=True, admitted=True,
                                   canonical63=dict(coordinate_RMSE_m=error, coordinate_count=189,
                                                    coordinate_squared_error_sum_m2=189*error**2),
                                   material=dict(covered95=True, absolute_log_median_error=error),
                                   support=dict(categorical_Brier=.1), predicted_mean_coordinate_variance_m2=error,
                                   canonical_pointwise95=dict(covered_count=27, z_covered_count=9),
                                   canonical_pointwise95_numerical_quantile_enclosure=dict(covered_count=27)))
        return result
    before = rows('before')
    choices = [dict(object_id=r['object_id'], choices=dict(
        information=DOWN if r['private_configuration_index'] < 2 else SIDE,
        target_risk=DOWN if r['private_configuration_index'] < 2 else SIDE,
        fixed_down=DOWN, fixed_side=SIDE, random=DOWN)) for r in before]
    return before, {DOWN: rows(DOWN), SIDE: rows(SIDE)}, choices


def test_known_direction_reversal_beats_both_fixed_without_pooling_two_probes():
    before, after, choices = fixture()
    x = evaluate_policies(before, after, choices)
    assert x['policies']['information']['canonical_RMSE_reduction_m']['mean_improvement'] == 2.
    assert x['policies']['information']['action_counts'] == {DOWN: 32, SIDE: 32}
    for baseline in ('fixed_down', 'fixed_side', 'random'):
        assert x['canonical_RMSE_advantage_over_baseline_m']['information'][baseline]['conditional_noise_bootstrap95'] == [.5, .5]
    assert all(v['total_acquired_coordinates'] == 54 for v in x['policies'].values())
    # Matching the fixed policy must give exactly zero paired advantage.
    for r in choices:
        r['choices']['information'] = DOWN
    y = evaluate_policies(before, after, choices)
    assert y['canonical_RMSE_advantage_over_baseline_m']['information']['fixed_down']['conditional_noise_bootstrap95'] == [0., 0.]


def test_reject_wrong_action_identity_and_missing_or_mismatched_pairs():
    before, after, choices = fixture()
    bad = copy.deepcopy(choices)
    bad[0]['choices']['information'] = 2.
    with pytest.raises(ValueError):
        evaluate_policies(before, after, bad)
    with pytest.raises(ValueError):
        evaluate_policies(before, {DOWN: after[DOWN], SIDE: after[SIDE][:-1]}, choices)
    after[SIDE][0]['repeat_index'] = 1
    with pytest.raises(ValueError):
        evaluate_policies(before, after, choices)


def test_numerical_failure_retains_denominator_and_suppresses_bootstrap():
    before, after, choices = fixture()
    after[SIDE][63]['admitted'] = False
    x = evaluate_policies(before, after, choices)
    v = x['policies']['information']
    assert v['aggregate']['record_count'] == 64
    assert v['aggregate']['numerically_admitted'] == 63
    assert 'canonical_RMSE_reduction_m' not in v
    assert x['canonical_RMSE_advantage_over_baseline_m']['information']['fixed_down']['available'] is False
