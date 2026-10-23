"""Pure paired evaluation of already sealed static one-probe policies.

Resampling is within the four FIXED physical configurations, over independent
noise repetitions. The intervals do not describe unseen physical objects.
Nothing here selects an action, draws an observation, or updates a predictor.
"""
from __future__ import annotations
import numpy as np
from experiments.campaign_c38_prediction_metrics import aggregate

POLICIES=('information','target_risk','fixed_down','fixed_up','random')
BOOTSTRAP_REPLICATES=5000
BOOTSTRAP_SEED=2026102304


def paired_noise_summary(values, identities):
    """Positive supplied differences indicate improvement; no missing pairs."""
    d=np.asarray(values,dtype=float)
    if d.shape!=(64,) or not np.isfinite(d).all() or len(identities)!=64 or set(identities)!={(c,r) for c in range(4) for r in range(16)}:
        raise ValueError('All64 finite paired values with four complete repeat groups')
    a=np.empty((4,16))
    for value,(c,r) in zip(d,identities):a[c,r]=value
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    indices=rng.integers(0,16,size=(4,BOOTSTRAP_REPLICATES,16))
    draws=np.take_along_axis(a[:,None,:],indices,axis=2).mean(axis=(0,2))
    return dict(mean_improvement=float(d.mean()),group_mean_improvements=a.mean(axis=1).tolist(),
                conditional_noise_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),
                bootstrap_replicates=BOOTSTRAP_REPLICATES,bootstrap_seed=BOOTSTRAP_SEED,
                paired_count=64,scope='Paired percentile bootstrap over independent noise repetitions within each of four fixed configurations; not population uncertainty')


def index_rows(rows):
    aggregate(rows)  # Exact64 identities/configuration/repeat/failure denominator.
    return {r['object_id']:r for r in rows}


def evaluate_policies(initial_rows, post_rows_by_force, choice_rows):
    """Reuse the two counterfactual posteriors according to frozen choices.

    Each evaluated policy gets one27-coordinate probe, in addition to the common
    prefix27 coordinates. The two actual action outcomes are never combined.
    """
    before=index_rows(initial_rows)
    if set(post_rows_by_force)!={-2.,2.}:raise ValueError('Both fixed-action baselines required')
    after={f:index_rows(rows) for f,rows in post_rows_by_force.items()}
    if any(set(v)!=set(before) for v in after.values()):raise ValueError('Exact matched post/prefix identities')
    choices={r['object_id']:r['choices'] for r in choice_rows}
    if len(choice_rows)!=64 or set(choices)!=set(before):raise ValueError('All64 frozen policy choices')
    for token,c in choices.items():
        if set(c)!=set(POLICIES) or c['fixed_down']!=-2. or c['fixed_up']!=2. or any(v not in (-2.,2.) for v in c.values()):raise ValueError('Five known one-probe policies')
        for force in (-2.,2.):
            if any(before[token][k]!=after[force][token][k] for k in ('private_configuration_index','repeat_index')):
                raise ValueError('Same physical/noise repetition required')
    tokens=sorted(before);identities=[(before[t]['private_configuration_index'],before[t]['repeat_index']) for t in tokens]
    selected={p:[after[choices[t][p]][t] for t in tokens] for p in POLICIES}
    results={}
    for name,rows in selected.items():
        admitted=all(before[t]['admitted'] and row['admitted'] for t,row in zip(tokens,rows))
        result=dict(aggregate=aggregate(rows),one_probe_coordinates=27,total_acquired_coordinates=54,
                    all64_pairs_numerically_admitted=admitted,
                    down_choices=sum(choices[t][name]==-2. for t in tokens),up_choices=sum(choices[t][name]==2. for t in tokens))
        if admitted:
            result['canonical_RMSE_reduction_m']=paired_noise_summary([before[t]['canonical63']['coordinate_RMSE_m']-r['canonical63']['coordinate_RMSE_m'] for t,r in zip(tokens,rows)],identities)
            result['material_absolute_log_error_reduction']=paired_noise_summary([before[t]['material']['absolute_log_median_error']-r['material']['absolute_log_median_error'] for t,r in zip(tokens,rows)],identities)
            result['support_Brier_reduction']=paired_noise_summary([before[t]['support']['categorical_Brier']-r['support']['categorical_Brier'] for t,r in zip(tokens,rows)],identities)
            result['predicted_variance_reduction_m2']=paired_noise_summary([before[t]['predicted_mean_coordinate_variance_m2']-r['predicted_mean_coordinate_variance_m2'] for t,r in zip(tokens,rows)],identities)
        else:result['paired_interval_unavailable_reason']='At least one unavailable/numerically unqualified pair; no selected-case confidence interval'
        results[name]=result
    comparisons={}
    for policy in ('information','target_risk','random'):
        comparisons[policy]={}
        for fixed in ('fixed_down','fixed_up'):
            if not all(r['admitted'] for r in selected[policy]+selected[fixed]):
                comparisons[policy][fixed]=dict(available=False,reason='Numerical/missing prediction failures retained')
                continue
            comparisons[policy][fixed]=paired_noise_summary([b['canonical63']['coordinate_RMSE_m']-a['canonical63']['coordinate_RMSE_m'] for a,b in zip(selected[policy],selected[fixed])],identities)
            comparisons[policy][fixed]['positive_favors']=policy
    return dict(completed=True,policies=results,canonical_RMSE_advantage_over_fixed_m=comparisons,
                four_fixed_physical_configurations=True,independent_population_objects=False,
                all_policy_choices_supplied_not_optimized_here=True,
                common_once_generated_action_outcomes=True,one_acquired_action_per_policy=True)
