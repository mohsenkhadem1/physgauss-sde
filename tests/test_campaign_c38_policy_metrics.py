import numpy as np
import pytest
from experiments.campaign_c38_policy_metrics import paired_noise_summary,evaluate_policies


def identities():return [(c,r) for c in range(4) for r in range(16)]


def test_paired_identical_policy_has_exact_zero_uncertainty():
    r=paired_noise_summary(np.zeros(64),identities())
    assert r['mean_improvement']==0 and r['conditional_noise_bootstrap95']==[0.,0.]


def test_stratification_keeps_fixed_configuration_composition():
    values=np.repeat([-3.,-1.,1.,3.],16)
    r=paired_noise_summary(values,identities())
    assert r['conditional_noise_bootstrap95']==[0.,0.]
    assert r['group_mean_improvements']==[-3.,-1.,1.,3.]
    # A bootstrap that resampled configurations would incorrectly vary here.


def test_paired_known_shift_order_invariance_and_failures():
    y=np.linspace(-1.,1.,64)+2
    a=paired_noise_summary(y,identities());b=paired_noise_summary(y[::-1],identities()[::-1])
    assert a['conditional_noise_bootstrap95']==b['conditional_noise_bootstrap95']
    assert a['mean_improvement']==pytest.approx(2)
    with pytest.raises(ValueError):paired_noise_summary(y[:-1],identities()[:-1])
    with pytest.raises(ValueError):paired_noise_summary(np.full(64,np.nan),identities())


def test_counterfactual_reuse_equal_budget_and_real_accuracy_vs_confidence():
    def rows(error,variance,material_error):
        return [dict(object_id=f'{c*16+r:032x}',private_configuration_index=c,repeat_index=r,
                     primary_prediction_available=True,admitted=True,
                     canonical63=dict(coordinate_RMSE_m=error,coordinate_count=189,coordinate_squared_error_sum_m2=189*error**2),
                     material=dict(covered95=True,absolute_log_median_error=material_error),
                     support=dict(categorical_Brier=.1),predicted_mean_coordinate_variance_m2=variance,
                     canonical_pointwise95=dict(covered_count=27,z_covered_count=9),
                     canonical_pointwise95_numerical_quantile_enclosure=dict(covered_count=27))
                for c,r in identities()]
    before=rows(.002,1e-6,.1);down=rows(.001,1e-7,.2);up=rows(.003,1e-8,.3)
    choices=[dict(object_id=r['object_id'],choices=dict(information=-2.,target_risk=-2.,fixed_down=-2.,fixed_up=2.,random=2.)) for r in before]
    x=evaluate_policies(before,{-2.:down,2.:up},choices)
    assert x['policies']['information']['canonical_RMSE_reduction_m']['mean_improvement']==pytest.approx(.001)
    assert x['policies']['information']['material_absolute_log_error_reduction']['mean_improvement']==pytest.approx(-.1)
    assert x['policies']['fixed_up']['predicted_variance_reduction_m2']['mean_improvement']>0
    assert x['policies']['fixed_up']['canonical_RMSE_reduction_m']['mean_improvement']<0
    assert x['canonical_RMSE_advantage_over_fixed_m']['information']['fixed_down']['conditional_noise_bootstrap95']==[0.,0.]
    assert all(r['one_probe_coordinates']==27 for r in x['policies'].values())
