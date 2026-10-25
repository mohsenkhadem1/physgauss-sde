import copy
import numpy as np
import pytest
from experiments import campaign_noisy_poke_benchmark_v01 as h

def test_fixed_interior8_32_trials_and_toy_only_noise_reproducibility(monkeypatch):
    assert len(h.objects())==8 and{v['E_Pa']for v in h.objects()}=={2500.,6500.}and len(h.trials())==32
    monkeypatch.setattr(h,'PREFIX_SEED',712);monkeypatch.setattr(h,'POKE_SEED',713);a=h.noise_streams();b=h.noise_streams();assert a['prefix'].shape==(4,8,4)and a['poke'].shape==(4,8,8)and np.array_equal(a['prefix'],b['prefix'])and np.array_equal(a['poke'],b['poke'])

def test_fixedbaseline_uses_all36_predicted_risks_only():
    data=[dict(index=i,ranking=dict(qualified=True,actions=[dict(action=j,risk=float((j-3)**2+1))for j in range(8)]))for i in range(36)]
    assert h.fixed_action(data)['action']==3
    with pytest.raises(ValueError):h.fixed_action(data[:-1])
    bad=copy.deepcopy(data);bad[0]['ranking']['actions'].reverse()
    with pytest.raises(ValueError):h.fixed_action(bad)
    data[0]['ranking']['qualified']=False
    with pytest.raises(ValueError):h.fixed_action(data)

def test_continuous_mixture_quantiles_not_averaged_component_quantiles():
    class Ref:
        def __init__(self,power):self.power=power
        def _state(self,*a):return dict(log_evidence=0.)
        def cdf(self,d,q,**k):return q**self.power
    out=h.material_intervals([Ref(1),Ref(2),Ref(3),Ref(4)],[.25]*4)
    for d in range(2):
        for q,alpha in zip(out['unit_quantiles'][d],[.05,.95]):assert abs(sum(q**k for k in range(1,5))/4-alpha)<1e-7
    assert out['physical_quantiles'].shape==(2,2)

def test_random_meanloss_not_loss_of_averaged_means_and_RMS_identities():
    truth=np.zeros(48);before=h.prediction_metrics(np.ones(48),np.ones(48)*2,truth);after=[h.prediction_metrics(np.ones(48)*(1 if i%2 else-1),np.ones(48)*(i+1),truth)for i in range(8)];result=h.compare_policies(before,after,0,0)
    assert result['uniform_random']['MSE_m2']==1. and result['uniform_random']['RMSE_m']==1.
    assert result['uniform_random']['posterior_RMS_SD_m']==np.sqrt(np.mean(np.arange(1,9)**2))
    assert result['adaptive']==result['fixed']and result['uniform_random']['moment_bands_have_no_nominal_coverage_level']

def test_narrower_but_worse_is_descriptive_and_material_intervals_actual():
    truth=np.zeros(48);before=h.prediction_metrics(np.ones(48),np.ones(48)*2,truth);after=[h.prediction_metrics(np.ones(48)*2,np.ones(48),truth)]*8
    assert h.compare_policies(before,after,0,1)['adaptive']['variance_down_error_up']
    result=h.material_metrics(dict(physical_quantiles=np.array([[2300,2700],[10,14]])),np.zeros(2),[2500,12.]);assert result['central90_inclusion']==[True,True]
    with pytest.raises(ValueError):h.prefix_only(np.zeros(60))

def test_all32_denominator_and_perobject4_noise_means_no_population_CI():
    rows=[dict(**v,qualified=True,policies=dict(paired_adaptive_minus_fixed_MSE_m2=float(v['object_index']),paired_adaptive_minus_random_MSE_m2=-1.))for v in h.trials()];out=h.aggregate(rows);assert out['equal_object_mean_adaptive_minus_fixed_MSE_m2']==3.5 and not out['population_or_nominal_calibration_claim']
    rows[0]['qualified']=False;out=h.aggregate(rows);assert out['qualified_trials']==31 and out['comparison_blocked']and out['planned_objects']==8
    with pytest.raises(ValueError):h.aggregate(rows[:-1])
    rows[0]['object_index']=7
    with pytest.raises(ValueError):h.aggregate(rows)
