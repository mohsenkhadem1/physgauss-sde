import copy
import numpy as np
import pytest
from experiments import campaign_fresh64_nonlinear_calibration_v01 as study


def test_private_streams_reproducible_with_separate_toy_seeds(monkeypatch):
    monkeypatch.setattr(study,'PARAMETER_SEED_BASE',10100);monkeypatch.setattr(study,'NOISE_SEED_BASE',10400)
    a=study.draw_fresh_object(0);b=study.draw_fresh_object(0);c=study.draw_fresh_object(1)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    assert not np.array_equal(a['unit'],c['unit'])and a['probe_noise'].shape==(2,3,24)
    with pytest.raises(ValueError):study.draw_fresh_object(64)


def test_prefix_probe_duplicates_have_independent_noise_and_alternative_phases():
    response=np.arange(144.).reshape(6,24);pn=np.full(24,.25);qn=np.empty((2,3,24));qn[0]=.5;qn[1]=.75
    prefix=study.paired_observations(response,pn,qn)
    normal=study.paired_observations(response,pn,qn,family=0,reset=0)
    axial=study.paired_observations(response,pn,qn,family=1,reset=0)
    assert len(prefix)==1 and len(normal)==2 and len(axial)==2 and normal[0][0]==normal[1][0]
    np.testing.assert_array_equal(normal[0][1],prefix[0][1]);np.testing.assert_array_equal(normal[1][1]-normal[0][1],np.full(24,.25))
    assert axial[1][0]=='paired_axial_x180'


def test_pit_uses_continuous_density_callback_at_true_unit():
    class Reference:
        def __init__(self):self.calls=[]
        def cdf(self,d,q,**kwargs):self.calls.append((d,q,kwargs));return q*q
    ref=Reference();final=dict(unit_quantiles=np.array([[.1,.5,.9],[.1,.5,.9]]),unit_mean=np.array([.5,.5]),log_evidence=3.)
    result=study.exact_density_calibration(ref,final,[.3,.8],outer=512,inner=512)
    np.testing.assert_allclose(result['PIT'],[.09,.64]);assert len(ref.calls)==2 and result['covered'].all()


def records():
    data=[]
    for i in range(64):
        entry=dict(qualified=True,physical_valid=True,target_RMS_m=float(i+1),covered=[i!=0,True],PIT=[(i+.5)/64,(i+.5)/64],intrinsic_interval_width=[.2,.3],intrinsic_signed_bias=[.01,-.02])
        data.append(dict(index=i,posteriors={m:{key:copy.deepcopy(entry)for key in('prefix','normal-r0','normal-r1','normal-r2','axial-r0','axial-r1','axial-r2')}for m in study.MODELS},oracle_RMS_m=dict(linear=.1,nonlinear=.01)))
    return data


def test_fixed64_denominators_and_phase_major_parameter_order():
    data=records();data[0]['posteriors']['nonlinear']['normal-r0']['qualified']=False
    covered,pit=study.primary_arrays(data,'nonlinear')
    assert covered.shape==(64,6)and np.isnan(covered[0,2:4]).all()and np.isfinite(covered[:,0]).all()
    result=study.summarize_calibration(data)
    assert result['comparison_blocked']and result['nonlinear']['entries'][2]['object_count']==64
    with pytest.raises(ValueError):study.primary_arrays(data[:-1],'linear')


def test_all_metrics_share64_bootstrap_and_mean_resets_within_object(monkeypatch):
    data=records();data[0]['posteriors']['nonlinear']['normal-r1']['target_RMS_m']=7.
    calls=[]
    def bootstrap(matrix,*,labels,seed,resamples):
        calls.append((matrix.copy(),labels));assert matrix.shape[0]==64 and resamples==10000
        return dict(same_object_resamples_across_all_metrics=True)
    monkeypatch.setattr(study,'paired_cluster_intervals',bootstrap)
    result=study.paired_prediction_metrics(data)
    assert result['complete']and len(calls)==1
    matrix,labels=calls[0]
    assert matrix[0,labels.index('nonlinear_normal_mean3_RMS_m')]==3.
    assert matrix[0,labels.index('nonlinear_minus_linear_prefix_RMS_m')]==0.
    data[0]['posteriors']['linear']['prefix']['qualified']=False
    assert study.paired_prediction_metrics(data)['comparison_blocked']and len(calls)==1


def test_entire_missing_object_posteriors_keep64_and_block_comparison():
    data=records();data[7].pop('posteriors')
    calibration=study.summarize_calibration(data)
    assert calibration['comparison_blocked']
    assert all(row['object_count']==64 and row['unresolved_coverage']==1 for row in calibration['nonlinear']['entries'])
    assert study.paired_prediction_metrics(data)['comparison_blocked']


def test_invalid_pit_is_unresolved_not_clipped():
    class Reference:
        def cdf(self,*args,**kwargs):return np.nextafter(1.,np.inf)
    final=dict(unit_quantiles=np.array([[.1,.5,.9],[.1,.5,.9]]),unit_mean=np.array([.5,.5]),log_evidence=0.)
    with pytest.raises(FloatingPointError,match='PIT'):study.exact_density_calibration(Reference(),final,[.5,.5],outer=512,inner=512)
