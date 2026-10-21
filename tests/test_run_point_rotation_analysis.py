"""Cross-module scientific/role integration, without cohort values or physics."""
import numpy as np
import pytest
from experiments import run_point_rotation_analysis as runner
from experiments.campaign_joint_gaussian_discrepancy import JointGaussianDiscrepancy
from experiments.campaign_projected_point_targets import ProjectedPointTargets
from experiments.censored_range_observation import CensoredRanges


def plan():
    objects=[dict(index=i,object_id=f'point_prior_{i:03d}',seed=2026120100+i,
        role='CALIBRATION' if i<16 else 'RESERVED_EVALUATION') for i in range(48)]
    return dict(schema='independent-point-rotation-prior-cohort-v0.2',objects=objects,
        roles=dict(CALIBRATION=list(range(16)),RESERVED_EVALUATION=list(range(16,48))))


def test_new_cohort_and_explicit_reserved_release_required():
    p=plan();kw=dict(execution_sha256='execution',method_sha256='method')
    assert len(runner.eligible_objects(p,'CALIBRATION',**kw))==16
    with pytest.raises(ValueError):runner.eligible_objects(p,'RESERVED_EVALUATION',**kw)
    p['objects'][0]['seed']=2026112000
    with pytest.raises(ValueError):runner.eligible_objects(p,'CALIBRATION',**kw)


def test_release_cannot_swap_method_execution_or_calibration():
    p=plan();marker=dict(path='marker',sha256='hash')
    freeze=dict(completed=True,method_sha256='method',execution_protocol_sha256='execution',calibration_object_indices=list(range(16)))
    release=dict(schema='point-rotation-reserved-release-v0.2',status='FROZEN_EXPLICIT_RESERVED_EVALUATION_RELEASE',
        method_sha256='method',execution_protocol_sha256='execution',object_indices=list(range(16,48)),calibration_freeze=marker)
    kw=dict(execution_sha256='execution',method_sha256='method',release=release,calibration_freeze=freeze,calibration_freeze_record=marker)
    assert len(runner.eligible_objects(p,'RESERVED_EVALUATION',**kw))==32
    release['execution_protocol_sha256']='old'
    with pytest.raises(ValueError):runner.eligible_objects(p,'RESERVED_EVALUATION',**kw)


def test_conditional_fantasy_design_selects_identifying_view_from_predictions_only():
    model=JointGaussianDiscrepancy(np.zeros(216),np.zeros((216,1)),np.zeros(9),np.zeros((9,1)))
    base=model.condition(CensoredRanges(np.ones((2,72),bool),np.zeros((2,72))),
        CensoredRanges(np.ones(72,bool),np.zeros(72)),np.arange(72))
    x=np.zeros((2,9));x[:,3]=[-.001,.001]
    target=ProjectedPointTargets(x,np.zeros(9),np.zeros((9,1)),np.zeros(2,dtype=np.int64),layout=(1,3,3))
    predictions=[]
    for separation in [0.,.004,0.,.0001]:
        values=np.zeros((2,36));values[:,0]=[-separation,separation]
        predictions.append(CensoredRanges(np.ones((2,36),bool),values))
    result=runner.select_view(target,base,predictions,model,device='cpu')
    assert result['selected_index']==1 and result['actual_frame24_opened'] is False
    assert result['expected_target_risk_m2'][1] < result['expected_target_risk_m2'][3]/100
    assert result['risk_includes_conditional_discrepancy'] and result['latent_discrepancy_shared_between_views']


def test_new_cohort_fitted_discrepancy_is_rejected_before_archive_load(monkeypatch):
    monkeypatch.setattr(runner,'checked',lambda x:'metadata')
    monkeypatch.setattr(runner,'read',lambda x:dict(completed=True,objects=16,new_cohort_response_or_truth_read=True))
    with pytest.raises(ValueError,match='OLD-development'):runner.load_discrepancy({})


def test_log_increment_weights_are_translation_invariant_and_numerically_stable():
    values=np.array([-5000.,0.,1.])
    np.testing.assert_allclose(runner.weights_from_increment(values),runner.weights_from_increment(values+10000))
    with pytest.raises(ValueError):runner.weights_from_increment(np.array([np.nan]))
