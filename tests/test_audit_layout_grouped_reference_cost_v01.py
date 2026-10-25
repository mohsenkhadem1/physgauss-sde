"""Manufactured scalar certificates only; no campaign files, FE or arrays."""
from copy import deepcopy
import numpy as np
import pytest
from experiments import audit_layout_grouped_reference_cost_v01 as m

def reference():
    r=dict(passed=True,log_evidence_difference=1e-5,maximum_CDF_difference=1e-5,maximum_unit_quantile_difference=1e-5,maximum_unit_mean_difference=1e-5,maximum_predictive_mean_difference_m=1e-8,maximum_predictive_SD_difference_m=1e-8)
    histories={axis:[dict(r,coarse=a,fine=b)for a,b in((128,256),(256,512))]for axis in('inner','outer')}
    adaptive=dict(passed=True,epsrel=[1e-6,1e-8],CDF_checks_per_epsrel=6,log_evidence_difference=1e-5,maximum_CDF_error=1e-5,adaptive_refinement_log_evidence_difference=1e-8,adaptive_refinement_maximum_CDF_difference=1e-8,maximum_QUADPACK_relative_error_estimates={'1e-06':1e-7,'1e-08':1e-9})
    d=dict(schema='conditional-amplitude-v02',gates=m.GATES,levels=[dict(outer=o,inner=i,status='computed')for o,i in((512,128),(512,256),(512,512),(128,512),(256,512))],successive=histories,independent_adaptive=adaptive)
    return dict(qualified=True,diagnostics=d,strict=dict(passed=True))

def test_all_original_reference_and_strict2um_gates():
    r=reference();assert m.reference_checks(r)['passed']
    r['diagnostics']['successive']['outer'][-1]['maximum_predictive_mean_difference_m']=3e-6;r['strict']['passed']=False;assert not m.reference_checks(r)['passed']
    r['strict']['passed']=True
    with pytest.raises(ValueError,match='strict2um'):m.reference_checks(r)

def test_adaptive_CDF_error_and_relative_error_cannot_claim_pass():
    for field,value in [('maximum_CDF_error',.006),('CDF_checks_per_epsrel',5),('maximum_QUADPACK_relative_error_estimates',{'1e-06':1e-4,'1e-08':1e-8})]:
        r=reference();r['diagnostics']['independent_adaptive'][field]=value;r['strict']['passed']=False;assert not m.reference_checks(r)['passed']

def test_missing_levels_or_changed_gates_refused():
    r=reference();r['diagnostics']['levels'].pop()
    with pytest.raises(ValueError,match='5reference'):m.reference_checks(r)
    r=reference();r['diagnostics']['gates']=dict(m.GATES,CDF=.01)
    with pytest.raises(ValueError,match='schema/gates'):m.reference_checks(r)

def decision():return dict(reference_checks=dict(passed=True,per_category=[dict(log_evidence=1e-5,mean_m=1e-8,SD_m=1e-8,passed=True)for _ in range(4)],mixture_mean_error_m=1e-8,mixture_SD_error_m=1e-8,layout_TV=.001))

def test_new_decision_measure_requires_ALLcategory_ANDmixture_checks():
    r=decision();assert m.decision_checks(r)['passed'];r['reference_checks']['mixture_SD_error_m']=3e-6;r['reference_checks']['passed']=False;assert not m.decision_checks(r)['passed']
    r['reference_checks']['passed']=True
    with pytest.raises(ValueError,match='Mixed48'):m.decision_checks(r)
    r=decision();r['reference_checks']['per_category'].pop()
    with pytest.raises(ValueError,match='Every category'):m.decision_checks(r)

def utility(o,n,risk):
    level=dict(expected_target_variance=risk,prior_target_variance=2e-9,variance_reduction=2e-9-risk,omitted_risk_bound=1e-24)
    return dict(job=dict(outer=o,amplitude=n,action=0),levels=[dict(level,order=8),dict(level,order=16)],outcome_check=dict(error=1e-24,passed=True,gate_relative_to_prior=1e-4))

def test_outcome_and_latent_risk_checks_recomputed_without_model():
    rows=[utility(512,32,1e-9),utility(256,64,1e-9),utility(512,64,1e-9)]
    assert all(m.outcome_checks(r)['passed']for r in rows)
    reported={a:dict(error=0.,relative_to_prior=0.,passed=True)for a in('amplitude','outer')};assert all(v['passed']for v in m.risk_refinement_checks(rows,reported).values())
    rows[0]['levels'][-1]['expected_target_variance']=1.5e-9
    with pytest.raises(ValueError):m.outcome_checks(rows[0])
    with pytest.raises(ValueError,match='refinement'):m.risk_refinement_checks(rows,reported)


def test_public_inventory_omits_category_and_action_arrays(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'LOCAL',tmp_path/'local');m.LOCAL.mkdir()
    monkeypatch.setattr(m.run,'record',lambda path:dict(path=str(path.relative_to(tmp_path)),sha256=m.run.sha(path)))
    for name in(m.SOURCE,m.TEST):p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic source only')
    summary=dict(completed=True,integrity_passed=True,execution_clean=True,continuous_categories_qualified=4,categories_retained=4,decision_banks_completed=5,oneaction_jobs_completed=3,oneaction_numerically_qualified=True,all8_ranks_pending=True,policy_use_blocked=True,original_grid_negative_preserved=True,scope='oneaction development only',external_wall_s=1.,peak_RSS_bytes=1,maximum_continuous_field_error_m=1e-8,maximum_decision_moment_error_m=1e-8,risk_refinements={},no_model_reference_or_array_replay=True,layout_probability=['SECRET']*4,expected_target_risk=['SECRET']*8)
    review=dict(summary=summary,source_SHA256={},protocol={},release={},execution={},worker={});m.run.write(m.LOCAL/'independent-terminal-review.json',review)
    monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('No arrays during scalar inventory'))
    inventory=m.publication_inventory();assert inventory['path'].endswith('publication-inventory-completed.json')
    public=(tmp_path/'reports/layout-grouped-reference-cost-v01/summary.json').read_text();assert 'SECRET'not in public and 'layout_probability'not in public and 'expected_target_risk'not in public
