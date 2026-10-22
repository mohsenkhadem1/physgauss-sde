import copy,json
import pytest
from experiments import run_campaign_c38_nonlinear_table as c

def test_baseline_numerical_prerequisite_does_not_filter_J_or_spatial_failures():
 summary=dict(completed=True,all8_basis_original_residual_checks_passed=True,all8_independent_direction_checks_passed=True,predictions_completed=28,all_prediction_J_passed=False,all_linear_q63_spatial_indicators_passed=False)
 terminal=dict(completed=True,owned_PID_absence={'123':True});seal=dict(completed=True,all28_predictions_sealed_before_nonlinear_target_access=True)
 assert c.baseline_numeric_check(summary,terminal,seal)
 bad=copy.deepcopy(summary);bad['all8_independent_direction_checks_passed']=False
 with pytest.raises(ValueError,match='numerical'):c.baseline_numeric_check(bad,terminal,seal)
 terminal['owned_PID_absence']['123']=False
 with pytest.raises(ValueError,match='terminal'):c.baseline_numeric_check(summary,terminal,seal)

def test_exact_disjoint_budget_and_output_contract():
 plans=c.catalog.phase_plan(range(6))
 assert [len(x['cases']) for x in plans]==[9,9,8]
 assert c.PHASE_CAPS==(8100,8100,7200)
 assert [x['CPU_affinity_cores'] for x in plans]==[[0,1],[2,3],[4,5]]
 assert len({x['output'] for x in plans})==3
 assert sum(len(x['cases'])*2 for x in plans)==52

def test_draft_dispatch_rejected_without_physics_or_output(tmp_path,monkeypatch):
 monkeypatch.setattr(c.catalog,'ROOT',tmp_path)
 p=tmp_path/'p.json';p.write_text(json.dumps(dict(schema='DRAFT',phase_id='gpu0')))
 monkeypatch.setattr(c.worker,'worker',lambda *a:pytest.fail('Physical worker called'))
 with pytest.raises(ValueError,match='prospective'):c.dispatch('p.json',c.worker.sha(p),'x')
 assert list(tmp_path.iterdir())==[p]
