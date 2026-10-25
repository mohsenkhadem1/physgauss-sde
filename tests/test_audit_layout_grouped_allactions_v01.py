import copy,pytest
from experiments import audit_layout_grouped_allactions_v01 as a
from tests.test_layout_grouped_allactions_v01 import rows
GATES=dict(outcome_relative=1e-4,latent_relative=1e-3)
def test_independent_risk_and_tie_arithmetic():
    data=rows();report=a.run.classify(data,GATES);assert a.check_rows(data,GATES,report)['qualified']
    changed=copy.deepcopy(report);changed['actions'][0]['outer_error']=1.
    with pytest.raises(ValueError,match='recomputed'):a.check_rows(data,GATES,changed)
def test_missing_density_and_job_membership_refused():
    data=rows();report=a.run.classify(data,GATES)
    with pytest.raises(ValueError):a.check_rows(data[:-1],GATES,report)
    data[0]['result']['predictive_mass']=.5
    with pytest.raises(ValueError,match='density'):a.check_rows(data,GATES,report)
def test_unresolved_all48_retained_and_no_false_winner():
    data=rows();data[0]['result']['expected_target_variance']+=1e-11;report=a.run.classify(data,GATES);result=a.check_rows(data,GATES,report);assert not result['qualified']and result['actions_qualified']==7 and not result['unique_winner_resolved']
