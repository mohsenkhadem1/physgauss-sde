import copy,hashlib,json
import pytest
from scripts.coupled_contact_v13 import surface_scope,surface_report,report,surface_pipeline


def fixture(tmp_path,stress=True):
    primary=['slide','curved','moving_sphere'];extra=['edge_drop']
    if not stress:primary+=extra;extra=[]
    scope='primary_with_stress' if stress else 'all_forecast_primary'
    p=dict(forecast_scenarios=['slide','curved','moving_sphere','edge_drop'],primary_forecast_scenarios=primary,stress_forecast_scenarios=extra,adopted_scope=scope)
    decision=dict(status='adopted',**{k:p[k] for k in ['adopted_scope','primary_forecast_scenarios','stress_forecast_scenarios']})
    def save(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    p['reference_settings_by_scenario']={s:dict(cells=[8]*3,companion_cells=[12]*3,dt=25e-6) for s in ['squeeze']+primary}
    ref=save('evidence.json',dict(scenarios={s:dict(qualified=True,primary_cells=8,companion_cells=12,dt=25e-6) for s in ['squeeze']+primary}))
    q=dict(**{k:p[k] for k in ['adopted_scope','primary_forecast_scenarios','stress_forecast_scenarios']},primary_gates_pass=True,all_gates_pass=not stress,decision=save('decision.json',decision),primary_evidence={s:[dict(ref,passed_key=['scenarios',s,'qualified'])] for s in ['squeeze']+primary})
    return p,q


@pytest.mark.parametrize('stress',[True,False])
def test_explicit_scope_evidence_and_no_false_allgate_claim(tmp_path,stress):
    p,q=fixture(tmp_path,stress)
    assert surface_scope.validate_adoption(p,q,tmp_path)==p['adopted_scope']
    bad=copy.deepcopy(q);bad['primary_gates_pass']=False
    with pytest.raises(RuntimeError):surface_scope.validate_adoption(p,bad,tmp_path)
    bad=copy.deepcopy(q);bad['all_gates_pass']=stress
    with pytest.raises(RuntimeError):surface_scope.validate_adoption(p,bad,tmp_path)
    bad=copy.deepcopy(q);bad['primary_evidence'].pop('squeeze')
    with pytest.raises(RuntimeError):surface_scope.validate_adoption(p,bad,tmp_path)


def test_hash_only_or_false_evidence_cannot_authorize(tmp_path,monkeypatch):
    p,q=fixture(tmp_path)
    evidence=tmp_path/'evidence.json';data=json.loads(evidence.read_text());data['scenarios']['slide']['qualified']=False;evidence.write_text(json.dumps(data))
    digest=hashlib.sha256(evidence.read_bytes()).hexdigest()
    for refs in q['primary_evidence'].values():refs[0]['sha256']=digest
    with pytest.raises(RuntimeError,match='did not pass'):surface_scope.validate_adoption(p,q,tmp_path)
    m=surface_pipeline.implementation();monkeypatch.setattr(m,'REPORT',tmp_path)
    (tmp_path/'protocol.json').write_text(json.dumps(p));(tmp_path/'physics-development-qualification.json').write_text(json.dumps(q))
    frozen=dict(source_sha256={},protocol_sha256=hashlib.sha256((tmp_path/'protocol.json').read_bytes()).hexdigest(),physics_qualification_sha256=hashlib.sha256((tmp_path/'physics-development-qualification.json').read_bytes()).hexdigest())
    (tmp_path/'method-freeze.json').write_text(json.dumps(frozen))
    monkeypatch.setattr(surface_pipeline,'ROOT',tmp_path)
    with pytest.raises(RuntimeError,match='did not pass'):m.check_freeze()


def test_fixed_primary_counts_ignore_quality_and_preserve_missing(tmp_path,monkeypatch):
    p,_=fixture(tmp_path)
    p['objects']=[dict(id=f'd{i}',split='development') for i in range(2)]+[dict(id=f'r{i}',split='reserved') for i in range(8)]
    monkeypatch.setattr(surface_report,'REPORT',tmp_path)
    rows=[dict(object='r0',scenario='slide',joint_reference_qualified=False,results=dict(fem=dict(complete=True,coordinate_rmse_m=.003,shape_rmse_m=.001)))]
    (tmp_path/'reserved-results.json').write_text(json.dumps(dict(cases=rows)))
    result=surface_report.forecast_cohorts(p,report)
    primary=result['reserved']['primary'];full=result['reserved']['full']
    assert primary['expected']==24 and full['expected']==32
    assert result['all_objects']['primary']['expected']==30 and result['all_objects']['full']['expected']==40
    assert primary['methods']['fem']['coordinate_mean_mm']==3
    assert primary['reference_qualified']==0 and primary['methods']['fem']['complete']==1
    assert len(primary['methods']['fem']['failures'])==23


def test_rounded_drop_cannot_enter_narrowed_stress_without_new_authorization(tmp_path):
    p,_=fixture(tmp_path)
    p['forecast_scenarios'][-1]='rounded_support_drop';p['stress_forecast_scenarios']=['rounded_support_drop']
    with pytest.raises(RuntimeError,match='sharp edge_drop'):surface_scope.partition(p)


def test_unrelated_true_evidence_and_wrong_grid_are_rejected(tmp_path):
    p,q=fixture(tmp_path)
    path=tmp_path/'unrelated.json';path.write_text(json.dumps(dict(passed=True)))
    ref=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),passed_key=['passed'])
    bad=copy.deepcopy(q);bad['primary_evidence']['slide']=[ref]
    with pytest.raises(RuntimeError,match='Scene-specific'):surface_scope.validate_adoption(p,bad,tmp_path)
    p['reference_settings_by_scenario']['slide']['cells']=[12]*3
    with pytest.raises(RuntimeError,match='discretization mismatch'):surface_scope.validate_adoption(p,q,tmp_path)
