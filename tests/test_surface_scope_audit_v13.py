import copy,hashlib,json
from pathlib import Path
import pytest
from scripts.coupled_contact_v13 import surface_scope_audit as audit

def fixture(root):
    primary=['slide','curved','moving_sphere'];stress=['edge_drop'];p=dict(adopted_scope='primary_with_stress',primary_forecast_scenarios=primary,stress_forecast_scenarios=stress,forecast_scenarios=['slide','edge_drop','curved','moving_sphere'])
    def save(name,data):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    decision=save('decision.json',dict(status='adopted',adopted_scope=p['adopted_scope'],primary_forecast_scenarios=primary,stress_forecast_scenarios=stress))
    p.update(coarse_cells=[4]*3,dt_coarse=12.5e-6,surface_spacing_m=.0015,friction_quadrature_level=1,reference_settings_by_scenario={s:dict(cells=[8]*3,companion_cells=[10]*3,dt=25e-6) for s in ['squeeze']+primary})
    entry=dict(qualified=True,primary_cells=8,companion_cells=10,dt=25e-6,half_dt=12.5e-6,predictor_cells=4,predictor_dt=12.5e-6,surface_spacing_m=.0015,friction_level=1,source_hashes_verified=True,pending=[],gates={g:dict(passed=True,expected_coverage_pass=True,source_hashes_verified=True) for g in ['spatial','primary_time','companion_time','contact','endpoints']})
    evidence=save('evidence.json',{'scenarios':{s:entry for s in ['squeeze']+primary}})
    q=dict(adopted_scope=p['adopted_scope'],primary_forecast_scenarios=primary,stress_forecast_scenarios=stress,primary_gates_pass=True,all_gates_pass=False,decision=decision,primary_evidence={s:[dict(evidence,passed_key=['scenarios',s,'qualified'])] for s in ['squeeze']+primary})
    def freeze():return dict(physics_qualification_sha256=save('reports/coupled-surface-v13/physics-development-qualification.json',q)['sha256'])
    return p,q,save,freeze

def test_scope_requires_hashed_true_evidence_and_preserves_failed_full_status(tmp_path):
    p,q,save,freeze=fixture(tmp_path);assert audit.validate_scope(p,q,freeze(),tmp_path)['primary_reserved_expected']==24
    q['all_gates_pass']=True
    with pytest.raises(ValueError,match='overall failed'):audit.validate_scope(p,q,freeze(),tmp_path)
    q['all_gates_pass']=False
    ref=save('evidence.json',{'pass':False})
    q['primary_evidence']={s:[dict(ref,passed_key=['pass'])] for s in q['primary_evidence']}
    with pytest.raises(ValueError,match='does not pass'):audit.validate_scope(p,q,freeze(),tmp_path)

def test_adoption_and_exact_calibration_coverage_required(tmp_path):
    p,q,save,freeze=fixture(tmp_path);q['decision']=save('decision.json',dict(status='conditional',adopted_scope=p['adopted_scope'],primary_forecast_scenarios=p['primary_forecast_scenarios'],stress_forecast_scenarios=p['stress_forecast_scenarios']))
    with pytest.raises(ValueError,match='not adoption'):audit.validate_scope(p,q,freeze(),tmp_path)
    q['decision']=save('decision.json',dict(status='adopted',adopted_scope=p['adopted_scope'],primary_forecast_scenarios=p['primary_forecast_scenarios'],stress_forecast_scenarios=p['stress_forecast_scenarios']));del q['primary_evidence']['squeeze']
    with pytest.raises(ValueError,match='evidence scenes'):audit.validate_scope(p,q,freeze(),tmp_path)

def test_supplement_fixed_membership_keeps_failures_and_unqualified_rows():
    from scripts.coupled_contact_v13.surface_ablation import cohort_summary
    p=dict(objects=[dict(id=f'o{i}',split='reserved' if i>=2 else 'development') for i in range(10)],primary_forecast_scenarios=['slide','curved','moving_sphere'],stress_forecast_scenarios=['edge_drop'],forecast_scenarios=['slide','edge_drop','curved','moving_sphere'])
    rows=[dict(object=o['id'],scenario=s,joint_reference_qualified=False,results=dict(vertex=dict(complete=False),nominal=dict(complete=True,coordinate_rmse_m=.01)),surface_fem_gain_mm=dict(nominal=.2)) for o in p['objects'] for s in p['forecast_scenarios']]
    independent=audit.supplement_cohorts(rows,p);produced=cohort_summary(rows,p)
    def equal(a,b):
        if isinstance(a,dict):
            assert set(a)==set(b)
            for key in a:equal(a[key],b[key])
        elif isinstance(a,float):assert a==pytest.approx(b,abs=1e-12)
        else:assert a==b
    equal(produced,independent)
    r=independent['reserved']['primary'];assert r['expected']==24 and r['reference_qualified']==0 and r['methods']['vertex']['failed_or_missing']==24 and r['methods']['nominal']['complete']==24
    assert independent['reserved']['full']['expected']==32 and independent['all_objects']['full']['expected']==40

def test_main_cohort_independent_audit_catches_dynamic_quality_filter(tmp_path,monkeypatch):
    import numpy as np
    from scripts.coupled_contact_v13 import surface_report,report
    p=dict(objects=[dict(id=f'o{i}',split='reserved' if i>=2 else 'development') for i in range(10)],primary_forecast_scenarios=['slide','curved','moving_sphere'],stress_forecast_scenarios=['edge_drop'],forecast_scenarios=['slide','edge_drop','curved','moving_sphere'])
    for split in ('development','reserved'):
        rows=[dict(object=o['id'],scenario=s,joint_reference_qualified=False,results=dict(fem=dict(complete=True,coordinate_rmse_m=.01,shape_rmse_m=.002),native=dict(complete=False))) for o in p['objects'] if o['split']==split for s in p['forecast_scenarios']]
        (tmp_path/(split+'-results.json')).write_text(json.dumps({'cases':rows}))
    monkeypatch.setattr(surface_report,'REPORT',tmp_path);cohorts=surface_report.forecast_cohorts(p,report)
    path=tmp_path/'summary.json';path.write_text(json.dumps({'forecast_cohorts':cohorts}))
    class Audit:
        def __init__(self):self.errors=[]
        def check(self,ok,label):
            if not ok:self.errors.append(label)
        def close(self,a,b,label):self.check(np.isclose(a,b),label)
    a=Audit();audit.verify_main_cohorts(a,p,tmp_path);assert not a.errors
    cohorts['reserved']['primary']['expected']=0;path.write_text(json.dumps({'forecast_cohorts':cohorts}))
    a=Audit();audit.verify_main_cohorts(a,p,tmp_path);assert any('fixed cohort accounting' in x for x in a.errors)
