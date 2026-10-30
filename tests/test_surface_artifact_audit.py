from scripts.coupled_contact_v13 import surface_verify,surface_report,verify,report


def test_surface_verifier_namespace_and_cohort_are_separate():
    old=(verify.BASE,verify.REPORT)
    impl=surface_verify.implementation()
    assert impl is not verify
    assert impl.BASE==surface_verify.BASE and impl.REPORT==surface_verify.REPORT
    assert (verify.BASE,verify.REPORT)==old
    objects=[dict(id=f'dev{i}',split='development') for i in range(2)]+[dict(id=f'fresh{i}',split='reserved') for i in range(8)]
    p=dict(objects=objects,forecast_scenarios=['a','b','c','d'])
    a=impl.Audit();surface_verify.validate_cohort(a,p);assert not a.errors
    p['objects'][-1]['split']='validation'
    a=impl.Audit();surface_verify.validate_cohort(a,p);assert len(a.errors)==2


def test_surface_report_namespace_is_separate():
    old=report.REPORT;impl=surface_report.implementation()
    assert impl is not report
    assert impl.REPORT==surface_report.REPORT and report.REPORT==old


def test_supplement_report_uses_matched_pairs_and_keeps_failures(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(surface_report,'REPORT',tmp_path)
    main={'cases':[{'object':'a','scenario':'one','results':{'fem':{'complete':True,'coordinate_rmse_m':.001}}},
                   {'object':'a','scenario':'two','results':{'fem':{'complete':True,'coordinate_rmse_m':.009}}}]}
    (tmp_path/'reserved-results.json').write_text(json.dumps(main))
    supplement={'expected_cases':2,'cases':[
        {'object':'a','scenario':'one','split':'reserved','results':{'vertex':{'complete':True,'coordinate_rmse_m':.003,'shape_rmse_m':.002},'nominal':{'complete':False,'failure':'nonfinite'}}},
        {'object':'a','scenario':'two','split':'reserved','results':{'vertex':{'complete':False,'failure':'J gate'},'nominal':{'complete':True,'coordinate_rmse_m':.007,'shape_rmse_m':.004}}}]}
    (tmp_path/'ablation-results.json').write_text(json.dumps(supplement))
    result,lines=surface_report.supplementary_comparisons()
    vertex=result['splits']['reserved']['vertex'];nominal=result['splits']['reserved']['nominal']
    assert vertex['paired_fem_mean_mm']==1 and vertex['mean_fem_gain_mm']==2
    assert vertex['complete']==1 and vertex['expected']==2 and vertex['means_conditional_on_success']
    assert vertex['failed_cases'][0]['reason']=='J gate'
    assert nominal['fem_losses']==1 and abs(nominal['mean_fem_gain_mm']+2)<1e-12
    assert any('vertex | 1/2' in line for line in lines)


def test_reference_accounting_counts_150_and_retains_missing(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(surface_report,'BASE',tmp_path)
    protocol=dict(objects=[dict(id=f'obj{i}') for i in range(10)],forecast_scenarios=['slide','curved','edge_drop','moving_sphere'],reference_settings_by_scenario={'edge_drop':dict(cells=[12]*3,companion_cells=[16]*3,dt=25e-6)})
    path=tmp_path/'reference'/'obj0'/'curved';path.mkdir(parents=True)
    (path/'complete.json').write_text(json.dumps(dict(success=True,simulation_settings={'cells':[8]*3})))
    (path/'spatial-quality.json').write_text(json.dumps(dict(qualified=False,reasons=['spatial companion unavailable'])))
    result=surface_report.reference_accounting(protocol)
    assert result['expected_total']==150
    assert all(level['expected']==50 for level in result['levels'].values())
    assert result['levels']['reference']['complete']==1
    assert result['levels']['reference-spatial']['present']==0
    assert result['spatial']['checked']==1 and result['spatial']['qualified']==0


def test_video_reference_label_retains_failed_qualification(tmp_path,monkeypatch):
    import json
    from scripts.coupled_contact_v13 import surface_render
    monkeypatch.setattr(surface_render,'REPORT',tmp_path)
    rows=[dict(object='ok',scenario='curved',joint_reference_qualified=True),dict(object='bad',scenario='curved',joint_reference_qualified=False,reference_qualification_reasons=['spatial difference'])]
    (tmp_path/'reserved-results.json').write_text(json.dumps(dict(cases=rows)))
    assert 'passed' in surface_render.reference_label_for_case('ok','curved')
    assert 'failed' in surface_render.reference_label_for_case('bad','curved')
    assert 'unavailable' in surface_render.reference_label_for_case('missing','curved')
    module=surface_render.implementation()
    assert module.reference_label_for_case is surface_render.reference_label_for_case


def test_stress_video_label_cannot_be_promoted_by_per_case_pass(tmp_path,monkeypatch):
    import json
    from scripts.coupled_contact_v13 import surface_render
    monkeypatch.setattr(surface_render,'REPORT',tmp_path)
    (tmp_path/'protocol.json').write_text(json.dumps(dict(stress_forecast_scenarios=['edge_drop'])))
    row=dict(object='ok',scenario='edge_drop',joint_reference_qualified=True,reference_qualified=True)
    (tmp_path/'reserved-results.json').write_text(json.dumps(dict(cases=[row])))
    label=surface_render.reference_label_for_case('ok','edge_drop')
    assert 'Exploratory stress' in label and 'development unqualified' in label
    assert 'checks passed' not in label
    assert surface_render.case_qualification('ok','edge_drop')['joint_reference_qualified'] is True


def test_supplement_explicit_primary_stress_full_tables(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(surface_report,'REPORT',tmp_path)
    def record(n,scenes):
        return dict(expected=n,present=n,scenarios=scenes,methods=dict(vertex=dict(expected=n,complete=1,failed_or_missing=n-1,mean_coordinate_mm=2.,paired_cases=1,surface_fem_gain_mm=.5)))
    cohorts=dict(reserved=dict(primary=record(24,['slide','curved','moving_sphere']),stress=record(8,['edge_drop']),full=record(32,['slide','curved','moving_sphere','edge_drop'])))
    (tmp_path/'ablation-results.json').write_text(json.dumps(dict(cases=[],expected_cases=32,forecast_cohorts=cohorts)))
    output,lines=surface_report.supplementary_comparisons()
    assert output['forecast_cohorts']==cohorts
    assert any('RESERVED / PRIMARY: expected 24' in s for s in lines)
    assert any('RESERVED / STRESS: expected 8' in s for s in lines)
    assert any('RESERVED / FULL: expected 32' in s for s in lines)
    assert any('RESERVED / FULL COHORT'==s for s in lines)
