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
