import json
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.validation_summary import aggregate_cases,verify_score,sha


def row(case,cell,methods):
    return dict(case=case,cell=cell,methods={k:dict(complete=True,coordinate_rmse_mm=v[0],aligned_shape_rmse_mm=v[1]) for k,v in methods.items()})


def test_failed_case_denominators_and_equal_cell_macro():
    rows=[row('a','one',{'flow_raw':(1.,2.)}),row('b','one',{'flow_raw':(3.,4.)}),
          row('c','two',{'flow_raw':(10.,9.)}),row('d','two',{})]
    methods,_=aggregate_cases(rows);m=methods['flow_raw']
    assert m['expected_cases']==4 and m['complete_cases']==3 and m['incomplete_cases']==['d']
    assert m['conditional_equal_case_means']['coordinate_rmse_mm']==pytest.approx(14/3)
    assert m['conditional_equal_cell_macro_means']['coordinate_rmse_mm']==6
    assert m['complete_cells']==1 and m['expected_cells']==2


def test_pair_intersection_metric_sign_and_secondary_label():
    rows=[row('a','one',{'flow_raw':(1.,5.),'rigid_contact':(3.,2.)}),
          row('b','one',{'flow_raw':(9.,9.)})]
    _,pairs=aggregate_cases(rows);p=pairs['flow_raw minus rigid_contact']
    assert p['role']=='secondary' and p['paired_complete_cases']==1 and p['expected_cases']==2
    assert p['excluded_cases']==['b']
    assert p['conditional_mean_difference']==dict(coordinate_rmse_mm=-2.,aligned_shape_rmse_mm=3.)
    assert p['wins']==dict(coordinate_rmse_mm=1,aligned_shape_rmse_mm=0)


def sealed_fixture(tmp_path):
    run=tmp_path/'run';run.mkdir();package=tmp_path/'package';package.mkdir();(package/'manifest.json').write_text('{}')
    nodes=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
    np.savez(run/'prediction.npz',nodes=nodes,observed_initial=nodes,observation_matrix=np.eye(3),flow_raw=np.tile(nodes,(4,1,1)))
    plan=dict(case='case',validation_protocol_sha256='abc',prefix_frames=[0,1],point_ids=[0,1,2],args={'nodes':3},input_digest='inputs',input_package=str(package),input_package_manifest_sha256=sha(package/'manifest.json'))
    (run/'plan.json').write_text(json.dumps(plan));(run/'inference-sealed.json').write_text(json.dumps(dict(plan_sha256=sha(run/'plan.json'),forecast_target_access=False,train_end=2)))
    (run/'complete.json').write_text(json.dumps(dict(completed=True,held_target_access=False,fit_end=2,frames=4,prediction_sha256=sha(run/'prediction.npz'))))
    score=dict(case='case',data_role='prospective_validation',validation_protocol_sha256='abc',interval=[2,4],nodes=3,observation_count=3,observed_node_ids=[4,5,6],baseline_budget=dict(frames=[0,1],point_ids=[0,1,2]),source_sha256=dict(prediction=sha(run/'prediction.npz'),plan=sha(run/'plan.json')),methods={'flow_raw':{'summary':dict(total_frames=2,valid_frames=2,failed_frames=0,no_reference_frames=0)}})
    return run,score


def test_score_verified_against_real_seals(tmp_path):
    run,score=sealed_fixture(tmp_path)
    budget,identity=verify_score(run,score,case='case',protocol_sha256='abc')
    assert budget==score['baseline_budget'] and identity[0]==[2,4]
    with (run/'prediction.npz').open('ab') as f:f.write(b'tampered')
    with pytest.raises(ValueError,match='Changed sealed'):verify_score(run,score,case='case',protocol_sha256='abc')


@pytest.mark.parametrize('change',["interval","case","budget","source","counts"])
def test_reject_mismatched_score_binding(tmp_path,change):
    run,score=sealed_fixture(tmp_path)
    if change=='interval':score['interval']=[1,4]
    if change=='case':score['case']='other'
    if change=='budget':score['baseline_budget']['point_ids']=[0,1]
    if change=='source':score['source_sha256']['plan']='wrong'
    if change=='counts':score['methods']['flow_raw']['summary']['valid_frames']=1
    with pytest.raises(ValueError):verify_score(run,score,case='case',protocol_sha256='abc')


def test_full_failed_cohort_retained_without_any_target_read(tmp_path):
    from scripts.gaussian_boundary_flow_v02.validation_summary import summarize
    root=tmp_path/'cohort';root.mkdir()
    cases=[dict(case=f'{suite}-task{task}-demo{episode}',subset=suite,task=task,episode=episode)
           for suite in ['object-soft','spatial-soft'] for task in range(3) for episode in [50,51]]
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(dict(status='frozen',role='validation',cases=cases,source_sha256={})))
    digest=sha(protocol);gate_rows=[]
    for spec in cases:
        folder=root/spec['case'];folder.mkdir();status=folder/'prediction-status.json'
        status.write_text(json.dumps(dict(case=spec['case'],protocol_sha256=digest,terminal=True,stages={'physics':{'state':'failed'}})))
        gate_rows.append(dict(case=spec['case'],status_sha256=sha(status)))
    (root/'evaluation-gate.json').write_text(json.dumps(dict(protocol_sha256=digest,all12_terminal_before_any_target_read=True,cases=gate_rows)))
    report=summarize(root,protocol)
    assert len(report['cases'])==12 and not report['target_data_access']
    assert report['methods']['flow_raw']['expected_cases']==12
    assert report['methods']['flow_raw']['complete_cases']==0
    assert report['methods']['flow_raw']['conditional_equal_case_means']['coordinate_rmse_mm'] is None
    assert report['methods']['rigid_contact']['role']=='secondary'
    (root/cases[0]['case']/'prediction-status.json').write_text('{}')
    with pytest.raises(ValueError,match='terminal status'):summarize(root,protocol)
