import copy
from pathlib import Path
import pytest
from experiments import audit_fixture_gaussian_decision_screen_v01 as a
from tests.test_fixture_gaussian_decision_screen_v01 import adequate_records
from tests.test_layout_grouped_allactions_v01 import rows as action_rows

def test_independent_adequacy_matches_positive_universal_and_missing():
    for mutate in(lambda x:None,lambda x:x[0].pop('ranking'),lambda x:[r['posterior_screen'].update(prior_variance=1e-14)for r in x]):
        data=adequate_records();mutate(data);assert a.independent_adequacy(data)==a.run.design.adequacy(data)

def test_missing_membership_and_false_unique_winner_refused():
    data=adequate_records()
    with pytest.raises(ValueError,match='36'):a.independent_adequacy(data[:-1])
    data[0]['ranking']['numerical_ties']=[0,1]
    with pytest.raises(ValueError,match='winner'):a.independent_adequacy(data)

def fake_case(monkeypatch,tmp_path):
    monkeypatch.setattr(a,'LOCAL',tmp_path);monkeypatch.setattr(a.run,'record',lambda p:dict(path=str(p),sha256=a.run.sha(p)));monkeypatch.setattr(a.refs_audit,'reference_checks',lambda value:dict(passed=value['qualified']));monkeypatch.setattr(a.refs_audit,'decision_checks',lambda value:dict(passed=value['reference_checks']['passed']))
    spec=a.run.design.worklist()[0];directory=tmp_path/'case-00';directory.mkdir();row=dict(index=0,case=spec,protocol_SHA256='proto',prefix={'bound':'prefix'},status='qualified',qualified=True,categories=[],decision_levels=[],jobs=action_rows(),partial_artifacts=[]);metadata=dict(prefixes=[dict(record={'bound':'prefix'})])
    for k in range(4):
        path=directory/f'category-{k}-LOCAL.json';a.run.write(path,dict(qualified=True,diagnostics=dict(independent_adaptive_per_posterior_required=True,levels=[dict(outer=o,inner=i,status='computed',log_evidence=0.)for o,i in((512,512),(512,256),(256,512))])));rec=a.run.record(path);row['categories'].append(dict(layout=k,qualified=True,record=rec));row['partial_artifacts'].append(rec)
    screen=dict(prior_variance=2e-10,between_layout_fraction=.2,maximum_layout_probability=.25,layout_TV=0.)
    for o,n in a.run.DECISION_LEVELS:
        path=directory/f'decision-{o}-{n}-LOCAL.json';a.run.write(path,dict(checks=dict(passed=True,layout_TV=0.),metadata=dict(layout_probability=[.25]*4),posterior_screen=screen));rec=a.run.record(path);row['decision_levels'].append(dict(outer=o,amplitude=n,checks=dict(passed=True,layout_TV=0.),record=rec));row['partial_artifacts'].append(rec)
    row['ranking']=a.run.old.classify(row['jobs'],dict(outcome_relative=1e-4,latent_relative=1e-3));row['posterior_screen']=dict(screen,between_layout_fraction_indicator=0.)
    return row,spec,metadata

def test_all144_witness_contract_and_independent48_arithmetic(monkeypatch,tmp_path):
    row,spec,metadata=fake_case(monkeypatch,tmp_path);result=a.check_case(row,spec,metadata,'proto');assert result==dict(reference_count=4,references_qualified=4,decision_count=5,utility_jobs=48,rank_verified=True)
    row['ranking']['actions'][0]['outer_error']=1.
    with pytest.raises(ValueError,match='recomputed'):a.check_case(row,spec,metadata,'proto')

def test_changed_input_hash_or_between_indicator_refused(monkeypatch,tmp_path):
    row,spec,metadata=fake_case(monkeypatch,tmp_path);row['posterior_screen']['between_layout_fraction_indicator']=.3
    with pytest.raises(ValueError,match='indicator'):a.check_case(row,spec,metadata,'proto')
    row['posterior_screen']['between_layout_fraction_indicator']=0.;(tmp_path/'case-00/category-0-LOCAL.json').write_text('{}')
    with pytest.raises(ValueError,match='bytes'):a.check_case(row,spec,metadata,'proto')

def test_missing_case_retains_unresolved_without_reconstructed_moments(monkeypatch,tmp_path):
    monkeypatch.setattr(a,'LOCAL',tmp_path);row=dict(index=0,case=a.run.design.worklist()[0],protocol_SHA256='proto',prefix={'x':'prefix'},status='unresolved',partial_artifacts=[])
    result=a.check_case(row,row['case'],dict(prefixes=[dict(record={'x':'prefix'})]),'proto');assert result['reference_count']==0 and result['utility_jobs']==0 and not result['rank_verified']

def test_mandatory_adaptive_and_finite_final_before_qualified_claim():
    value=dict(qualified=True,strict=dict(passed=True),diagnostics=dict(independent_adaptive_per_posterior_required=True,levels=[dict(outer=o,inner=i,status='computed',log_evidence=0.)for o,i in((512,512),(512,256),(256,512))]));assert a.reference_metadata(value)
    changed=copy.deepcopy(value);changed['diagnostics'].pop('independent_adaptive_per_posterior_required')
    with pytest.raises(ValueError,match='adaptive'):a.reference_metadata(changed)
    for change in(dict(status='unresolved',reason='Preserved numerical failure'),dict(log_evidence=float('nan'))):
        changed=copy.deepcopy(value);changed['diagnostics']['levels'][0].update(change)
        with pytest.raises(ValueError,match='finite final'):a.reference_metadata(changed)
    changed=copy.deepcopy(value);changed.update(qualified=False,strict=dict(passed=False));changed['diagnostics']['levels'][0].update(status='unresolved',reason='Preserved numerical failure');assert not a.reference_metadata(changed)

def test_qualified_outer_final_pair_coarse_state_required():
    value=dict(qualified=True,diagnostics=dict(independent_adaptive_per_posterior_required=True,levels=[dict(outer=o,inner=i,status='computed',log_evidence=0.)for o,i in((512,512),(512,256),(256,512))]));value['diagnostics']['levels'][2].update(status='unresolved',reason='Preserved final outer comparison failure')
    with pytest.raises(ValueError,match='final-pair'):a.reference_metadata(value)
