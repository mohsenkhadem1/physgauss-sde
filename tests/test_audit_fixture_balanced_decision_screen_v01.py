import copy
from pathlib import Path
import pytest
from experiments import audit_fixture_balanced_decision_screen_v01 as a
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
    spec=a.run.design.worklist()[0];directory=tmp_path/'case-00';directory.mkdir();row=dict(index=0,case=spec,protocol_SHA256='proto',prefix={'bound':'prefix'},status='qualified',qualified=True,categories=[],decision_levels=[],jobs=action_rows(),partial_artifacts=[]);metadata=dict(prefixes=[dict(record={'bound':'prefix'})]);old_directory=tmp_path/'old';old_directory.mkdir();old_witness=dict(index=0,categories=[],decisions=[]);origin=dict(old36_witnesses=[old_witness])
    for k in range(4):
        path=directory/f'category-{k}-LOCAL.json';a.run.write(path,dict(qualified=True,diagnostics=dict(independent_adaptive_per_posterior_required=True,levels=[dict(outer=o,inner=i,status='computed',log_evidence=0.)for o,i in((512,512),(512,256),(256,512),(512,128),(128,512))])));value=a.run.read(path);oldpath=old_directory/path.name;a.run.write(oldpath,value);oldrec=a.run.record(oldpath);old_witness['categories'].append(oldrec);value.update(old_reference=oldrec,unchanged_prefix=a.unchanged_reference(value['diagnostics'],value['diagnostics']));a.run.write(path,value);rec=a.run.record(path);row['categories'].append(dict(layout=k,qualified=True,record=rec));row['partial_artifacts'].append(rec)
    screen=dict(prior_variance=2e-10,between_layout_fraction=.2,maximum_layout_probability=.25,layout_TV=0.)
    for o,n in a.run.DECISION_LEVELS:
        path=directory/f'decision-{o}-{n}-LOCAL.json';a.run.write(path,dict(checks=dict(passed=True,layout_TV=0.),metadata=dict(layout_probability=[.25]*4,log_evidence=[0.]*4),posterior_screen=screen));value=a.run.read(path);oldpath=old_directory/path.name;a.run.write(oldpath,value);oldrec=a.run.record(oldpath);old_witness['decisions'].append(oldrec);value.update(old_decision=oldrec,unchanged_prefix=a.unchanged_decision(value['metadata'],value['metadata']));a.run.write(path,value);rec=a.run.record(path);row['decision_levels'].append(dict(outer=o,amplitude=n,checks=dict(passed=True,layout_TV=0.),unchanged_prefix=value['unchanged_prefix'],record=rec));row['partial_artifacts'].append(rec)
    row['ranking']=a.run.old.classify(row['jobs'],dict(outcome_relative=1e-4,latent_relative=1e-3));row['posterior_screen']=dict(screen,between_layout_fraction_indicator=0.)
    return row,spec,metadata,origin

def test_all144_witness_contract_and_independent48_arithmetic(monkeypatch,tmp_path):
    row,spec,metadata,origin=fake_case(monkeypatch,tmp_path);result=a.check_case(row,spec,metadata,'proto',origin);assert result==dict(reference_count=4,references_qualified=4,decision_count=5,utility_jobs=48,rank_verified=True)
    row['ranking']['actions'][0]['outer_error']=1.
    with pytest.raises(ValueError,match='recomputed'):a.check_case(row,spec,metadata,'proto',origin)

def test_changed_input_hash_or_between_indicator_refused(monkeypatch,tmp_path):
    row,spec,metadata,origin=fake_case(monkeypatch,tmp_path);row['posterior_screen']['between_layout_fraction_indicator']=.3
    with pytest.raises(ValueError,match='indicator'):a.check_case(row,spec,metadata,'proto',origin)
    row['posterior_screen']['between_layout_fraction_indicator']=0.;(tmp_path/'case-00/category-0-LOCAL.json').write_text('{}')
    with pytest.raises(ValueError,match='bytes'):a.check_case(row,spec,metadata,'proto',origin)

def test_missing_case_retains_unresolved_without_reconstructed_moments(monkeypatch,tmp_path):
    monkeypatch.setattr(a,'LOCAL',tmp_path);row=dict(index=0,case=a.run.design.worklist()[0],protocol_SHA256='proto',prefix={'x':'prefix'},status='unresolved',partial_artifacts=[])
    result=a.check_case(row,row['case'],dict(prefixes=[dict(record={'x':'prefix'})]),'proto',dict(old36_witnesses=[dict(index=0)]));assert result['reference_count']==0 and result['utility_jobs']==0 and not result['rank_verified']

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


def fake_builder(monkeypatch):
    monkeypatch.setattr(a,'bound',lambda rec:rec)
    old_models=[dict(record=dict(path=f'old-{k}',sha256=str(k)))for k in range(4)]
    p=dict(origin=dict(old36_prefixes=[dict(index=i,record=dict(path=f'prefix-{i}',sha256=str(i)))for i in range(36)],old36_models=old_models))
    reaction=dict(free_residual_N=1e-15,tolerance_N=1e-10,force_balance_error_N=1e-15,force_balance_tolerance_N=1e-8,moment_balance_error_Nm=1e-15,moment_balance_tolerance_Nm=1e-8,exact_clamp=True,passed=True)
    rows=[dict(index=i,load_ids=[s['id']for s in a.run.balanced.load_specs()[9:]],free_residual_l2_N=[1e-15]*2,exact_clamp_zeros=True,maximum_modal_parity_m=1e-16,equilibrium=[copy.deepcopy(reaction),copy.deepcopy(reaction)])for i in range(36)]
    models=[dict(layout=k,old_model=old_models[k]['record'],first12_invariance=dict(all_eigenvalues_byte_identical=True,first12_labels_identical=True,maximum_full_prior_response_difference_bound_m=1e-16))for k in range(4)]
    return dict(prefixes=copy.deepcopy(p['origin']['old36_prefixes']),old_negative_preserved=True,no_new_NH=True,new_held_equilibria=72,new_held_validations=rows,models=models,held_response_record=dict(path='hash-bound-only'),closed36_target_separation=dict(diagnostic_only_not_gate=True,not_assumed_equal=True)),p


def test_all72_new_held_witnesses_and_exact_old_prefix_binding(monkeypatch):
    value,p=fake_builder(monkeypatch);check=a.check_builder(value,p);assert check['new_held_equilibria_verified']==72 and check['held_balance_is_descriptive_only']
    value['prefixes'][0]['record']['sha256']='changed'
    with pytest.raises(ValueError,match='prefix'):a.check_builder(value,p)


def test_false_equilibrium_flags_missing_held_and_changed_first12_refused(monkeypatch):
    value,p=fake_builder(monkeypatch);value['new_held_validations'][0]['equilibrium'][1]['force_balance_error_N']=1.
    with pytest.raises(ValueError,match='equilibrium'):a.check_builder(value,p)
    value,p=fake_builder(monkeypatch);value['new_held_validations'].pop()
    with pytest.raises(ValueError,match='36'):a.check_builder(value,p)
    value,p=fake_builder(monkeypatch);value['models'][0]['first12_invariance']['maximum_full_prior_response_difference_bound_m']=1e-10
    with pytest.raises(ValueError,match='first12'):a.check_builder(value,p)


def test_root_reap_and_exact_five_owned_pid_membership_before_scientific_reads(monkeypatch):
    e=dict(controller_pid=10,builder=dict(pid=11),workers=[dict(pid=i)for i in(12,13,14)]);p=dict(source_SHA256={'s':'h'});closure=dict(root_exit_code=0,root_reaped=True,all_owned_PID_and_groups_absent=True,protocol_SHA256='proto',execution_SHA256='e',source_SHA256=p['source_SHA256'],owned_pids=[10,11,12,13,14]);seen=[]
    monkeypatch.setattr(a.run,'read',lambda _:closure);monkeypatch.setattr(a.run,'sha',lambda _:'e');monkeypatch.setattr(a.run,'record',lambda p:dict(path=str(p)));monkeypatch.setattr(a.run,'absent',lambda pid:seen.append(pid))
    a.root_closure(p,e,'proto');assert seen==[10,11,12,13,14]
    closure['root_reaped']=False
    with pytest.raises(ValueError,match='reaped'):a.root_closure(p,e,'proto')
    closure['root_reaped']=True;closure['owned_pids'][-1]=15
    with pytest.raises(ValueError,match='identities'):a.root_closure(p,e,'proto')


def test_unchanged_evidence_and_probabilities_recomputed_from_exact_old_records(monkeypatch,tmp_path):
    row,spec,metadata,origin=fake_case(monkeypatch,tmp_path)
    path=tmp_path/'case-00/category-0-LOCAL.json';value=a.run.read(path);value['diagnostics']['levels'][0]['log_evidence']=1e-4;a.run.write(path,value)
    rec=a.run.record(path);row['partial_artifacts'][0]=rec;row['categories'][0]['record']=rec
    with pytest.raises(ValueError,match='unchanged reference'):a.check_case(row,spec,metadata,'proto',origin)


def test_unchanged_decision_rejects_false_rho_witness_and_widened_gate():
    before=dict(log_evidence=[0.]*4,layout_probability=[.25]*4);after=copy.deepcopy(before);after['layout_probability']=[.251,.249,.25,.25]
    v=a.unchanged_decision(before,after);assert not v['passed'] and v['layout_probability_gate']==1e-10
    after=copy.deepcopy(before);after['log_evidence'][0]=1e-7;assert not a.unchanged_decision(before,after)['passed']
