"""Manufactured finite recovery controls; no saved campaign arrays are read."""
import copy
from pathlib import Path
import numpy as np
import pytest
from experiments import run_matched_finite_poke_control_v02 as m
from experiments import run_matched_finite_poke_control_v01 as old

def good_choice():
    a=np.zeros((8,60));a[:,12:]=np.arange(8)[:,None]*1e-5
    return old.model(a).choose(np.zeros(4))
def step(x,n):
    for _ in range(n):x=np.nextafter(x,np.inf)
    return x

@pytest.mark.parametrize('prior',[0.,np.nextafter(0.,1.),np.finfo(float).tiny,2.**-30,1e-9,1.,2.])
def test_eight_versus_nine_ULPs_without_physical_absolute_halo(prior):
    assert m.prior_witness_ULPs(prior,step(prior,8))==8
    assert m.prior_witness_close(prior,step(prior,8))and not m.prior_witness_close(prior,step(prior,9))
    assert m.prior_witness_close(prior,prior)
    if prior>=np.finfo(float).tiny:assert not m.prior_witness_close(prior,prior*(1+1e-8))

def test_zero_sign_nonfinite_negative_and_bool_failclosed():
    assert m.prior_witness_ULPs(0.,-0.)==0
    for bad in(-np.nextafter(0.,1.),-1.,np.nan,np.inf,-np.inf,True,'0'):
        assert m.prior_witness_ULPs(bad,0.)is None and not m.prior_witness_close(bad,0.)

def test_original_immutable_choices_accept_ULP_drift_but_not_changed_risk_or_tie():
    choice=good_choice();original=copy.deepcopy(choice)
    assert m.choice_check(choice)==old.choice_check(choice)
    for row in choice['actions']:
        for level in row['levels']:level['prior_variance_m2']=step(choice['prior_variance_m2'],1)
    before=copy.deepcopy(choice)
    assert m.choice_check(choice)is True and choice==before
    with pytest.raises(ValueError):old.choice_check(choice)
    bad=copy.deepcopy(choice);bad['actions'][0]['levels'][0]['prior_variance_m2']=step(choice['prior_variance_m2'],9)
    with pytest.raises(ValueError):m.choice_check(bad)
    bad=copy.deepcopy(choice);bad['actions'][0]['refinement_indicator_m2']+=1e-20
    with pytest.raises(ValueError,match='flags'):m.choice_check(bad)
    bad=copy.deepcopy(choice);bad['selected_action']=7
    with pytest.raises(ValueError,match='tie'):m.choice_check(bad)
    assert original['actions'][0]['levels'][0]['prior_variance_m2']!=choice['actions'][0]['levels'][0]['prior_variance_m2']

def metadata_fixture(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m.old,'LOCAL',tmp_path/'old');monkeypatch.setattr(m.continuous,'LOCAL',tmp_path/'parent')
    monkeypatch.setattr(m,'record',lambda p:dict(path=str(Path(p).relative_to(tmp_path)),sha256=m.sha(p)))
    monkeypatch.setattr(m.old,'sources',lambda:{'original':'source'})
    for p in(m.old.LOCAL/'release.json',m.continuous.LOCAL/'protocol.json',m.continuous.LOCAL/'release.json',m.continuous.LOCAL/'execution-choices.json',m.continuous.LOCAL/'root-closure-choices.json',m.continuous.LOCAL/'inputs.json',m.old.LOCAL/'catalogs-LOCAL.npz',m.old.LOCAL/'root-closure-choose.json'):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    m.write(m.old.LOCAL/'release.json',dict(approved=True,parent_protocol_SHA256='p',source_SHA256=m.old.sources(),resources=m.old.LIMITS))
    inputs=dict(prefixes=[dict(index=i,record={'input':i},observation_fingerprint=str(i))for i in range(32)])
    seals=[dict(index=i)for i in range(32)];monkeypatch.setattr(m.continuous,'validate',lambda p:{});monkeypatch.setattr(m.continuous,'all_seals_gate',lambda p:seals);monkeypatch.setattr(m.continuous,'inputs_gate',lambda p:inputs)
    captures=[]
    for i in range(8):
        for label in(s['id']for s in m.continuous.design.physics.load_specs()):captures.append(dict(state=i,label=label,capture={'capture':f'{i}-{label}'}))
    monkeypatch.setattr(m.continuous,'capture',lambda p,i,label:next(v['capture']for v in captures if v['state']==i and v['label']==label));monkeypatch.setattr(m.continuous,'verify',lambda x:None)
    closed=dict(pid=123,stage={'stage':'old'},preapproval={'preapproval':'old'})
    monkeypatch.setattr(m.old_audit,'closure',lambda *a:closed)
    choice=good_choice();choice['actions'][0]['levels'][0]['prior_variance_m2']=step(choice['prior_variance_m2'],1)
    value=dict(completed=True,pid=123,source_SHA256=m.old.sources(),parent_protocol=m.record(m.continuous.LOCAL/'protocol.json'),parent_release=m.record(m.continuous.LOCAL/'release.json'),root_release=m.record(m.old.LOCAL/'release.json'),continuous32_choice_seals=seals,continuous_choice_execution=m.record(m.continuous.LOCAL/'execution-choices.json'),continuous_choice_root_closure=m.record(m.continuous.LOCAL/'root-closure-choices.json'),inputs=m.record(m.continuous.LOCAL/'inputs.json'),no_trial_identity_or_mapped_outcome_used=True,candidate_labels=list(m.CANDIDATE_LABELS),fixed_action=5,fixed_action_label=m.CANDIDATE_LABELS[5],wall_s=1.,source_captures=captures,catalog=m.record(m.old.LOCAL/'catalogs-LOCAL.npz'),choices=[dict(model=name,index=i,choice=copy.deepcopy(choice),prefix_input=inputs['prefixes'][i]['record'],observation_fingerprint=str(i),fixed_action=5)for name in m.MODELS for i in range(32)])
    m.write(m.old.LOCAL/'finite-choice-seals.json',value);return value

def test_all64_original_metadata_and_newverifier_before_any_array(monkeypatch,tmp_path):
    value=metadata_fixture(monkeypatch,tmp_path);before=m.sha(m.old.LOCAL/'finite-choice-seals.json');monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('No NPZ within original metadata gate'))
    *_,origin=m.original_choices_gate('p');assert origin['maximum_prior_ULP_distance']==1 and origin['nonidentical_prior_witnesses']==64
    assert before==m.sha(m.old.LOCAL/'finite-choice-seals.json')
    value['choices'][-1]['choice']['actions'][0]['levels'][0]['prior_variance_m2']=step(value['choices'][-1]['choice']['prior_variance_m2'],9);m.write(m.old.LOCAL/'finite-choice-seals.json',value)
    with pytest.raises(ValueError):m.original_choices_gate('p')

def test_recovery_evaluate512_unchanged_single_pokes_and_original_bytes(monkeypatch,tmp_path):
    from tests.test_matched_finite_poke_control_v01 import shared_files,catalogs
    # Reuse manufactured temporary observations only, never actual campaign files.
    local,parent,inputs=shared_files(monkeypatch,tmp_path);original=tmp_path/'original';original.mkdir();(original/'finite-choice-seals.json').write_text('{"unchanged":true}')
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'LOCAL',local);monkeypatch.setattr(m.old,'LOCAL',original);monkeypatch.setattr(m,'record',lambda p:dict(path=str(Path(p).relative_to(tmp_path)),sha256=m.sha(p)));monkeypatch.setattr(m,'guard',lambda start:None);monkeypatch.setattr(m,'sources',lambda:{'new':'producer'})
    arrays=catalogs();catalog=original/'catalog.npz';np.savez_compressed(catalog,**arrays);choice=good_choice()
    sealed=dict(catalog=m.record(catalog),source_SHA256={'original':'producer'},wall_s=0.,choices=[dict(model=name,index=i,choice=copy.deepcopy(choice),observation_fingerprint=inputs['prefixes'][i]['observation_fingerprint'])for name in m.MODELS for i in range(32)])
    p=dict(trials=m.continuous.design.trials());monkeypatch.setattr(m,'admission',lambda *a:(p,inputs,sealed,{'old':'bound'}))
    originals=[original/'finite-choice-seals.json',catalog,parent/'noise.npz'];hashes={p:m.sha(p)for p in originals}
    m.evaluate('p');value=m.read(local/'evaluation-results-LOCAL.json');assert len(value['rows'])==64 and value['source_SHA256']=={'new':'producer'}and value['original_chooser_source_SHA256']=={'original':'producer'}
    assert len(list((local/'posteriors-LOCAL').glob('*.npz')))==64 and not value['summary']['comparison_blocked']
    assert all(len(r['counterfactuals'])==8 and all(v['fresh_prefix_plus_ONE_scalar']for v in r['counterfactuals'])for r in value['rows'])
    assert all(m.sha(p)==h for p,h in hashes.items())and all(r['choice_seal']==m.record(original/'finite-choice-seals.json')for r in value['rows'])
    with pytest.raises(FileExistsError):m.evaluate('p')

def test_release_failure_before_outcome_deserialization(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'guard',lambda start:None);monkeypatch.setattr(m,'release_gate',lambda p:(_ for _ in()).throw(ValueError('No ROOTv02 approval')));monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('No recovery arrays before release'))
    with pytest.raises(ValueError):m.evaluate('p')
