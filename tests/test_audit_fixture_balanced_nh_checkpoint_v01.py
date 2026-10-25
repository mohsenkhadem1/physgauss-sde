import copy
import numpy as np
import pytest
from experiments import audit_fixture_balanced_nh_checkpoint_v01 as m


def fixture(monkeypatch,tmp_path):
    monkeypatch.setattr(m.stage,'LOCAL',tmp_path);monkeypatch.setattr(m.stage,'ROOT',tmp_path)
    for name in(m.SOURCE,m.TEST):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic audit procedure')
    monkeypatch.setattr(m.stage,'record',lambda p:dict(path=str(p),sha256=m.stage.sha(p)))
    for name in('protocol.json','release.json'):(tmp_path/name).write_text('{}')
    context=m.stage.groups()[0];p=dict(plan=dict(groups=[context]),source_SHA256={'frozen-physical':'pin'})
    monkeypatch.setattr(m.stage,'validate',lambda digest:p);monkeypatch.setattr(m.frozen,'guard',lambda start:None)
    rows=[dict(n=n,label=l,capture=dict(path=f'synthetic-n{n}-{l}',sha256='bound'))for n in m.stage.LEVELS for l in m.stage.LOADS]
    spatial={l:dict(spatial_development_passed=True)for l in m.stage.LOADS}
    flags=dict(all33_retained=True,all11_reference_valid=True,all_fields_excitation_goal_met=False,references={l:dict(reference_valid=True,all_meshes_excitation_goal_met=False)for l in m.stage.LOADS},bank_admitted=False)
    group=dict(group=context,rows=rows,meshes={str(n):dict(surface={'known':'surface'})for n in m.stage.LEVELS},spatial=spatial,**flags)
    item=dict(index=0,result=dict(path='bound-group',sha256='bound'),value=group);seen=[]
    def barrier(*args):seen.append('all33-bytebarrier');return dict(execution={'hash':'phase'},root_closure={'hash':'root'}),[item]
    monkeypatch.setattr(m.frozen,'phase_metadata',barrier)
    def capture(row):
        assert seen and seen[0]=='all33-bytebarrier';seen.append((row['n'],row['label']));return np.zeros((1,3)),dict(synthetic='deformation')
    monkeypatch.setattr(m.frozen.field_audit,'check_capture',capture);monkeypatch.setattr(m.frozen,'field_status',lambda row,d:dict(physical_passed=True,excitation_goal_met=False,legacy_load_eligible=False,legacy_weak_prefix_control=False))
    monkeypatch.setattr(m.frozen,'spatial_results',lambda fields:spatial);monkeypatch.setattr(m.frozen,'group_classification',lambda *args:copy.deepcopy(flags))
    return p,item,flags,seen


def test_all33_guarded_then_compact_positive_reference_with_weak_excitation(monkeypatch,tmp_path):
    p,item,flags,seen=fixture(monkeypatch,tmp_path);rec=m.audit('protocol',m.stage.sha(tmp_path/m.SOURCE),m.stage.sha(tmp_path/m.TEST))
    value=m.frozen.read(tmp_path/'checkpoint-independent-review.json');assert value['audit_passed']and value['all11_reference_valid']and not value['all_fields_excitation_goal_met']and len(value['captures'])==33 and len(seen)==34
    assert value['source_physical_SHA256']==p['source_SHA256']and not value['bank_admitted']and 'no_PDE_factor_RNG_or_old_source_mutation'in value['checks']
    with pytest.raises(FileExistsError):m.audit('protocol',m.stage.sha(tmp_path/m.SOURCE),m.stage.sha(tmp_path/m.TEST))


def test_incomplete_or_unreaped_barrier_never_loads_fields(monkeypatch,tmp_path):
    fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(m.frozen,'phase_metadata',lambda *a:(_ for _ in()).throw(ValueError('rootreap/capturehash failed')))
    monkeypatch.setattr(m.frozen.field_audit,'check_capture',lambda *a:pytest.fail('No arrays before complete barrier'))
    with pytest.raises(ValueError,match='rootreap'):m.audit('protocol',m.stage.sha(tmp_path/m.SOURCE),m.stage.sha(tmp_path/m.TEST))
    assert not(tmp_path/'checkpoint-independent-review.json').exists()


def test_own_source_precommit_and_all11_reference_tamper_refused(monkeypatch,tmp_path):
    _,item,flags,_=fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='prepinned'):m.audit('protocol','wrong',m.stage.sha(tmp_path/m.TEST))
    item['value']['all11_reference_valid']=False
    with pytest.raises(ValueError,match='classification'):m.audit('protocol',m.stage.sha(tmp_path/m.SOURCE),m.stage.sha(tmp_path/m.TEST))
    assert not(tmp_path/'checkpoint-independent-review.json').exists()


def test_honest_complete_negative_does_not_drop_fields_or_claim_referencevalid(monkeypatch,tmp_path):
    _,item,flags,seen=fixture(monkeypatch,tmp_path);flags['all11_reference_valid']=False;item['value']['all11_reference_valid']=False
    rec=m.audit('protocol',m.stage.sha(tmp_path/m.SOURCE),m.stage.sha(tmp_path/m.TEST));value=m.frozen.read(tmp_path/'checkpoint-independent-review.json')
    assert value['audit_passed']and value['all33_retained']and not value['all11_reference_valid']and len(value['captures'])==33
