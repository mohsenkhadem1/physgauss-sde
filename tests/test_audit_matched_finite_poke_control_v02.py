import copy,sys
from pathlib import Path
import numpy as np
import pytest
from experiments import audit_matched_finite_poke_control_v02 as a
from tests import test_audit_matched_finite_poke_control_v01 as manufactured

@pytest.fixture
def use_manufactured(monkeypatch):
    monkeypatch.setattr(manufactured,'a',a)

def full_fixture(monkeypatch,tmp_path,negative=False):
    monkeypatch.setattr(manufactured,'a',a)
    local,result,arrays=manufactured.make_full_result(monkeypatch,tmp_path,negative)
    monkeypatch.setattr(a.run.old,'LOCAL',local)
    return local,result,arrays

@pytest.mark.parametrize('negative',[False,True])
def test_full512_independent_updates_and_originalchoice_newaudit_lineage(monkeypatch,tmp_path,negative):
    local,result,_=full_fixture(monkeypatch,tmp_path,negative)
    value=a.audit('p');assert value['audit_passed']and value['independently_recomputed_posterior_updates']==512 and value['summary']['comparison_blocked']is negative
    assert value['sealed_choices_reused_without_modification']and value['verifier_rule']==a.run.PRIOR_RULE
    assert value['finite_choices']==a.run.record(a.run.old.LOCAL/'finite-choice-seals.json')
    assert value['original_chooser_source_SHA256']==a.run.old.sources()

def test_independent_posterior_and_scalar_metrics_reject_tamper(monkeypatch,tmp_path):
    local,result,_=full_fixture(monkeypatch,tmp_path);row=result['rows'][0];path=a.bound(row['posterior_archive'])
    with np.load(path)as z:values={k:z[k].copy()for k in z.files}
    values['weights'][0]=np.roll(values['weights'][0],1);np.savez(path,**values);row['posterior_archive']=a.run.record(path)
    with pytest.raises(ValueError,match='numerical'):a.audit('p')

def closure_fixture(monkeypatch,tmp_path):
    local=tmp_path/'new';op=tmp_path/'operator';local.mkdir();op.mkdir();monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'LOCAL',local);monkeypatch.setattr(a,'OPERATOR',op)
    monkeypatch.setattr(a.run,'record',lambda p:dict(path=str(Path(p).relative_to(tmp_path)),sha256=a.run.sha(p)))
    for n in('audit.py','test.py','new-producer.py'):(tmp_path/n).write_text(n)
    pins={n:a.run.sha(tmp_path/n)for n in('audit.py','test.py','new-producer.py')};monkeypatch.setattr(a,'sources',lambda:{n:pins[n]for n in('audit.py','test.py')});monkeypatch.setattr(a.run,'sources',lambda:{'new-producer.py':pins['new-producer.py']})
    approval=op/'root-preapproval.json';a.run.write(approval,dict(approved=True,schema='noisy-poke-recovery-operator-preapproval-v02',source_SHA256=pins,stage_caps=dict(finite_evaluate=[17,1800,4*1024**3],finite_audit=[16,1800,4*1024**3])))
    result=local/'evaluation-results-LOCAL.json';result.write_text('{}');(local/'compact-summary.json').write_text('{}');r=a.run.record(result);identity=dict(pid=7654321,PGID=7654321,PPID=100,starttime=123,state='R',command='new-writer')
    stage=op/'stages/finite_evaluate/execution.json';value=dict(phase='finite_evaluate',parent_protocol_SHA256='p',completed=True,reaped=True,exit_code=0,cleanup_exit_code=0,child_identity=identity,source_SHA256=pins,operator_preapproval=a.run.record(approval),resources=dict(CPU=17,wall_s=1800,RSS_bytes=4*1024**3),command=['taskset','-c','17',sys.executable,'-u','-m',a.run.__name__,'evaluate','--parent-sha','p'],outputs=[r,a.run.record(local/'compact-summary.json')],result=r,observed_owned=[identity],wall_s=1.,peak_RSS_bytes=1024)
    a.run.write(stage,value);closure=local/'root-closure-evaluate.json';v=dict(root_reaped=True,root_exit_code=0,root_owned_supervisor=True,external_subprocess_wait=True,all_owned_PID_and_groups_absent=True,parent_protocol_SHA256='p',source_SHA256=a.run.sources(),owned_pids=[identity['pid']],operator_preapproval=a.run.record(approval),operator_stage_execution=a.run.record(stage),result=r);a.run.write(closure,v);monkeypatch.setattr(a.c.base,'absent',lambda pid:None)
    return r,stage,value,closure,v

def test_new_evaluator_identity_command_and_rootclosure_not_oldchooser(monkeypatch,tmp_path):
    r,stage,value,path,v=closure_fixture(monkeypatch,tmp_path);assert a.closure('p',r)['pid']==7654321
    value['command'][6]='experiments.run_matched_finite_poke_control_v01';a.run.write(stage,value);v['operator_stage_execution']=a.run.record(stage);a.run.write(path,v)
    with pytest.raises(ValueError,match='command'):a.closure('p',r)

def test_unreaped_newclosure_refuses_before_any_NPZ(monkeypatch,tmp_path):
    r,stage,value,path,v=closure_fixture(monkeypatch,tmp_path);v['root_reaped']=False;a.run.write(path,v);monkeypatch.setattr(a.np,'load',lambda *a,**k:pytest.fail('No arrays before closure'))
    with pytest.raises(ValueError,match='root closure'):a.closure('p',r)

def test_all64_newarchive_hashes_before_any_deserialization(monkeypatch,tmp_path):
    local=tmp_path/'new';old=tmp_path/'old';local.mkdir();old.mkdir();monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'LOCAL',local);monkeypatch.setattr(a.run,'LOCAL',local);monkeypatch.setattr(a.run.old,'LOCAL',old)
    monkeypatch.setattr(a.run,'record',lambda p:dict(path=str(Path(p).relative_to(tmp_path)),sha256=a.run.sha(p)));monkeypatch.setattr(a.run,'sources',lambda:{'new':'producer'});monkeypatch.setattr(a.run.old,'sources',lambda:{'old':'chooser'})
    for path in(old/'finite-choice-seals.json',a.c.LOCAL/'protocol.json'):
        if path==a.c.LOCAL/'protocol.json':continue
        path.write_text('{}')
    parent=tmp_path/'parent';parent.mkdir();(parent/'protocol.json').write_text('{}');monkeypatch.setattr(a.c,'LOCAL',parent)
    approval=tmp_path/'approval.json';approval.write_text('{}');audit_rec={'newaudit':'bound'};origin=dict(original_choose_closure={'closed':'old'},original_choose_stage={'stage':'old'},original_operator_preapproval={'approval':'old'});seal=dict(wall_s=1.,source_captures=[],catalog={'catalog':'old'})
    release=dict(origin=origin,continuous_audit=audit_rec,root_owned_operator_preapproval=a.run.record(approval));monkeypatch.setattr(a.run,'release_gate',lambda p:release);monkeypatch.setattr(a.run,'original_choices_gate',lambda p:({}, {},seal,origin));monkeypatch.setattr(a.run,'continuous_admission',lambda p:audit_rec);monkeypatch.setattr(a,'closure',lambda *args:dict(preapproval=a.run.record(approval)));monkeypatch.setattr(a.c,'verify',lambda *a:None)
    rows=[]
    for name in a.run.MODELS:
        for i in range(32):
            path=local/'posteriors-LOCAL'/f'{name}-{i:02d}.npz';path.parent.mkdir(exist_ok=True);path.write_text('not NPZ');rows.append(dict(model=name,index=i,posterior_archive=a.run.record(path),choice_seal=a.run.record(old/'finite-choice-seals.json')))
    result=dict(completed=True,source_SHA256=a.run.sources(),original_chooser_source_SHA256=a.run.old.sources(),recovery_origin=origin,verifier_rule=a.run.PRIOR_RULE,parent_protocol=a.run.record(parent/'protocol.json'),finite_choices=a.run.record(old/'finite-choice-seals.json'),rows=rows,wall_s=1.,combined_owned_wall_s=2.);a.run.write(local/'evaluation-results-LOCAL.json',result)
    monkeypatch.setattr(a,'bound',lambda rec:(tmp_path/rec['path'])if'path'in rec else tmp_path)
    # Real bound for archive entries rejects the last corrupt commitment.
    def bound(rec):
        if'path'not in rec:return tmp_path
        path=tmp_path/rec['path'];a.need(a.run.record(path)==rec,'Exact archive bytes');return path
    monkeypatch.setattr(a,'bound',bound);monkeypatch.setattr(a.np,'load',lambda *a,**k:pytest.fail('No NPZ within metadata barrier'))
    assert len(a.metadata_barrier('p')[3]['rows'])==64
    (local/'posteriors-LOCAL/linear-31.npz').write_text('tampered')
    with pytest.raises(ValueError):a.metadata_barrier('p')
