"""Manufactured metadata and callbacks only; no actual observations/models."""
import json
from pathlib import Path
import numpy as np
import pytest
from experiments import run_layout_grouped_reference_cost_v01 as m

def test_unqualified_reference_cannot_enter_decision_phase():
    assert not m.reference_check(dict(qualified=False,diagnostics={}))['passed']

def test_original_negative_gate_refuses_incomplete_before_saved_prefix(tmp_path,monkeypatch):
    monkeypatch.setattr(m.old,'LOCAL',tmp_path);monkeypatch.setattr(m.old,'validate',lambda _:None)
    for name,value in [('execution.json',dict(completed=False)),('root-execution.json',{}),('independent-negative-review.json',{})]: (tmp_path/name).write_text(json.dumps(value))
    monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('No saved prefix before prior negative admission'))
    with pytest.raises(ValueError,match='Closed original'):m.origin_gate()

def test_worker_rootrelease_before_data_or_model(monkeypatch):
    monkeypatch.setattr(m,'validate',lambda _:(_ for _ in()).throw(ValueError('release missing')));monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('No saveddata before release'))
    with pytest.raises(ValueError,match='release missing'):m.worker('none')

def test_full_fixed_cost_plan_retained_on_timeout(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'guard',lambda :None);monkeypatch.setattr(m,'record',lambda p:dict(path=str(p),sha256=m.sha(p)));(tmp_path/'release.json').write_text('{}')
    work=[dict(outer=o,amplitude=n,action=0)for o,n in((512,32),(256,64),(512,64))];monkeypatch.setattr(m,'validate',lambda _:dict(utility_worklist=work,origin=dict(prefix_observation=dict(path='unopened'))))
    monkeypatch.setattr(m.np,'load',lambda *a,**k:(_ for _ in()).throw(TimeoutError('manufactured cap')))
    with pytest.raises(TimeoutError):m.worker('toy')
    state=json.loads((tmp_path/'execution.json').read_text());assert not state['completed']and not state['qualified']and len(state['prefix_categories'])==4 and len(state['all_utility_worklist'])==3 and state['all8_action_ranks']=='PENDING_NOT_ADMITTED'

def test_outcome_gate_tightness_unchanged_no_clipping():
    coarse=dict(expected_target_variance=1e-9);fine=dict(expected_target_variance=2e-9,omitted_risk_bound=1e-24,prior_target_variance=3e-9);assert not m.risk_check(coarse,fine)['passed']
    coarse['expected_target_variance']=fine['expected_target_variance'];assert m.risk_check(coarse,fine)['passed']

def test_decision_reference48moments_mismatch_blocks():
    meta=dict(log_evidence=[0.]*4,category_moments=[dict(mean=np.zeros(48),SD=np.ones(48)*1e-5)for _ in range(4)]);finals=[dict(log_evidence=0.,predictive_mean=np.zeros(48),predictive_SD=np.ones(48)*1e-5)for _ in range(4)]
    assert m.grouped.reference_moment_checks(meta,finals)['passed'];finals[2]['predictive_mean'][4]=3e-6;assert not m.grouped.reference_moment_checks(meta,finals)['passed']


def test_protocol_JSON_roundtrip_and_worklist_tamper_before_data(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'LOCAL',tmp_path/'pilot');monkeypatch.setattr(m.old,'ROOT',tmp_path)
    monkeypatch.setattr(m,'record',lambda path:dict(path=str(Path(path).relative_to(tmp_path)),sha256=m.sha(path)))
    source=tmp_path/'source.py';source.write_text('x=1');external=tmp_path/'external.json';external.write_text(json.dumps(dict(controller_pid=123,child_pid=124)))
    origin=dict(external=m.record(external));monkeypatch.setattr(m,'origin_gate',lambda :origin);monkeypatch.setattr(m,'sources',lambda :{'source.py':m.sha(source)});monkeypatch.setattr(m.old.mechanical_screen.shared.old,'absent',lambda _:None)
    protocol=m.prepare();p=json.loads((m.LOCAL/'protocol.json').read_text());release=dict(approved=True,protocol_SHA256=protocol['sha256'],source_SHA256=p['source_SHA256'],origin=origin,external_watchdog_600s_4GiB=True);m.write(m.LOCAL/'release.json',release)
    assert len(m.validate(protocol['sha256'])['utility_worklist'])==3
    p['utility_worklist'].pop();m.write(m.LOCAL/'protocol.json',p)
    with pytest.raises(ValueError,match='sources/config'):m.validate(m.sha(m.LOCAL/'protocol.json'))


def test_actual_absolute2um_gate_cannot_be_replaced_by_relative_pass(monkeypatch):
    monkeypatch.setattr(m.old.mechanical_screen.shared,'engine_gates',lambda _:dict(passed=True))
    row=dict(passed=True,maximum_predictive_mean_difference_m=3e-6,maximum_predictive_SD_difference_m=1e-8,coarse=256,fine=512)
    result=dict(qualified=True,diagnostics=dict(successive={axis:[row]for axis in('inner','outer')}))
    check=m.reference_check(result);assert not check['passed']and not check['strict_absolute_checks']['passed']

def test_external_controller_release_before_spawn(monkeypatch):
    monkeypatch.setattr(m,'validate',lambda _:(_ for _ in()).throw(ValueError('release missing')));monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:pytest.fail('No external child without release'))
    with pytest.raises(ValueError,match='release missing'):m.run('none')


def test_external_child_failure_cannot_publish_qualified_true(tmp_path,monkeypatch):
    import os
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'validate',lambda _:{});monkeypatch.setattr(m.os,'sched_getaffinity',lambda _:{18})
    for key,value in [('CUDA_VISIBLE_DEVICES',''),('OMP_NUM_THREADS','1'),('OPENBLAS_NUM_THREADS','1'),('MKL_NUM_THREADS','1')]:monkeypatch.setenv(key,value)
    monkeypatch.setattr(m,'record',lambda path:dict(path=str(path),sha256=m.sha(path)))
    m.write(tmp_path/'execution.json',dict(completed=True,qualified=True))
    class FailedChild:
        pid=123
        def poll(self):return 1
        def wait(self):return 1
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:FailedChild());monkeypatch.setattr(m.old.mechanical_screen.shared.old.cost,'cleanup',lambda _:1)
    result=m.run('toy');assert not result['completed']and not result['qualified']and result['reaped']and result['exit_code']==1
