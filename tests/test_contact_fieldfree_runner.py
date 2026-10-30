import json
from pathlib import Path
import numpy as np
import pytest
from scripts.contact_benchmark_v12 import fieldfree_runner as runner
from scripts.contact_benchmark_v12 import model_selection_jobs as jobs


def fixture(dataset='SoftVTBench'):
    nodes=np.array([[0,0,0],[.04,0,0],[0,.04,0],[0,0,.04]],float)
    return dict(name='fixture',dataset=dataset,nodes=nodes,train_end=5,attachment_weights=np.ones((1,4)),observed=np.broadcast_to(nodes,(5,4,3)).copy(),evaluation={'points':np.full((20,4,3),1e99)})


@pytest.mark.parametrize('dataset',['SoftVTBench','PhysTwin'])
def test_runner_refits_only_fieldoff_with_isolated_payload_and_original_budget(tmp_path,monkeypatch,dataset):
    data=fixture(dataset);calls=[];monkeypatch.setattr(runner,'OUTPUT',tmp_path/'outputs');monkeypatch.setattr(runner,'check_deadline',lambda:None);monkeypatch.setattr(runner,'check_freeze',lambda:None)
    freeze=tmp_path/'freeze.json';freeze.write_text('{}');protocol=tmp_path/'protocol.json';protocol.write_text('{}');monkeypatch.setattr(runner,'FREEZE',freeze);monkeypatch.setattr(runner,'PROTOCOL',protocol)
    def fake_load(spec,validation):assert validation is True;return data
    monkeypatch.setattr(runner,'load',fake_load)
    def fake_fit(payload,out,**kwargs):
        assert 'evaluation' not in payload
        calls.append(kwargs)
        n=kwargs['materials'];theta=np.r_[[-4.,2.,5.],np.zeros(n),0.,1.];frames=np.arange(5)
        np.savez_compressed(out/'best.npz',theta=theta,nodes=payload['nodes'],selected_frames=frames)
        (out/'fit.json').write_text(json.dumps(dict(fit_end=5,best=dict(theta=theta.tolist()),**runner.settings(payload))))
        return theta
    monkeypatch.setattr(runner.oldfit,'fit',fake_fit)
    original_minimize=runner.oldfit.minimize
    def fake_rollout(payload,theta,settings,device):
        assert 'evaluation' not in payload and device=='cpu' and settings==runner.settings(payload)
        return np.broadcast_to(payload['nodes'],(7,4,3)).copy(),dict(failed_frames=[],diagnostics=[])
    monkeypatch.setattr(runner,'rollout',fake_rollout)
    runner.run({'case':'fixture'},validation=True)
    assert runner.oldfit.minimize is original_minimize
    assert len(calls)==1 and calls[0]['iterations']==100 and calls[0]['frames']==36
    assert calls[0]['field'] is False and calls[0]['contact'] is True
    assert calls[0]['materials']==(6 if dataset=='PhysTwin' else 1)
    assert calls[0]['nonconvex']==(dataset=='PhysTwin')
    result=json.loads((tmp_path/'outputs/validation/fixture/complete.json').read_text())
    assert result['family']=='field_free' and result['fit_end']==5
    runner.run({'case':'fixture'},validation=True)
    assert len(calls)==1,'Completed result should validate, not refit'


def test_runner_rejects_gpu_and_requires_explicit_incomplete_checkpoint_continuation(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='CPU-only'):runner.run({},device='cuda:0')
    data=fixture();monkeypatch.setattr(runner,'OUTPUT',tmp_path);monkeypatch.setattr(runner,'check_deadline',lambda:None);monkeypatch.setattr(runner,'check_freeze',lambda:None);monkeypatch.setattr(runner,'load',lambda *a:data)
    out=tmp_path/'full/fixture';out.mkdir(parents=True);np.savez_compressed(out/'best.npz',theta=np.zeros(6))
    with pytest.raises(ValueError,match='explicit timeout'):runner.run({})


def test_exact_existing_protocol_cases_form296_independent_cpu_jobs():
    planned=jobs.planned_jobs()
    assert len(planned)==296 and [j['index'] for j in planned]==list(range(2000,2296))
    assert sum(j['validation'] for j in planned)==148
    assert all(j['command'][1]=='scripts/contact_benchmark_v12/fieldfree_runner.py' for j in planned)
    for j in planned:
        assert j['command'][j['command'].index('--device')+1]=='cpu'
        assert ('--validation' in j['command'])==j['validation']


def test_claim_lock_prevents_duplicate_claim_and_separates_timeout_repair(tmp_path,monkeypatch):
    monkeypatch.setattr(jobs,'QUEUE',tmp_path/'queue');jobs.QUEUE.mkdir();monkeypatch.setattr(jobs,'LOCK',tmp_path/'lock');monkeypatch.setattr(jobs,'check_freeze',lambda:None)
    jobs.write_json(jobs.QUEUE/'2000.json',dict(index=2000,state='pending'))
    first=jobs.claim('worker-a');assert first[1]['state']=='running'
    assert jobs.claim('worker-b') is None
    value=first[1];value.update(state='failed',returncode=124);jobs.write_json(first[0],value)
    assert jobs.claim('worker-c') is None
    repaired=jobs.claim('repair',True);assert repaired[1]['state']=='repairing'
    assert jobs.claim('other-repair',True) is None


def test_terminal_failure_is_explicit_and_never_fabricates_fit(tmp_path,monkeypatch):
    data=fixture();data['times']=np.arange(7,dtype=float)
    monkeypatch.setattr(runner,'OUTPUT',tmp_path/'outputs');monkeypatch.setattr(runner,'load',lambda *a:data)
    monkeypatch.setattr(runner,'check_deadline',lambda:None);monkeypatch.setattr(runner,'check_freeze',lambda:None)
    for key in ['FREEZE','PROTOCOL']:
        path=tmp_path/(key+'.json');path.write_text('{}');monkeypatch.setattr(runner,key,path)
    record=runner.terminal_failure({},False,'Numerical fit failure',2000)
    out=runner.OUTPUT/'full/fixture'
    assert record['terminal_failure'] and record['failed_frames']==7
    assert not (out/'fit.json').exists() and not (out/'best.npz').exists()
    with np.load(out/'prediction.npz') as saved:
        assert saved['predicted'].shape==(7,4,3) and np.isnan(saved['predicted']).all()
    monkeypatch.setattr(runner,'validate_checkpoint',lambda *a:pytest.fail('Terminal failure is not a fitted checkpoint'))
    runner.run({})
    complete=json.loads((out/'complete.json').read_text());assert complete['job_index']==2000


def test_terminal_failure_preserves_completed_candidate(tmp_path,monkeypatch):
    data=fixture();monkeypatch.setattr(runner,'OUTPUT',tmp_path);monkeypatch.setattr(runner,'load',lambda *a:data)
    out=tmp_path/'full/fixture';out.mkdir(parents=True);(out/'complete.json').write_text('{"terminal_failure":false}')
    result=runner.terminal_failure({},False,'late process exit')
    assert result['preserved_existing_complete'] and not (out/'terminal-failure.json').exists()
