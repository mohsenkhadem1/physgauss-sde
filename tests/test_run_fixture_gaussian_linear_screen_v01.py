import json,os,signal,subprocess,sys
import pytest
from experiments import run_fixture_gaussian_linear_screen_v01 as m


def test_fixed_protocol_keeps_entire_queue_and_new_resource_ownership():
    p=m.plan();assert p['states']==72 and p['load_responses']==792
    assert p['first_paired_timing_checkpoint_indices']==[0,1] and p['full_queue_unconditional']
    assert m.RESOURCES['worker_CPU']==14 and m.RESOURCES['controller_CPU']==15
    assert m.RESOURCES['wall_cap_s']==900 and m.RESOURCES['aggregate_RSS_cap_bytes']==12*1024**3
    assert 'tests/test_fixture_gaussian_linear_screen_v01.py' in m.sources()


def test_live_origin_stops_at_metadata_before_opening_result_or_any_arrays(tmp_path,monkeypatch):
    monkeypatch.setattr(m.origin,'LOCAL',tmp_path);monkeypatch.setattr(m.origin,'validate',lambda digest:{})
    (tmp_path/'protocol.json').write_text('{}');(tmp_path/'execution.json').write_text(json.dumps(dict(terminal=False,completed=False)))
    with pytest.raises(ValueError,match='Closed'):m.origin_gate()
    assert not (tmp_path/'result-LOCAL.json').exists()


def test_release_and_fixed72_plan_tamper_fail_before_new_mechanics(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'sources',lambda:{'s':'h'});monkeypatch.setattr(m,'origin_gate',lambda:{'closed':True})
    p=dict(plan=m.plan(),source_SHA256={'s':'h'},origin={'closed':True})
    (tmp_path/'protocol.json').write_text(json.dumps(p));digest=m.sha(tmp_path/'protocol.json')
    (tmp_path/'release.json').write_text(json.dumps(dict(approved=False)))
    with pytest.raises(ValueError,match='rootrelease'):m.validate(digest)
    (tmp_path/'release.json').write_text(json.dumps(dict(approved=True,protocol_SHA256=digest,source_SHA256={'s':'h'})))
    assert m.validate(digest)==p
    p['plan']['worklist'].pop();(tmp_path/'protocol.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='exact72'):m.validate(m.sha(tmp_path/'protocol.json'))


def test_interrupted_controller_drains_only_owned_worker_and_restores_signals(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'guard',lambda role:None);monkeypatch.setattr(m,'validate',lambda digest:{})
    (tmp_path/'release.json').write_text('{}');monkeypatch.setattr(m,'record',lambda path:{'path':str(path),'sha256':'toy'})
    before={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)};cleaned=[]
    class Child:
        pid=999999
        def poll(self):signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
    child=Child();monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**kw:child)
    monkeypatch.setattr(m.origin.prior.shared.old.cost,'cleanup',lambda c:cleaned.append(c) or -15)
    result=m.run('toy')
    assert result['terminal'] and not result['completed'] and 'InterruptedError' in result['failure']
    assert cleaned==[child] and all(signal.getsignal(sig)==fn for sig,fn in before.items())


def test_real_cpu15_controller_and_generated_cpu14_worker_guard_no_models():
    m.guard('controller')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    result=subprocess.run(['taskset','-c','14',sys.executable,'-c',"from experiments.run_fixture_gaussian_linear_screen_v01 import guard;guard('worker');print('guard-PASS')"],cwd=m.ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0 and result.stdout.strip()=='guard-PASS',result.stderr
