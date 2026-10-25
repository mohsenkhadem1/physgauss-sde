"""Manufactured metadata/callbacks plus guards only; no CUDA/model/data calls."""
import inspect,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments import run_layout_grouped_gpu_parity_v01 as m

def test_strict_saved_CPU_risk_and_mass_parity():
    tol=dict(risk_absolute_m2=1e-20,risk_relative=1e-10,predictive_mass=2e-12);saved=dict(expected_target_variance=1e-9,prior_target_variance=2e-9,predictive_mass=1.,outcome_nodes=100)
    assert m.compare(saved,dict(saved),tol)['passed']
    assert not m.compare(saved,dict(saved,expected_target_variance=1.0001e-9),tol)['passed']
    assert not m.compare(saved,dict(saved,predictive_mass=.999999),tol)['passed']
    assert not m.compare(saved,dict(saved,outcome_nodes=101),tol)['passed']

def test_replay_accepts_only_saved_prefix_and_never_qualifies_references(monkeypatch):
    assert list(inspect.signature(m.replay_bank).parameters)==['observation']
    names=m.old.old.labels();all_names=sum(names.values(),());modal=SimpleNamespace(output_shape=(1,60),eigenvalues=np.array([1.]),coefficients=np.zeros((60,1)))
    monkeypatch.setattr(m.old.old.mechanics,'load_catalog',lambda _ :({},{}));monkeypatch.setattr(m.old.old.mechanics,'make_forward',lambda *a:None);monkeypatch.setattr(m.old.old.mechanics,'make_measurement_modal',lambda *a:SimpleNamespace(modal=modal,labels=all_names))
    monkeypatch.setattr(m.ConditionalAmplitudeReferenceV02,'qualify',lambda *a,**k:pytest.fail('No new reference qualification'))
    def fake(refs,actions,targets,**kw):
        assert len(refs)==4 and actions==names['candidate']and targets==names['target']and kw==dict(outer=512,amplitude_order=64)
        for r in refs:np.testing.assert_array_equal(r.y,np.arange(4)*1e-5)
        return 'bank','metadata'
    monkeypatch.setattr(m.CPU,'decision_bank',fake);assert m.replay_bank(np.arange(4)*1e-5)==('bank','metadata')

def test_rootrelease_precedes_saved_data_or_GPU(monkeypatch):
    monkeypatch.setattr(m,'validate',lambda _:(_ for _ in()).throw(ValueError('No rootrelease')));monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('No data before release'));monkeypatch.setattr(m.GPU,'Float64GroupedGPU',lambda *a,**k:pytest.fail('No CUDA before release'))
    with pytest.raises(ValueError,match='No rootrelease'):m.worker(0,'none')

def test_protocol_roundtrip_and_device_worklist_tamper(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'LOCAL',tmp_path/'pilot');monkeypatch.setattr(m.old.old,'ROOT',tmp_path)
    monkeypatch.setattr(m,'record',lambda p:dict(path=str(Path(p).relative_to(tmp_path)),sha256=m.sha(p)))
    source=tmp_path/'toy.py';source.write_text('value=1');external=tmp_path/'external.json';external.write_text(json.dumps(dict(controller_pid=123,child_pid=124)));origin=dict(external=m.record(external));monkeypatch.setattr(m,'origin_gate',lambda :origin);monkeypatch.setattr(m,'sources',lambda :{'toy.py':m.sha(source)});monkeypatch.setattr(m.old.old.mechanical_screen.shared.old,'absent',lambda _:None)
    protocol=m.prepare();p=json.loads((m.LOCAL/'protocol.json').read_text());m.write(m.LOCAL/'release.json',dict(approved=True,protocol_SHA256=protocol['sha256'],source_SHA256=p['source_SHA256'],origin=origin));assert len(m.validate(protocol['sha256'])['devices'])==3
    p['devices'].pop();m.write(m.LOCAL/'protocol.json',p)
    with pytest.raises(ValueError,match='frozen parity'):m.validate(m.sha(m.LOCAL/'protocol.json'))

def test_external_child_failure_retains_all3devices_no_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'validate',lambda _:dict(devices=list(m.GPU.ALLOWED_UUIDS)));monkeypatch.setattr(m,'guard',lambda *a:None)
    (tmp_path/'release.json').write_text('{}');monkeypatch.setattr(m,'record',lambda path:dict(path=str(path),sha256=m.sha(path)))
    class FailedChild:
        pid=123
        def poll(self):return 1
        def wait(self):return 1
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:FailedChild());monkeypatch.setattr(m.old.old.mechanical_screen.shared.old.cost,'cleanup',lambda _:1);monkeypatch.setattr(m,'owned_VRAM',lambda *a:pytest.fail('No GPUquery for exited child'))
    result=m.run('toy');assert not result['completed']and not result['qualified']and result['terminal']
    assert [d['status']for d in result['devices']]==['unresolved','pending','pending']

def test_all3_real_CPU16_guards_never_initialize_CUDA():
    for slot,uuid in enumerate(m.GPU.ALLOWED_UUIDS):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        result=subprocess.run(['taskset','-c','16',sys.executable,'-m','experiments.run_layout_grouped_gpu_parity_v01','worker-guard','--slot',str(slot)],cwd=m.ROOT,env=env,capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout)['no_models_or_CUDA']is True
