import json,os,subprocess
from pathlib import Path
import numpy as np
import pytest
from experiments import run_surface_nonlinear_GPU_control_v02 as run


def test_release_before_archive_noise_or_GPU(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'guard',lambda slot=None:None);monkeypatch.setattr(run,'validate',lambda _:{});monkeypatch.setattr(run,'LOCAL',tmp_path)
    monkeypatch.setattr(run.np,'load',lambda *a,**k:pytest.fail('No archive before root release'))
    monkeypatch.setattr(run.np.random,'default_rng',lambda *a:pytest.fail('No noise before root release'))
    monkeypatch.setattr(run.study,'make_model',lambda *a,**k:pytest.fail('No GPU before root release'))
    with pytest.raises(FileNotFoundError):run.worker(0,'unreleased')


def test_fixed_cases_independent_reset_labels_and_seeds():
    assert run.CASES[0]['noise_seeds']==[]
    assert run.CASES[1]['labels']==[run.PREFIX,run.PREFIX]
    assert run.CASES[2]['unit']==[0.,1.]
    assert run.CASES[2]['labels']==[run.PREFIX,run.AXIAL]
    assert [seed for case in run.CASES for seed in case['noise_seeds']]==[2026103001,2026103002,2026103003]
    assert len(set(run.GPU_UUIDS))==3 and run.RESOURCES['worker_wall_s']==1800


def test_corner_observations_use_direct_fields_independent_noise_and_fail_tamper(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'ROOT',tmp_path)
    direct={run.PREFIX:np.arange(24)*1e-6,run.AXIAL:np.arange(24)*2e-6}
    origin={'corner':{}}
    for label,response in direct.items():
        p=tmp_path/(label+'.npz');np.savez(p,unit=np.array([0.,1.]),response_m=response)
        origin['corner'][label]={'raw':{'path':p.name,'sha256':run.sha(p)}}
    rng=run.np.random.default_rng
    # Distinct toy streams, never draw actual prospective control noise in tests.
    monkeypatch.setattr(run.np.random,'default_rng',lambda seed:rng(12000+seed%100))
    got=run.observations(2,origin,None)
    assert [label for label,_ in got]==[run.PREFIX,run.AXIAL]
    np.testing.assert_array_equal(got[0][1],direct[run.PREFIX]+rng(12002).normal(0,20e-6,24))
    np.testing.assert_array_equal(got[1][1],direct[run.AXIAL]+rng(12003).normal(0,20e-6,24))
    (tmp_path/(run.AXIAL+'.npz')).write_bytes(b'altered')
    with pytest.raises(ValueError,match='raw bytes'):run.observations(2,origin,None)


def test_case0_CPU_parity_is_strict_and_nonfinite_fails():
    b={key:np.zeros(3)for key in('predictive_mean','predictive_SD','unit_CDF')}
    assert run.parity(b,b)['passed']
    c={key:v.copy()for key,v in b.items()};c['predictive_mean'][0]=3e-12
    assert not run.parity(c,b)['passed']
    c={key:v.copy()for key,v in b.items()};c['unit_CDF'][1]=2e-8
    assert not run.parity(c,b)['passed']
    c['unit_CDF'][1]=np.nan
    assert not run.parity(c,b)['passed']


def test_completed_gate_refuses_negative_execution_before_anyarrays(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'LOCAL',tmp_path);monkeypatch.setattr(run,'validate',lambda _:{});monkeypatch.setattr(run,'release_gate',lambda _:None)
    (tmp_path/'execution.json').write_text(json.dumps(dict(completed=True,qualified=False,terminal=True)))
    monkeypatch.setattr(run.np,'load',lambda *a,**k:pytest.fail('Metadata gate only'))
    with pytest.raises(ValueError,match='qualified three-GPU'):run.completed_GPU_control_gate('negative')


def test_generated_three_GPU_guards_no_context_or_data():
    for slot in range(3):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':run.GPU_UUIDS[slot],'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        out=subprocess.run(run.worker_command(slot,'no-protocol',guard_only=True),cwd=run.ROOT,env=env,capture_output=True,text=True,timeout=20)
        assert out.returncode==0,out.stderr


def test_cost_admission_precedes_prepare_no_draw(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'guard',lambda *a:None)
    monkeypatch.setattr(run.cost,'completed_cost_gate',lambda _:(_ for _ in()).throw(ValueError('CPUcost unavailable')))
    monkeypatch.setattr(run.np,'load',lambda *a,**k:pytest.fail('No arrays in prepare'))
    monkeypatch.setattr(run.np.random,'default_rng',lambda *a:pytest.fail('No noise in prepare'))
    with pytest.raises(ValueError,match='CPUcost'):run.prepare()


def test_shared_callback_keeps_singleton_CPU_and_batched_GPU(monkeypatch):
    class Model:
        modal=object();labels=['a']
        def ratio_select(self,labels):
            def evaluate(z):calls.append(('gpu',len(z)));return np.exp(z)[:,None,None]
            return evaluate
    calls=[]
    def CPU(modal,labels):
        def evaluate(r,selected):calls.append(('cpu',len(r)));return r[:,None,None]
        return evaluate
    monkeypatch.setattr(run.study,'RatioModal',CPU);monkeypatch.setattr(run.study,'model_fingerprint',lambda _: 'toy')
    monkeypatch.setattr(run.study,'ExactRatioMemo',lambda callback,**kwargs:callback)
    cb=run.exact_ratio_callback(Model())
    for n in(1,15,16,32):
        z=np.linspace(-3,-2,n);np.testing.assert_array_equal(cb(z,['a']),np.exp(z)[:,None,None])
    assert calls==[('cpu',1),('cpu',15),('gpu',16),('gpu',32)]


def test_unicode_archive_labels_use_original_Python_string_fingerprint(monkeypatch,tmp_path):
    from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas
    values=np.zeros((9,9,2,24));labels=('normal','axial');base=NonlinearResidualAtlas(values,labels,provenance_fingerprint='toy-origin')
    path=tmp_path/'atlas.npz';np.savez(path,delta=values,labels=np.array(labels),axis=base.axis)
    monkeypatch.setattr(run,'ROOT',tmp_path)
    origin=dict(full_atlas=dict(archive=dict(path=path.name,sha256=run.sha(path))),full_protocol_SHA256='toy-origin',atlas_fingerprint=base.fingerprint,labels=list(labels))
    got=run.admitted_atlas(origin)
    assert got.fingerprint==base.fingerprint and all(type(v)is str for v in got.labels)


def test_real_admitted_atlas_loader_only_no_observation_or_rng(monkeypatch):
    p=json.loads((run.cost.LOCAL/'protocol.json').read_text())
    origin=dict(full_atlas=p['origin_v01']['full_atlas'],full_protocol_SHA256=p['full_protocol_SHA256'],atlas_fingerprint=p['full_atlas_fingerprint'],labels=p['all_target_labels'])
    monkeypatch.setattr(run.np.random,'default_rng',lambda *a:pytest.fail('No noise/material draw in loader control'))
    got=run.admitted_atlas(origin)
    assert got.fingerprint=='6e485eaf5266aa1cc04cd4882a8c0652b5922d6f6f99a3a9d9edab41a416ec13'
    assert got.delta.shape==(9,9,10,24)and all(type(v)is str for v in got.labels)


def test_old_technical_failure_gate_rejects_missing_closure_before_arrays(monkeypatch,tmp_path):
    monkeypatch.setattr(run.previous,'validate',lambda _:None);monkeypatch.setattr(run.previous,'release_gate',lambda _:None);monkeypatch.setattr(run.previous,'LOCAL',tmp_path)
    (tmp_path/'execution.json').write_text(json.dumps(dict(terminal=False)))
    monkeypatch.setattr(run.np,'load',lambda *a,**k:pytest.fail('Oldnegative gate metadata only'))
    with pytest.raises(ValueError,match='technical negative'):run.failed_v01_gate()


def test_actual_old_failure_binding_is_metadata_only(monkeypatch):
    monkeypatch.setattr(run.np,'load',lambda *a,**k:pytest.fail('No actualposterior/observation arrays in closuregate'))
    monkeypatch.setattr(run.np.random,'default_rng',lambda *a:pytest.fail('No oldfailure RNG'))
    origin=run.failed_v01_gate()
    assert len(origin['results'])==3 and origin['no_control_observations_generated']
