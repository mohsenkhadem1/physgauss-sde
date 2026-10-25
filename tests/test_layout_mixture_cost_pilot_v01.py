"""Manufactured models/metadata only; no FE, campaign fields or random draws."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from scipy.special import logsumexp
from experiments import run_layout_mixture_cost_pilot_v01 as m

class ToyModel:
    def __init__(self,k):self.k=k
    def select(self,names):
        roles=m.labels();all_labels=sum(roles.values(),());indices=[all_labels.index(s)for s in names]
        def predict(unit):
            unit=np.asarray(unit);full=np.empty((len(unit),60,1))
            for j in range(60):full[:,j,0]=1e-5*((j%4+1)*unit[:,0]+(j%3-1)*unit[:,1]+self.k*.3)
            return full[:,indices]
        return predict

def test_positive_GL_prior_normalized_and_exact_polynomial_moments():
    for order in m.ORDERS:
        u,p=m.positive_grid(order);assert len(u)==order**2 and np.all(p>0)and np.sum(p)==pytest.approx(1,abs=1e-14)
        np.testing.assert_allclose(p@u,[.5,.5],atol=1e-14);assert p@(u[:,0]**4*u[:,1]**2)==pytest.approx(1/15,abs=1e-14)

def test_generating_interface_reads_only_prefix_not_future_targets():
    calls=[]
    def prefix(unit):calls.append(unit.copy());return np.ones((1,4,1))*1e-5
    y=m.prefix_only(prefix);np.testing.assert_array_equal(calls[0],[[.5,.5]]);assert y.shape==(4,)
    with pytest.raises(ValueError,match='Only canonical'):m.prefix_only(lambda u:np.zeros((1,60,1)))

def test_all_category_evidence_and_prefix_bank_equal_independent_flat_Bayes():
    models=[ToyModel(k)for k in range(4)];y=np.array([1,2,3,4])*1e-5;bank,meta=m.posterior_bank(models,9,y);u,prior=m.positive_grid(9);lls=[]
    for model in models:
        pred=model.select(m.labels()['prefix'])(u)[:,:,0];ll=-.5*np.sum(((pred-y)/m.mechanics.SIGMA_M)**2,axis=1)-4*np.log(m.mechanics.SIGMA_M*np.sqrt(2*np.pi));lls.extend((np.log(.25*prior)+ll).tolist())
    expected=np.exp(np.asarray(lls)-logsumexp(lls));np.testing.assert_allclose(bank.probability,expected,rtol=1e-13)
    assert meta['hypotheses']==324 and bank.candidate.shape==(324,8)and bank.target.shape==(324,48)
    assert len(meta['layout_probability'])==4

def test_policy_bank_signature_has_no_generating_case_or_future_observation():
    import inspect
    assert list(inspect.signature(m.posterior_bank).parameters)==['models','order','observation']
    with pytest.raises(ValueError,match='All four'):m.posterior_bank([ToyModel(0)],9,np.zeros(4))

def test_parameter_witness_and_all_action_retention_no_silent_filter():
    b,meta=m.posterior_bank([ToyModel(k)for k in range(4)],9,np.zeros(4));actions=[dict(action=i,levels=[dict(expected_target_variance=(i+1)*1e-10)],successive_error=1e-15,qualified=True)for i in range(8)]
    with pytest.raises(ValueError,match='All8'):m.level_summary(b,meta,actions[:-1])
    s=m.level_summary(b,meta,actions);assert len(s['fixed_action_regret'])==8 and s['unique_best_action']==0
    bad=[dict(a,qualified=False)for a in actions];assert m.level_summary(b,meta,bad)['unique_best_action']is None
    t=dict(s,parameter_order=17);assert m.parameter_witness(s,t)['passed']
    t=dict(t,expected_target_risk=[1.]*8);assert not m.parameter_witness(s,t)['passed']

def test_failed_mechanics_gate_no_models_arrays_or_noise(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path/'pilot')
    monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('No arrays before mechanics/root admission'))
    monkeypatch.setattr(m.mechanical_screen,'completed_screen_gate',lambda _:(_ for _ in()).throw(ValueError('Closed mechanical screen')))
    with pytest.raises(ValueError,match='Closed mechanical'):m.prepare('none')


def test_worker_release_precedes_any_FE_or_data(monkeypatch):
    monkeypatch.setattr(m,'validate',lambda _:(_ for _ in()).throw(ValueError('No release')));monkeypatch.setattr(m.mechanics,'make_forward',lambda *a:pytest.fail('No FE before release'))
    with pytest.raises(ValueError,match='No release'):m.worker('none')

def test_frozen_worklist_failure_keeps_all_pending_contexts(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path);work=[dict(order=n,action=a)for n in m.ORDERS for a in range(8)]
    (tmp_path/'release.json').write_text('{}');monkeypatch.setattr(m,'record',lambda p:dict(path=str(p),sha256=m.sha(p)));monkeypatch.setattr(m,'validate',lambda _:dict(worklist=work));monkeypatch.setattr(m,'guard',lambda :None)
    monkeypatch.setattr(m.mechanics,'load_catalog',lambda _:(_ for _ in()).throw(TimeoutError('manufactured cap')))
    with pytest.raises(TimeoutError):m.worker('toy')
    state=json.loads((tmp_path/'execution.json').read_text());assert not state['completed']and not state['qualified']and len(state['all_worklist'])==24 and state['completed_actions']==[]


def test_frozen_protocol_JSON_roundtrip_source_and_receipt_before_data(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path/'pilot');monkeypatch.setattr(m,'ROOT',tmp_path)
    source=tmp_path/'toy.py';source.write_text('value=1');artifact=tmp_path/'raw.npz';artifact.write_bytes(b'bound metadata only')
    execution=tmp_path/'execution.json';execution.write_text(json.dumps(dict(controller_pid=123,worker_pid=124)))
    monkeypatch.setattr(m.mechanical_screen.shared.old,'absent',lambda _:None)
    admission=dict(all_numeric_checks_passed=True,development_only=True,raw=m.record(artifact),execution=m.record(execution))
    monkeypatch.setattr(m.mechanical_screen,'completed_screen_gate',lambda _:admission);monkeypatch.setattr(m,'sources',lambda :{'toy.py':m.sha(source)})
    p=m.prepare('fake');protocol=json.loads((m.LOCAL/'protocol.json').read_text());release=dict(approved=True,protocol_SHA256=p['sha256'],source_SHA256=protocol['source_SHA256'],mechanics_admission=admission,external_watchdog_120s_4GiB=True)
    m.write(m.LOCAL/'release.json',release);assert len(m.validate(p['sha256'])['worklist'])==24
    artifact.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='mechanical input'):m.validate(p['sha256'])


def test_real_CPU17_resource_guard_no_models_or_data():
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    out=subprocess.run(['taskset','-c','17',sys.executable,'-m','experiments.run_layout_mixture_cost_pilot_v01','worker-guard'],cwd=m.ROOT,env=env,capture_output=True,text=True,timeout=20)
    assert out.returncode==0,out.stderr
    assert json.loads(out.stdout)['no_models_or_data']is True
