import copy,inspect,json,os,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
import pytest
from experiments import run_fixture_balanced_decision_screen_v01 as r

def test_prefix_zip_row_allowlist_and_no_future_materialization(tmp_path):
    x=np.full((72,60),np.nan);x[13,:4]=[1,2,3,4];p=tmp_path/'x.npz';np.savez_compressed(p,responses=x,forbidden=np.ones((3,3)))
    assert np.array_equal(r.prefix_row(p,13),[1,2,3,4])
    with pytest.raises(ValueError):r.prefix_row(p,72)
    np.savez_compressed(p,responses=np.zeros((72,59)))
    with pytest.raises(ValueError,match='row-major'):r.prefix_row(p,0)

def test_policy_signature_is_prefix_only_and_release_first():
    assert list(inspect.signature(r.policy).parameters)==['observation','callbacks','names','directory','checkpoint','old_witnesses']
    for fn in(r.build,r.worker):
        s=inspect.getsource(fn);assert s.index('validate(digest)')<s.index('np.load')
    assert '.qualify('not in inspect.getsource(r.build)

def test_model_archive_canonical_labels_all_modes_no_clipping(tmp_path,monkeypatch):
    e=np.array([-1e-15,0.,.1]);c=np.arange(180.).reshape(60,3);labels=tuple('label'+str(i)for i in range(60));p=tmp_path/'m.npz';np.savez_compressed(p,eigenvalues=e,coefficients=c,labels=np.array(labels));metadata=dict(models=[dict(record=dict(path=str(p)),model_fingerprint=r.model_fingerprint(e,c,labels),all_modes=3)])
    models=r.load_models(metadata);assert np.array_equal(models[0].modal.eigenvalues,e)and models[0].labels==labels
    metadata['models'][0]['model_fingerprint']='changed'
    with pytest.raises(ValueError,match='fingerprint'):r.load_models(metadata)

def test_real_all_roles_guards_without_model_data_or_cuda():
    entries=[('builder',17,None,''),('controller',18,None,'')]+[('worker',i,i,u)for i,u in enumerate(r.old.parity.GPU.ALLOWED_UUIDS)]
    for role,cpu,slot,uuid in entries:
        command=['taskset','-c',str(cpu),sys.executable,'-m','experiments.run_fixture_balanced_decision_screen_v01','guard','--role',role]
        if slot is not None:command+=['--slot',str(slot)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');v=subprocess.run(command,env=env,capture_output=True,text=True,timeout=15);assert v.returncode==0,v.stderr

def test_json_protocol_roundtrip_config_and_release(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'LOCAL',tmp_path);monkeypatch.setattr(r,'guard',lambda *a,**k:None);monkeypatch.setattr(r,'origin_gate',lambda:{'safe':'metadata'});monkeypatch.setattr(r,'sources',lambda:{'test':'hash'});monkeypatch.setattr(r,'record',lambda path:dict(path=str(path),sha256=r.sha(path)));monkeypatch.setattr(r,'verify_records',lambda p:None)
    rec=r.prepare();p=json.loads((tmp_path/'protocol.json').read_text());assert len(p['worklist'])==36 and sum(map(len,p['split']))==36 and len(p['action_worklist'])==48
    r.write(tmp_path/'release.json',dict(approved=False))
    with pytest.raises(ValueError,match='release'):r.validate(rec['sha256'])
    with pytest.raises(FileExistsError):r.prepare()

def test_full_prefix_policy_keeps_all_categories_banks_actions_and_no_truth(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import contextlib
    calls=[]
    class Ref:
        def __init__(self,*a,**k):pass
        def qualify(self,plan):calls.append('qualify');return dict(qualified=True,diagnostics=dict(levels=[dict(outer=o,inner=i,status='computed',log_evidence=0.)for o,i in((512,128),(512,256),(512,512),(128,512),(256,512))]),local_final={})
    class Bank:
        def prior_moments(self):return dict(total_variance=1e-8,between_layout_variance=2e-9)
    def bank(*a,outer,amplitude_order):
        calls.append((outer,amplitude_order));return Bank(),dict(layout_probability=[.25]*4,category_moments=[],log_evidence=[0]*4,weighted_hypotheses=32)
    class Engine:
        def __init__(self,*a,**k):self.torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
        def expected_risk(self,action,sigma,order):return dict(expected_target_variance=(1+action*.1)*1e-10,prior_target_variance=1e-8,predictive_mass=1.,omitted_mass_bound=1e-15,omitted_risk_bound=1e-20)
    monkeypatch.setattr(r,'ConditionalAmplitudeReferenceV02',Ref);monkeypatch.setattr(r.old.parity.old,'reference_check',lambda v:{'passed':True});monkeypatch.setattr(r.old.parity.CPU,'decision_bank',bank);monkeypatch.setattr(r.old.parity.CPU,'reference_moment_checks',lambda *a:dict(passed=True,layout_TV=0.));monkeypatch.setattr(r.old.parity.GPU,'Float64GroupedGPU',Engine);monkeypatch.setattr(r,'record',lambda p:dict(path=str(p),sha256=r.sha(p)));monkeypatch.setenv('CUDA_VISIBLE_DEVICES','toy')
    old_witnesses=dict(categories=[],decisions=[])
    for k in range(4):
        path=tmp_path/f'oldcat-{k}.json';r.write(path,dict(diagnostics=Ref().qualify(None)['diagnostics']));old_witnesses['categories'].append(r.record(path))
    for k in range(5):
        path=tmp_path/f'olddec-{k}.json';r.write(path,dict(metadata=dict(log_evidence=[0]*4,layout_probability=[.25]*4)));old_witnesses['decisions'].append(r.record(path))
    calls.clear()
    result=r.policy(np.zeros(4),[(None,None)]*4,dict(prefix=tuple('p'+str(i)for i in range(4)),candidate=tuple('a'+str(i)for i in range(8)),target=tuple('t'+str(i)for i in range(48))),tmp_path,lambda:None,old_witnesses)
    assert calls.count('qualify')==4 and len(result['decision_levels'])==5 and len(result['jobs'])==48 and result['qualified']and result['ranking']['unique_best_action']==0

def test_real_closed_origin_metadata_only_layout_mapping():
    value=r.origin_gate();assert len(value['prefix_rows'])==36 and[value['case_index']for value in value['prefix_rows']]==list(range(36))
    assert len({value['raw_row']for value in value['prefix_rows']})==36

def test_builder_uses_new_helper_and_reuses_prefix_bytes():
    text=inspect.getsource(r.build)
    assert 'balanced.make_forward' in text and 'balanced.load_catalog'in text and 'balanced.make_measurement_modal'in text
    assert "prefixes=p['origin']['old36_prefixes']"in text and 'observed_prefix='not in text
    assert all(isinstance(r.layout_id(i),str)for i in range(4))

def test_already_exited_child_failure_and_cleanup_failure_refused(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(r.old.parity.old.old.mechanical_screen.shared.old.cost,'cleanup',lambda child:0)
    for exit_code in(1,-15):
        child=SimpleNamespace(wait=lambda:exit_code);row={}
        with pytest.raises(RuntimeError,match='final wait'):r.finalize_child(child,row)
        assert row['reaped']and row['exit_code']==exit_code
    monkeypatch.setattr(r.old.parity.old.old.mechanical_screen.shared.old.cost,'cleanup',lambda child:1)
    with pytest.raises(RuntimeError):r.finalize_child(SimpleNamespace(wait=lambda:0),{})

def test_first12_full_prior_coefficient_bound_and_all_eigenvalues():
    from types import SimpleNamespace
    e=np.array([-1e-15,0.,.3]);c=np.arange(180.).reshape(60,3)*.001;labels=tuple(str(i)for i in range(60));model=r.LabelledModal(SimpleNamespace(eigenvalues=e,coefficients=c.copy(),output_shape=(1,60)),labels);other=r.LabelledModal(SimpleNamespace(eigenvalues=e.copy(),coefficients=c.copy(),output_shape=(1,60)),labels)
    assert r.coefficient_invariance(model,other)['maximum_full_prior_response_difference_bound_m']==0
    model.modal.coefficients[40]+=2.;assert r.coefficient_invariance(model,other)['maximum_full_prior_response_difference_bound_m']==0
    model.modal.coefficients[0,0]+=.01
    with pytest.raises(ValueError,match='coefficient'):r.coefficient_invariance(model,other)
    model.modal.coefficients=other.modal.coefficients.copy();model.modal.eigenvalues=np.maximum(e,0)
    with pytest.raises(ValueError,match='spectrum'):r.coefficient_invariance(model,other)

def test_latent_evidence_probability_invariance_mandatory():
    levels=[dict(outer=o,inner=i,status='computed',log_evidence=0.)for o,i in((512,128),(512,256),(512,512),(128,512),(256,512))];assert r.reference_evidence_invariance(dict(levels=levels),dict(levels=levels))['passed']
    changed=copy.deepcopy(levels);changed[0]['log_evidence']=.001;assert not r.reference_evidence_invariance(dict(levels=levels),dict(levels=changed))['passed']
    old=dict(log_evidence=[0.]*4,layout_probability=[.25]*4);assert r.decision_evidence_invariance(old,old)['passed']
    new=copy.deepcopy(old);new['layout_probability'][0]+=.0001;assert not r.decision_evidence_invariance(old,new)['passed']

def test_target_energy_share_measured_not_assumed_balance():
    cases=r.design.worklist();responses=np.zeros((36,48))
    for i,s in enumerate(cases):responses[i,:24]=s['layout'];responses[i,24:]=2*s['layout']
    out=r.separation_energy(responses,cases);assert out['same_E_K_layout_pairs']==54 and np.allclose(out['energy_share'],[.2,.8])and out['diagnostic_only_not_gate']
    with pytest.raises(ValueError,match='36'):r.separation_energy(responses[:-1],cases[:-1])
