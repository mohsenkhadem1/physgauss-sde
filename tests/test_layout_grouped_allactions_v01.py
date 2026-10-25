import copy,inspect,json,os,subprocess,sys
import pytest
from experiments import run_layout_grouped_allactions_v01 as r

def rows():
    return [dict(job=j,result=dict(expected_target_variance=(1+j['action']*.01)*1e-10,prior_target_variance=2e-10,omitted_risk_bound=1e-20,predictive_mass=1.,omitted_mass_bound=1e-15))for j in r.worklist()]
def test_all48_and_unique_split():
    assert len(r.worklist())==48 and sorted(i for x in r.SPLIT for i in x)==list(range(8))
    out=r.classify(rows(),dict(outcome_relative=1e-4,latent_relative=1e-3));assert out['qualified']and out['unique_best_action']==0
    for changed in(rows()[:-1],rows()[:-1]+[rows()[0]]):
        with pytest.raises(ValueError,match='48'):r.classify(changed,dict(outcome_relative=1e-4,latent_relative=1e-3))
def test_refinement_negative_cannot_choose_and_ties():
    v=rows();v[0]['result']['expected_target_variance']+=1e-11
    out=r.classify(v,dict(outcome_relative=1e-4,latent_relative=1e-3));assert not out['qualified']and out['unique_best_action']is None
    v=rows()
    for row in v:row['result']['expected_target_variance']=1e-10
    out=r.classify(v,dict(outcome_relative=1e-4,latent_relative=1e-3));assert out['qualified']and out['numerical_ties']==list(range(8))and out['unique_best_action']is None

def test_release_before_data_and_no_true_parameters():
    text=inspect.getsource(r.worker);assert text.index('validate(digest)')<text.index('np.load')
    assert list(inspect.signature(r.make_refs).parameters)==['observation']
    assert '.qualify('not in inspect.getsource(r.make_refs)

def test_real_generated_guards_no_cuda():
    for slot,uuid in enumerate(r.parity.GPU.ALLOWED_UUIDS):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        result=subprocess.run(['taskset','-c','16',sys.executable,'-m','experiments.run_layout_grouped_allactions_v01','worker-guard','--slot',str(slot)],env=env,capture_output=True,text=True,timeout=15)
        assert result.returncode==0,result.stderr

def test_json_roundtrip_exact_worklist_and_release(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'LOCAL',tmp_path);monkeypatch.setattr(r,'record',lambda path:dict(path=str(path),sha256=r.sha(path)));monkeypatch.setattr(r,'origin_gate',lambda:{'safe':'metadata'});monkeypatch.setattr(r,'sources',lambda:{'test':'hash'})
    receipt=r.prepare();p=json.loads((tmp_path/'protocol.json').read_text());assert p['worklist']==r.worklist()and p['split']==[list(x)for x in r.SPLIT]
    with pytest.raises(FileExistsError):r.prepare()
    monkeypatch.setattr(r.parity.old.old,'verify_mechanics_receipt',lambda p:None)
    r.write(tmp_path/'release.json',dict(approved=False))
    with pytest.raises(ValueError,match='release'):r.validate(receipt['sha256'])

def test_lost_predictive_density_and_invalid_tail_refused():
    for key,value in(('predictive_mass',.8),('omitted_mass_bound',-1.),('omitted_risk_bound',float('nan'))):
        data=rows();data[0]['result'][key]=value
        with pytest.raises(ValueError,match='predictive mass'):r.classify(data,dict(outcome_relative=1e-4,latent_relative=1e-3))

def test_worker_device_protocol_pid_and_job_context(monkeypatch):
    monkeypatch.setattr(r,'record',lambda p:{'bound':'release'})
    p=dict(devices=['G0','G1','G2'],split=[list(x)for x in r.SPLIT],worklist=r.worklist());row=dict(slot=0,pid=321,physical_UUID='G0');value=dict(row,completed=True,qualified=True,protocol_SHA256='proto',release={'bound':'release'},jobs=[j for j in rows()if j['job']['action']in r.SPLIT[0]])
    assert r.worker_context(value,row,p,'proto')is value
    for key,bad in(('physical_UUID','G3'),('slot',1),('protocol_SHA256','wrong'),('pid',999)):
        changed=copy.deepcopy(value);changed[key]=bad
        with pytest.raises(ValueError,match='context'):r.worker_context(changed,row,p,'proto')
    changed=copy.deepcopy(value);changed['jobs'].pop()
    with pytest.raises(ValueError,match='jobs'):r.worker_context(changed,row,p,'proto')
