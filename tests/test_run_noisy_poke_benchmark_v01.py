"""Manufactured orchestration controls; no campaign RNG, PDE, CUDA or truth reads."""
import copy,json,os,subprocess,sys,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments import run_noisy_poke_benchmark_v01 as s

@pytest.fixture
def local(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(s,'LOCAL',tmp_path/'benchmark')
    def record(path):
        p=Path(path);return dict(path=str(p.relative_to(tmp_path)),sha256=s.sha(p))
    def verify(v):
        if isinstance(v,dict):
            if set(('path','sha256'))<=v.keys():assert record(tmp_path/v['path'])==v
            for value in v.values():verify(value)
        elif isinstance(v,list):
            for value in v:verify(value)
    monkeypatch.setattr(s,'record',record);monkeypatch.setattr(s,'verify',verify)
    monkeypatch.setattr(s,'guard',lambda *a:None)
    return tmp_path

def test_frozen_design_metadata_only_matches_interior8_seeds_shapes_baseline():
    assert s.design_gate()['sha256']==s.DESIGN_SHA256

def test_prepare_JSON_roundtrip_release_and_design_tampering_failclosed(local,monkeypatch):
    approved=local/'design.json';s.write(approved,dict(source='manufactured'));r=s.record(approved)
    monkeypatch.setattr(s,'design_gate',lambda:r);monkeypatch.setattr(s,'sources',lambda:{'toy.py':'a'})
    monkeypatch.setattr(s,'seed_registration_check',lambda:dict(no_noise_generated=True));origin=dict(prospective_design=r,fixed_baseline=dict(action=5))
    monkeypatch.setattr(s,'origin_gate',lambda:origin)
    monkeypatch.setattr(s.design,'noise_streams',lambda:pytest.fail('prepare must not draw'))
    result=s.prepare();p=s.read(s.LOCAL/'protocol.json');digest=result['sha256']
    s.write(s.LOCAL/'release.json',dict(approved=True,protocol_SHA256=digest,source_SHA256=p['source_SHA256'],origin=origin))
    assert s.validate(digest)['reference_plan']==s.CONTEXT_PLAN
    with pytest.raises(FileExistsError):s.prepare()
    p['noise_shapes']['poke']=[4,8,7];s.write(s.LOCAL/'protocol.json',p)
    with pytest.raises(ValueError):s.validate(s.sha(s.LOCAL/'protocol.json'))

def test_closed_or_unqualified_origin_blocks_before_protocol_or_noise(local,monkeypatch):
    monkeypatch.setattr(s,'seed_registration_check',lambda:{})
    def blocked():raise ValueError('NH catalog not admitted')
    monkeypatch.setattr(s,'origin_gate',blocked);monkeypatch.setattr(s.design,'noise_streams',lambda:pytest.fail('No draw before admission'))
    with pytest.raises(ValueError):s.prepare()
    assert not(s.LOCAL/'protocol.json').exists()

def test_prefix_reader_only_four_public_Z_rows_no_other_member(local,monkeypatch):
    path=local/'capture.npz';values=np.arange(48.,dtype=float).reshape(16,3);np.savez_compressed(path,query_displacement_m=values,forbidden_future=np.array([999.]))
    old=zipfile.ZipFile.open;seen=[]
    def opened(self,name,*a,**k):seen.append(name);return old(self,name,*a,**k)
    monkeypatch.setattr(zipfile.ZipFile,'open',opened)
    assert np.array_equal(s.prefix_capture_only(path),values[:4,2]);assert seen==['query_displacement_m.npy']
    np.savez_compressed(path,query_displacement_m=values.astype(np.float32))
    with pytest.raises(ValueError):s.prefix_capture_only(path)

def test_inputs_release_barrier_before_rng(local,monkeypatch):
    monkeypatch.setattr(s,'validate',lambda _:(_ for _ in()).throw(ValueError('release missing')))
    monkeypatch.setattr(s.design,'noise_streams',lambda:pytest.fail('No pre-release RNG'))
    with pytest.raises(ValueError):s.inputs('toy')
    assert not(s.LOCAL/'inputs').exists()

def setup_reference(monkeypatch,*,unresolved=False,drift=False):
    class Ref:
        def __init__(self,k,*a,**kw):self.k=k
        def _state(self,outer,inner):return dict(log_evidence=float(np.log(self.k+1)))
        def cdf(self,d,q,outer,inner,**k):return q**(self.k+1+(1 if drift and(outer,inner)==(256,512)else 0))
        def qualify(self,**kw):return dict(qualified=not(unresolved and self.k==2),diagnostics={},local_final=dict(log_evidence=float(np.log(self.k+1)),predictive_mean=np.full(48,self.k*1e-5),predictive_SD=np.full(48,1e-5),unit_mean=np.full(2,.5)))
    monkeypatch.setattr(s.base,'ConditionalAmplitudeReferenceV02',Ref)
    monkeypatch.setattr(s.base.old.parity.old,'reference_check',lambda result:dict(passed=result['qualified']))
    monkeypatch.setattr(s.base.old.parity.CPU,'decision_bank',lambda *a,**k:(None,dict(deliberately_not_continuous_probability=[1.,0,0,0])))
    monkeypatch.setattr(s.base.old.parity.CPU,'reference_moment_checks',lambda *a:dict(passed=True))
    return [(i,SimpleNamespace(model_fingerprint=f'{i:064x}'))for i in range(4)],dict(prefix=list('abcd'),candidate=list('efghijkl'),target=[f't{i}'for i in range(48)])

def test_infer_continuous_logZ_mixture_and_three_axis_interval_witnesses(local,monkeypatch):
    callbacks,names=setup_reference(monkeypatch);directory=local/'infer';directory.mkdir()
    result,refs,_,_=s.infer([],callbacks,names,directory,lambda:None)
    assert result['qualified']and len(result['categories'])==4 and len(result['decision_levels'])==5
    assert result['mixture_interval_checks']['levels']==[[512,512],[512,256],[256,512]]
    for k,v in enumerate(result['categories']):
        context=s.read(local/v['record']['path'])['reference_context'];assert context==dict(observations_fingerprint=s.observations_fingerprint([]),model_fingerprint=f'{k:064x}',reference_plan=s.CONTEXT_PLAN,target_labels=names['target'],sigma_m=s.design.physics.SIGMA_M)
    with np.load(local/result['snapshot']['path'])as z:
        assert np.allclose(z['layout_probability'],[.1,.2,.3,.4]);assert np.allclose(z['mean_m'],2e-5)
        for q,alpha in zip(z['material_unit_quantiles'][0],[.05,.95]):assert abs(sum((k+1)*q**(k+1)/10 for k in range(4))-alpha)<1e-7

def test_failed_category_or_mixture_axis_keeps_negative_no_snapshot(local,monkeypatch):
    for case in('category','mixture'):
        callbacks,names=setup_reference(monkeypatch,unresolved=case=='category',drift=case=='mixture');directory=local/case;directory.mkdir()
        result,refs,_,_=s.infer([],callbacks,names,directory,lambda:None)
        assert result['qualified']is False and refs is None and not(directory/'posterior-LOCAL.npz').exists()
        assert len(result['categories'])==4
        assert all('reference_context'in s.read(local/v['record']['path'])for v in result['categories'])

def evaluation_fixture(local,monkeypatch,*,failed_action=None):
    p=dict(trials=s.design.trials(),objects=s.design.objects(),evaluation_checkpoint=[0],evaluation_remaining=[[i for i in range(1,32)if i%3==k]for k in range(3)],devices=list(s.base.old.parity.GPU.ALLOWED_UUIDS))
    s.write(s.LOCAL/'release.json',dict(approved=True));prefix_dir=s.LOCAL/'toy-prefix';prefix_dir.mkdir()
    snap=prefix_dir/'snapshot.npz';np.savez_compressed(snap,mean_m=np.ones(48)*1e-5,SD_m=np.ones(48)*2e-5,unit_mean=np.ones(2)*.5,layout_probability=np.ones(4)/4,material_quantiles=np.array([[2100.,7900.],[5.1,29.]]))
    result=prefix_dir/'result.json';s.write(result,dict(qualified=True,snapshot=s.record(snap)))
    seals=[]
    for i in range(32):
        path=s.LOCAL/f'toy-seal-{i}.json';s.write(path,dict(qualified=True,adaptive_action=2,fixed_action=5,prefix_result=s.record(result)));seals.append(dict(index=i,seal=s.record(path)))
    prefix=prefix_dir/'observed.npz';np.savez_compressed(prefix,observed_prefix=np.array([11.,12.,13.,14.])*1e-5)
    poke=prefix_dir/'poke.npz';np.savez_compressed(poke,poke_noise=np.arange(256.,dtype=float).reshape(4,8,8)*1e-7)
    m=dict(prefixes=[dict(record=s.record(prefix))for i in range(32)],poke_noise=s.record(poke));events=[];seen=[]
    monkeypatch.setattr(s,'validate',lambda _:p);monkeypatch.setattr(s,'inputs_gate',lambda _:m)
    def sealed(_):events.append('all32_sealed');return seals
    monkeypatch.setattr(s,'all_seals_gate',sealed)
    names=dict(prefix=[f'p{i}'for i in range(4)],candidate=[f'a{i}'for i in range(8)],target=[f't{i}'for i in range(48)])
    class Model:
        def select(self,_):return lambda u:np.zeros((len(u),48,1))
    monkeypatch.setattr(s,'load_common',lambda *a:([Model()]*4,None,names))
    def truth(*a):assert events==['all32_sealed'];events.append('future_truth');return np.arange(60.)*1e-5,np.zeros(48),[]
    monkeypatch.setattr(s,'query_truth',truth)
    def inferred(obs,callbacks,labels,directory,checkpoint):
        seen.append(copy.deepcopy(obs));a=len(seen)-1
        if a==failed_action:return dict(qualified=False),None,None,None
        out=directory/'toy-snapshot.npz';np.savez_compressed(out,mean_m=np.ones(48)*(a+1)*1e-5,SD_m=np.ones(48)*1e-5,unit_mean=np.ones(2)*.5,layout_probability=np.ones(4)/4,material_quantiles=np.array([[2100.,7900.],[5.1,29.]]));return dict(qualified=True,snapshot=s.record(out)),None,None,None
    monkeypatch.setattr(s,'infer',inferred)
    return events,seen

def test_separate_evaluator_sameprefix_eight_independent_single_pokes(local,monkeypatch):
    events,seen=evaluation_fixture(local,monkeypatch);state=s.evaluate_worker(0,'checkpoint','toy')
    assert state['completed']and events==['all32_sealed','future_truth']and len(seen)==8
    for a,obs in enumerate(seen):
        assert len(obs)==5 and[v[0]for v in obs]==['p0','p1','p2','p3',f'a{a}'];assert np.array_equal(np.array([v[1][0]for v in obs[:4]]),np.array([11,12,13,14])*1e-5)
    trial=s.read(local/state['trials'][0]['record']['path']);assert trial['qualified']and len(trial['posteriors'])==8
    risks=[v['metrics']['field']['MSE_m2']for v in trial['posteriors']]
    assert trial['policies']['uniform_random']['MSE_m2']==np.mean(risks)

def test_one_unresolved_action_retained_and_blocks_trial_no_replacement(local,monkeypatch):
    _,seen=evaluation_fixture(local,monkeypatch,failed_action=3);state=s.evaluate_worker(0,'checkpoint','toy');trial=s.read(local/state['trials'][0]['record']['path'])
    assert len(seen)==8 and len(trial['posteriors'])==8 and not trial['qualified']and'policies'not in trial

def test_evaluator_rejects_incomplete_seals_before_model_or_future_arrays(local,monkeypatch):
    evaluation_fixture(local,monkeypatch)
    monkeypatch.setattr(s,'all_seals_gate',lambda _:(_ for _ in()).throw(ValueError('missing trial31 seal')))
    monkeypatch.setattr(s,'load_common',lambda *a:pytest.fail('No model before complete seals'));monkeypatch.setattr(s,'query_truth',lambda *a:pytest.fail('No heldtruth before seals'))
    with pytest.raises(ValueError):s.evaluate_worker(0,'checkpoint','toy')

def test_generated_five_resource_guards_only_no_GPU_or_data():
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    commands=[('controller',None,18),('input',None,17)]+[('worker',k,k)for k in range(3)]
    for role,slot,cpu in commands:
        e=dict(env)
        if slot is not None:e.update(CUDA_VISIBLE_DEVICES=s.base.old.parity.GPU.ALLOWED_UUIDS[slot],CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=['taskset','-c',str(cpu),sys.executable,'-m','experiments.run_noisy_poke_benchmark_v01','guard','--role',role]
        if slot is not None:cmd+=['--slot',str(slot)]
        result=subprocess.run(cmd,cwd=s.ROOT,env=e,capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr

def test_terminal_phase_external_closure_device_membership_and_hashes(local,monkeypatch):
    s.write(s.LOCAL/'protocol.json',dict(source_SHA256={'toy':'sha'},devices=['g0','g1','g2'],choice_split=[[i for i in range(32)if i%3==k]for k in range(3)]));digest=s.sha(s.LOCAL/'protocol.json');s.write(s.LOCAL/'release.json',dict(approved=True));release=s.record(s.LOCAL/'release.json')
    artifact=local/'toy.json';s.write(artifact,{})
    e=dict(completed=True,terminal=True,phase='choices',protocol_SHA256=digest,release=release,controller_pid=987001,workers=[])
    for k in range(3):
        path=s.LOCAL/f'choices-worker-{k}.json';s.write(path,dict(completed=True,pid=987002+k,slot=k,phase='choices',protocol_SHA256=digest,release=release,physical_UUID=f'g{k}',trials=[dict(index=i,record=s.record(artifact))for i in range(32)if i%3==k]));e['workers'].append(dict(slot=k,pid=987002+k,physical_UUID=f'g{k}',exit_code=0,cleanup_exit_code=0,reaped=True,result=s.record(path)))
    s.write(s.LOCAL/'execution-choices.json',e);monkeypatch.setattr(s.base,'absent',lambda _:None)
    with pytest.raises(FileNotFoundError):s.completed_phase('choices',digest)
    closure=dict(root_reaped=True,root_exit_code=0,protocol_SHA256=digest,source_SHA256={'toy':'sha'},execution_SHA256=s.sha(s.LOCAL/'execution-choices.json'),owned_pids=[987001,987002,987003,987004]);s.write(s.LOCAL/'root-closure-choices.json',closure)
    assert s.completed_phase('choices',digest)['completed']
    for mutation in('UUID','membership','exit','closure'):
        bad=copy.deepcopy(e)
        if mutation=='UUID':bad['workers'][0]['physical_UUID']='forbidden'
        if mutation=='membership':bad['workers'].pop()
        if mutation=='exit':bad['workers'][0]['exit_code']=1
        s.write(s.LOCAL/'execution-choices.json',bad)
        c=copy.deepcopy(closure);c['execution_SHA256']=s.sha(s.LOCAL/'execution-choices.json')
        if mutation=='closure':c['root_reaped']=False
        s.write(s.LOCAL/'root-closure-choices.json',c)
        with pytest.raises(ValueError):s.completed_phase('choices',digest)

def test_all32_canonical_seals_rankings_and_original_inputs(local,monkeypatch):
    p=dict(devices=['g0','g1','g2'],fixed_action=5,choice_split=[[i for i in range(32)if i%3==k]for k in range(3)]);s.write(s.LOCAL/'protocol.json',p);s.write(s.LOCAL/'release.json',dict(approved=True));s.write(s.LOCAL/'inputs.json',{})
    inputs=[]
    for i in range(32):
        path=s.LOCAL/f'toy-input-{i}.json';s.write(path,dict(index=i));inputs.append(dict(index=i,record=s.record(path)))
        directory=s.LOCAL/f'trial-{i:02d}'/'prefix';directory.mkdir(parents=True)
        ranking=dict(qualified=True,unique_best_action=None,numerical_ties=[1,2]);result=directory/'result-LOCAL.json';s.write(result,dict(index=i,protocol_SHA256='toy',prefix_input=inputs[-1]['record'],qualified=True,ranking=ranking,candidate_or_held_truth_read=False))
        s.write(directory.parent/'choice-seal.json',dict(index=i,protocol_SHA256='toy',release=s.record(s.LOCAL/'release.json'),prefix_result=s.record(result),prefix_input=inputs[-1]['record'],qualified=True,fixed_action=5,adaptive_action=1,numerical_ties=[1,2],candidate_or_held_truth_read=False))
    for k in range(3):s.write(s.LOCAL/f'choices-worker-{k}.json',dict(completed=True,protocol_SHA256='toy',release=s.record(s.LOCAL/'release.json'),slot=k,pid=987001+k,physical_UUID=f'g{k}',input=s.record(s.LOCAL/'inputs.json'),trials=[dict(index=i,seal=s.record(s.LOCAL/f'trial-{i:02d}'/'choice-seal.json'))for i in p['choice_split'][k]]))
    monkeypatch.setattr(s,'inputs_gate',lambda _:dict(prefixes=inputs));monkeypatch.setattr(s,'completed_phase',lambda *a:None);monkeypatch.setattr(s.base,'absent',lambda _:None)
    assert len(s.all_seals_gate('toy'))==32
    # Rehashing a modified seal/worker cannot replace the declared numerical tie rule.
    path=s.LOCAL/'trial-00/choice-seal.json';seal=s.read(path);seal['adaptive_action']=2;s.write(path,seal);worker=s.read(s.LOCAL/'choices-worker-0.json');worker['trials'][0]['seal']=s.record(path);s.write(s.LOCAL/'choices-worker-0.json',worker)
    with pytest.raises(ValueError,match='ranking'):s.all_seals_gate('toy')


def test_preflight_reserves_all_worker_budgets_before_any_launch(local,monkeypatch):
    monkeypatch.setattr(s,'validate',lambda _:{});monkeypatch.setattr(s,'inputs_gate',lambda _:None)
    monkeypatch.setattr(s,'available_memory',lambda:s.RESOURCES['RAM_reserve_bytes']+1)
    monkeypatch.setattr(s.subprocess,'Popen',lambda *a,**k:pytest.fail('No child launch without aggregateplusreserve'))
    with pytest.raises(MemoryError):s.run('choices','toy')
    assert not(s.LOCAL/'execution-choices.json').exists()


def test_modal_callback_fingerprint_cannot_cross_categories(monkeypatch):
    items=[dict(model_fingerprint=f'{k:064x}')for k in range(4)]
    monkeypatch.setattr(s.base,'load_models',lambda _:[])
    callbacks=[(None,SimpleNamespace(model_fingerprint=f'{k:064x}'))for k in range(4)]
    monkeypatch.setattr(s.base,'make_ratio_callbacks',lambda *a:callbacks)
    callbacks[0][1].model_fingerprint=callbacks[1][1].model_fingerprint
    with pytest.raises(ValueError,match='fingerprint'):s.load_common(dict(origin=dict(models=items),devices=['g0']),0)
