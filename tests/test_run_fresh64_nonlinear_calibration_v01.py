import json,os,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments import run_fresh64_nonlinear_calibration_v01 as run


def resources(mode='cpu',workers=16):
    GiB=1024**3
    return dict(workers=workers,worker_CPU_sets=[[i]for i in range(workers)],controller_CPU=16,threads_per_worker=1,device_mode=mode,worker_RSS_cap_bytes=4*GiB,aggregate_RSS_cap_bytes=workers*4*GiB,RAM_reserve_bytes=30*GiB,disk_reserve_bytes=30*GiB,maximum_writes_bytes=12*GiB,host_preflight_available_bytes=(workers*4+30)*GiB,worker_wall_cap_s=30,global_wall_cap_s=60,allowed_GPU_UUIDs=list(run.oldstudy.GPU_UUIDS)if mode=='gpu'else[],worker_GPU_UUIDs=[run.oldstudy.GPU_UUIDS[i%3]for i in range(workers)]if mode=='gpu'else[],study_VRAM_cap_bytes_per_physical_GPU=8*GiB if mode=='gpu' else 0)


def final():
    return dict(predictive_mean=np.zeros(96),predictive_SD=np.full(96,1e-6),unit_mean=np.array([.4,.5]),unit_covariance=np.eye(2)*.01,unit_CDF=np.tile(np.linspace(0,1,33),(2,1)),unit_quantiles=np.array([[.1,.4,.9],[.1,.5,.9]]),plugin_unit=np.array([.4,.5]),plugin_prediction=np.zeros(96),plugin_log_likelihood=0.,log_evidence=0.)


def test_both_models_strict_witness_before_continuous_pit(monkeypatch,tmp_path):
    calls=[]
    class Reference:
        def __init__(self,*a,**k):pass
        def qualify(self,**kwargs):
            witness=dict(passed=True,maximum_predictive_mean_difference_m=1e-7,maximum_predictive_SD_difference_m=1e-7)
            return dict(qualified=True,local_final=final(),diagnostics=dict(successive={axis:[witness]for axis in('inner','outer')}))
        def cdf(self,d,q,**kwargs):calls.append((d,q,kwargs));return q*q
    monkeypatch.setattr(run,'ConditionalAmplitudeReferenceV02',Reference);monkeypatch.setattr(run,'NonlinearConditionalReference',Reference);monkeypatch.setattr(run.oldstudy,'model_fingerprint',lambda _: 'toy')
    atlas=SimpleNamespace(fingerprint='atlas',predict=lambda *a:None)
    for kind in run.study.MODELS:
        p=tmp_path/(kind+'.json');rec=run.one_posterior(p,object(),atlas,None,[('normal_x180',np.zeros(24))],[.4,.5],kind=kind,context=dict(protocol_SHA256='toy'))
        assert rec['qualified']and rec['strict_absolute_checks']['passed']
        np.testing.assert_allclose(rec['PIT'],[.16,.25])
    assert len(calls)==4


def test_failed_absolute_witness_forbids_pit_and_invalid_pit_not_clipped(monkeypatch,tmp_path):
    class Reference:
        def __init__(self,*a,**k):pass
        def qualify(self,**kwargs):return dict(qualified=True,local_final=final(),diagnostics=dict(successive={axis:[dict(passed=True,maximum_predictive_mean_difference_m=3e-6,maximum_predictive_SD_difference_m=0.)]for axis in('inner','outer')}))
        def cdf(self,*a,**k):pytest.fail('No PIT before strictabsolute2umqualification')
    monkeypatch.setattr(run,'ConditionalAmplitudeReferenceV02',Reference);monkeypatch.setattr(run.oldstudy,'model_fingerprint',lambda _: 'toy')
    rec=run.one_posterior(tmp_path/'failed.json',object(),SimpleNamespace(fingerprint='atlas'),None,[('normal_x180',np.zeros(24))],[.4,.5],kind='linear',context={})
    assert not rec['qualified']and 'PIT'not in rec


def test_orphan_posterior_refuses_before_inference(monkeypatch,tmp_path):
    p=tmp_path/'orphan.json';p.with_suffix('.npz').write_bytes(b'unbound')
    monkeypatch.setattr(run.oldstudy,'model_fingerprint',lambda _: 'toy')
    monkeypatch.setattr(run,'ConditionalAmplitudeReferenceV02',lambda *a,**k:pytest.fail('No inference fororphan'))
    with pytest.raises(ValueError,match='Orphan'):run.one_posterior(p,object(),SimpleNamespace(fingerprint='atlas'),None,[('normal_x180',np.zeros(24))],[.4,.5],kind='linear',context={})


def test_object_toy14contexts_same_data_paired_across_models(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'LOCAL',tmp_path);monkeypatch.setattr(run,'guard',lambda *a:None)
    monkeypatch.setattr(run.study,'draw_fresh_object',lambda *a:pytest.fail('No plannedtruth/noisedraws incontrol'))
    def draws(index,directory,digest):
        value=dict(unit=np.array([.4,.5]),physical_parameters=np.array([1.,1.]),prefix_noise=np.full(24,1e-6),probe_noise=np.full((2,3,24),2e-6));np.savez(directory/'draws-LOCAL.npz',**value);return value
    monkeypatch.setattr(run,'data_for_object',draws)
    response=np.arange(144.).reshape(6,24)*1e-6
    def truths(index,directory,*args):
        for i,label in enumerate(run.study.TRUTH_LABELS):np.savez(directory/(label+'-truth-LOCAL.npz'),response_m=response[i])
        return response,dict(physical_valid=True)
    monkeypatch.setattr(run,'truth_fields',truths);monkeypatch.setattr(run.oldstudy,'model_fingerprint',lambda _: 'toy')
    calls=[]
    def posterior(path,model,atlas,ratio,observations,true_unit,*,kind,context):
        calls.append((context['phase'],kind,run.oldstudy.observation_fingerprint(observations)))
        rec=dict(qualified=True,physical_valid=True,covered=[True,True],PIT=[.4,.5],intrinsic_interval_width=[.2,.3],intrinsic_signed_bias=[0.,0.]);run.write(path,rec);np.savez(path.with_suffix('.npz'),mean=np.zeros(96),MAP_prediction=np.zeros(96));return rec
    monkeypatch.setattr(run,'one_posterior',posterior)
    model=SimpleNamespace(select=lambda labels:lambda unit:np.zeros((len(unit),len(labels),24)));atlas=SimpleNamespace(fingerprint='toy-atlas',predict=lambda unit,labels:np.zeros((len(unit),len(labels),24)))
    result=run.process_object(0,object(),model,atlas,{},'toy',0,None)
    assert result['completed']and sum(len(v)for v in result['posteriors'].values())==14 and len(calls)==14
    for key in ['prefix']+[f+'-r'+str(r)for f in run.study.FAMILIES for r in range(3)]:
        a=[x[2]for x in calls if x[0]==key];assert len(a)==2 and a[0]==a[1]


def test_whole_truth_failure_still_has14_unresolved_contexts(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'LOCAL',tmp_path);monkeypatch.setattr(run,'data_for_object',lambda *a:(_ for _ in()).throw(RuntimeError('synthetictruthfailure')))
    result=run.process_object(0,None,None,None,{},'toy',0,None)
    assert not result['physical_valid']and sum(len(v)for v in result['posteriors'].values())==14
    assert all(not r['qualified']for v in result['posteriors'].values()for r in v.values())


def test_all32_generated_worker_guards_no_context_init(tmp_path):
    for mode in('cpu','gpu'):
        r=resources(mode);path=tmp_path/(mode+'.json');run.write(path,r)
        for slot in range(16):
            env={**os.environ,'CUDA_VISIBLE_DEVICES':r['worker_GPU_UUIDs'][slot]if mode=='gpu'else'','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            cmd=run.worker_command(slot,'noactualprotocol',guard_only=True)+['--resources-json',str(path)]
            out=subprocess.run(cmd,cwd=run.ROOT,env=env,capture_output=True,text=True,timeout=20)
            assert out.returncode==0,out.stderr


def test_malformed_gpu_and_unqualifiedcost_prevent_prepare_before_draw(monkeypatch):
    r=resources('gpu');r['allowed_GPU_UUIDs'].append('GPU3forbidden')
    with pytest.raises(ValueError):run.validate_resources(r)
    monkeypatch.setattr(run.os,'sched_getaffinity',lambda _:{16});monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(run.cost,'completed_cost_gate',lambda *a:(_ for _ in()).throw(ValueError('Unqualifiedcost')))
    monkeypatch.setattr(run.study,'draw_fresh_object',lambda *a:pytest.fail('No draw beforecompletedcost+release'))
    with pytest.raises(ValueError,match='Unqualifiedcost'):run.prepare('not-a-cost-protocol',resources())


def test_missing64_objects_aggregate896_unresolved_without_draws(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'LOCAL',tmp_path);monkeypatch.setattr(run,'validate',lambda _:dict())
    monkeypatch.setattr(run,'record',lambda p:dict(path=str(p),sha256=run.sha(p)))
    monkeypatch.setattr(run.study,'draw_fresh_object',lambda *a:pytest.fail('No draws duringmissingrecordaggregation'))
    summary=run.aggregate('toy')
    assert summary['all64_denominators_retained']and summary['posterior_contexts_retained']==896 and summary['comparison_blocked']and summary['qualified_posteriors']==0
    assert all(row['coverage_bounds']==[0.,1.]and row['object_count']==64 for row in summary['primary_calibration']['nonlinear']['entries'])


def test_unfinished_mis_v03_cost_blocks_fresh_draws(monkeypatch):
    assert run.cost.__name__.endswith('run_surface_nonlinear_posterior_cost_v03')
    assert run.NonlinearConditionalReference.__name__=='NonlinearConditionalReferenceV03'
    monkeypatch.setattr(run.os,'sched_getaffinity',lambda _:{16});monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(run.cost,'completed_cost_gate',lambda *a:(_ for _ in()).throw(ValueError('MISv03 not completed/qualified')))
    monkeypatch.setattr(run.study,'draw_fresh_object',lambda *a:pytest.fail('No fresh draw before MISv03 completedqualification'))
    with pytest.raises(ValueError,match='MISv03'):run.prepare('unqualified-v03',resources())


def test_GPU_numerical_control_is_mandatory_after_CPUcost_before_draw(monkeypatch):
    monkeypatch.setattr(run.os,'sched_getaffinity',lambda _:{16});monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(run.cost,'completed_cost_gate',lambda _:dict())
    monkeypatch.setattr(run.GPU_control,'completed_GPU_control_gate',lambda _:(_ for _ in()).throw(ValueError('GPUcontrol not qualified')))
    monkeypatch.setattr(run.study,'draw_fresh_object',lambda *a:pytest.fail('No draw before qualified joint GPUcontrols'))
    with pytest.raises(ValueError,match='GPUcontrol'):run.prepare('closed-CPUcost',resources(),'unfinishedGPU')


def test_real_JSON_prepare_validate_roundtrip_preserves_plan_without_draw(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'LOCAL',tmp_path);monkeypatch.setattr(run.os,'sched_getaffinity',lambda _:{16});monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    sha=run.sha;monkeypatch.setattr(run,'sha',lambda p:run.DESIGN_SHA if p==run.DESIGN else sha(p))
    monkeypatch.setattr(run,'sources',lambda :dict(toy='a'*64));monkeypatch.setattr(run,'record',lambda p:dict(path=str(p),sha256=run.sha(p)))
    admission=dict(summary=dict(path='toy-full',sha256='b'*64))
    receipt=dict(origin_v01=dict(full_atlas=admission))
    monkeypatch.setattr(run.cost,'completed_cost_gate',lambda _:receipt)
    monkeypatch.setattr(run.GPU_control,'completed_GPU_control_gate',lambda _:dict(all_checks_passed=True))
    monkeypatch.setattr(run,'full_atlas_gate',lambda *a:(dict(atlas_fingerprint='toy-atlas'),{},admission))
    monkeypatch.setattr(run.oldstudy,'available_memory',lambda :200*1024**3)
    monkeypatch.setattr(run.study,'draw_fresh_object',lambda *a:pytest.fail('No draw in JSON prepare/validate control'))
    prepared=run.prepare('toy-CPUcost',resources(),'toy-GPUcost')
    p=run.validate(prepared['sha256'])
    assert p['plan']==dict(inner=[128,256,512],outer=[128,256,512],moment_chunk=1024)
    assert p['expected_posteriors']==896 and p['objects']==64


def test_real_admitted_fresh_atlas_loader_canonical_fingerprint(monkeypatch):
    p=json.loads((run.cost.LOCAL/'protocol.json').read_text())
    expected=dict(full_atlas=p['origin_v01']['full_atlas'],atlas_fingerprint=p['full_atlas_fingerprint'])
    monkeypatch.setattr(run.np.random,'default_rng',lambda *a:pytest.fail('No actualfreshdraws in admittedloadercontrol'))
    atlas=run.load_atlas(expected)
    assert atlas.fingerprint==p['full_atlas_fingerprint']and all(type(v)is str for v in atlas.labels)


def test_prospective_longer_wall_budget_has_hard_ceilings():
    r=resources('gpu');r.update(worker_wall_cap_s=18000,global_wall_cap_s=19800)
    assert run.validate_resources(r)==r
    r['worker_wall_cap_s']=18001
    with pytest.raises(ValueError,match='bounded worker/globalwalls'):run.validate_resources(r)
    r.update(worker_wall_cap_s=18000,global_wall_cap_s=19801)
    with pytest.raises(ValueError,match='bounded worker/globalwalls'):run.validate_resources(r)
