"""Synthetic metadata/archives only; never reads the original campaign."""
from copy import deepcopy
import json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments import run_fresh64_numerical_continuation_v01 as run

DIGEST='c'*64


def resources(workers=2,cap=4):
    G=1024**3
    return dict(workers=workers,worker_CPUs=list(range(workers)),controller_CPU=16,threads=1,GPU=False,worker_RSS_bytes=cap*G,controller_RSS_bytes=4*G,aggregate_RSS_bytes=(workers*cap+4)*G,RAM_reserve_bytes=30*G,disk_reserve_bytes=30*G,host_preflight_bytes=(workers*cap+34)*G,maximum_writes_bytes=12*G,worker_wall_s=2400,global_wall_s=2700)


def local_records(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'LOCAL',tmp_path/'continued');monkeypatch.setattr(run.parent,'LOCAL',tmp_path/'original')
    def record(path):return dict(path=str(Path(path).relative_to(tmp_path)),sha256=run.sha(path))
    monkeypatch.setattr(run,'record',record)
    return record


def entry(kind='nonlinear',phase='normal-r2',index=5):
    plan=dict(inner=[128,256,512,1024],outer=[128,256,512],moment_chunk=1024)
    old=dict(protocol_SHA256='a'*64,index=index,phase=phase,draws_SHA256='d'*64,observations_SHA256='e'*64,model_fingerprint='f'*64,true_unit_SHA256='0'*64)
    return dict(id=run.lineage.context_id(index,kind,phase),index=index,kind=kind,phase=phase,decision='refine',plan=plan,levels=run.lineage.integration_levels(plan),original=dict(context=old,json=dict(path='old.json',sha256='1'*64),raw=dict(path='old.npz',sha256='b'*64)))


def test_flat_closure_all_transitive_metadata_source_and_raw_bytes(monkeypatch,tmp_path):
    record=local_records(monkeypatch,tmp_path)
    source=tmp_path/'toy.py';source.write_text('value=3\n');raw=tmp_path/'raw.npz';np.savez(raw,values=np.arange(3))
    child=tmp_path/'child.json';child.write_text(json.dumps(dict(raw=record(raw),source_SHA256={'toy.py':run.sha(source)})))
    root=dict(child=record(child));pins=run.flatten_pins([root]);assert set(pins)=={'child.json','toy.py','raw.npz'}
    run.verify_flat(pins);raw.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed flat'):run.verify_flat(pins)


def test_flat_closure_refuses_missing_or_conflicting_hash_before_arrays(monkeypatch,tmp_path):
    record=local_records(monkeypatch,tmp_path);p=tmp_path/'file.json';p.write_text('{}')
    with pytest.raises(ValueError,match='Exact in-workspace'):run.flatten_pins([dict(path='file.json',sha256='1'*64)])
    with pytest.raises(ValueError,match='Missing transitive source/artifact'):run.flatten_pins([dict(source_SHA256={'experiments/missing.py':'1'*64})])


@pytest.mark.parametrize('missing',['missing-LOCAL.npz','runs/missing/archive.commit.json','missing-input.json'])
def test_flat_closure_refuses_missing_transitive_non_source_map(monkeypatch,tmp_path,missing):
    local_records(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='Missing transitive source/artifact'):
        run.flatten_pins([dict(checkpoint_SHA256={missing:'a'*64})])


def test_both_axis_release_hold_precedes_release_file_arrays_and_model(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'LOCAL',tmp_path)
    monkeypatch.setattr(run,'validate',lambda _:({'both_axis_control_required':True},{}))
    monkeypatch.setattr(run.np,'load',lambda *a,**k:pytest.fail('No arrays before reviewed both-axis control'))
    monkeypatch.setattr(run.parent.oldstudy,'make_model',lambda *a,**k:pytest.fail('No model before reviewed both-axis control'))
    with pytest.raises(ValueError,match='separately reviewed same-observation 8GiB control'):
        run.release_gate(DIGEST)


def test_prepare_JSON_roundtrip_flat_validation_never_replays_old_gates(monkeypatch,tmp_path):
    record=local_records(monkeypatch,tmp_path);source=tmp_path/'toy.py';source.write_text('x=1\n')
    protocol=tmp_path/'parent-protocol.json';protocol.write_text('{}');release=tmp_path/'parent-release.json';release.write_text('{}');execution=tmp_path/'fine-execution.json';execution.write_text(json.dumps(dict(controller_pid=211,worker_pid=212)))
    old=dict(protocol=record(protocol),release=record(release));pp=dict(atlas_fingerprint='toy',full_atlas={},cost_receipt={},target_labels=list(run.design.TARGETS))
    control=dict(origin=dict(primary_protocol=old['protocol'],primary_release=old['release']),execution=record(execution),source_SHA256={'toy.py':run.sha(source)})
    monkeypatch.setattr(run,'parent_admission',lambda :(pp,[],{},old,[111]));monkeypatch.setattr(run.fine,'completed_control_gate',lambda _:control);monkeypatch.setattr(run,'guard',lambda *a:None);monkeypatch.setattr(run.parent.oldstudy,'available_memory',lambda :200*1024**3);monkeypatch.setattr(run,'sources',lambda :{'toy.py':run.sha(source)});monkeypatch.setattr(run.parent.full.prefix,'absent',lambda _:None)
    prepared=run.prepare('fine',resources());assert prepared['path'].endswith('protocol.json')
    monkeypatch.setattr(run,'parent_admission',lambda :pytest.fail('No recursive parent semantic replay in validate'))
    monkeypatch.setattr(run.fine,'completed_control_gate',lambda _:pytest.fail('No recursive fine semantic replay in validate'))
    p,c=run.validate(prepared['sha256']);assert len(c['entries'])==896 and c['counts']['blocked']==896
    assert p['both_axis_control_required']is False
    protocol_path=run.LOCAL/'protocol.json';p['both_axis_control_required']=True;protocol_path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='matches complete census'):run.validate(run.sha(protocol_path))
    p['both_axis_control_required']=False;protocol_path.write_text(json.dumps(p));prepared['sha256']=run.sha(protocol_path)
    source.write_text('tampered\n')
    with pytest.raises(ValueError,match='Frozen new source'):run.validate(prepared['sha256'])


def test_resource_both_axes_requires_explicit8GiB_not_discarding_jobs():
    with pytest.raises(ValueError,match='Frozen4/8GiB'):run.validate_resources(resources(cap=4),both_axes=True)
    assert run.validate_resources(resources(workers=8,cap=8),both_axes=True)['aggregate_RSS_bytes']==68*1024**3
    with pytest.raises(ValueError):run.validate_resources(resources(workers=9))


def test_only_acquired_observations_replayed_exactly_no_rng(monkeypatch,tmp_path):
    local_records(monkeypatch,tmp_path);i=5;directory=run.parent.LOCAL/f'object-{i:02d}';directory.mkdir(parents=True)
    pn=np.arange(24)*1e-8;probes=np.arange(2*3*24).reshape(2,3,24)*1e-8;normal=np.arange(24)*1e-5;axial=normal*2
    np.savez(directory/'draws-LOCAL.npz',prefix_noise=pn,probe_noise=probes,unit=np.array([.4,.8]))
    for label,v in[(run.design.PREFIX,normal),(run.design.PROBES[1],axial)]:np.savez(directory/(label+'-truth-LOCAL.npz'),response_m=v)
    monkeypatch.setattr(run.np.random,'default_rng',lambda *a:pytest.fail('No new draw/noise allowed'))
    for phase in('prefix','normal-r2','axial-r1'):
        e=entry(phase=phase);expected=[(run.design.PREFIX,normal+pn)]
        if phase=='normal-r2':expected.append((run.design.PREFIX,normal+probes[0,2]))
        if phase=='axial-r1':expected.append((run.design.PROBES[1],axial+probes[1,1]))
        e['original']['context']['observations_SHA256']=run.parent.oldstudy.observation_fingerprint(expected)
        got=run.acquired_observations(e);assert [k for k,_ in got]==[k for k,_ in expected]
        for (_,a),(_,b)in zip(got,expected):np.testing.assert_array_equal(a,b)
    e['original']['context']['observations_SHA256']='9'*64
    with pytest.raises(ValueError,match='Exact original acquired'):run.acquired_observations(e)


def test_qualified_original_cannot_enter_job_or_touch_arrays(monkeypatch,tmp_path):
    local_records(monkeypatch,tmp_path);e=entry();e['decision']='reuse_qualified'
    monkeypatch.setattr(run.np,'load',lambda *a,**k:pytest.fail('No original qualified rerun'))
    with pytest.raises(ValueError,match='Only original unqualified'):run.posterior_job(e,{},DIGEST,None,None,None)


def test_strict_admission_before_PIT_trueunit_or_held_truth(monkeypatch,tmp_path):
    local_records(monkeypatch,tmp_path);e=entry();labels=list(run.design.TARGETS)
    final=dict(predictive_mean=np.zeros(96),predictive_SD=np.ones(96)*1e-6,unit_mean=np.ones(2)*.5,unit_covariance=np.eye(2),unit_CDF=np.tile(np.linspace(0,1,33),(2,1)),unit_quantiles=np.tile([.05,.5,.95],(2,1)),plugin_unit=np.ones(2)*.5,plugin_prediction=np.zeros(96),plugin_log_likelihood=0.)
    class Fake:
        def __init__(self,*a,**k):pass
        def qualify(self,**kwargs):return dict(qualified=True,local_final=final,diagnostics=dict(levels=e['levels']))
    monkeypatch.setattr(run,'NonlinearConditionalReferenceV04',Fake);monkeypatch.setattr(run,'acquired_observations',lambda _:[]);monkeypatch.setattr(run.parent.oldstudy,'model_fingerprint',lambda _:'f'*64);monkeypatch.setattr(run.parent.cost,'engine_gates',lambda _:dict(passed=True));monkeypatch.setattr(run.parent.cost,'absolute_engine_witnesses',lambda _:dict(passed=False));monkeypatch.setattr(run.design,'exact_density_calibration',lambda *a,**k:pytest.fail('No PIT without absolute witness'))
    monkeypatch.setattr(run.np,'load',lambda *a,**k:pytest.fail('No trueunit or heldtruth before numerical qualification'))
    p=dict(parent_configuration=dict(target_labels=labels,atlas_fingerprint='toy'))
    packet=run.posterior_job(e,p,DIGEST,None,SimpleNamespace(predict=lambda *a:None),None)
    assert packet['record']['qualified']is False and packet['record']['context']['refinement_plan']==e['plan']
    assert packet['binding']['raw']and 'PIT'not in packet['record']


def test_missing_refinements_aggregate64_896_blocked_without_data_or_draws(monkeypatch,tmp_path):
    local_records(monkeypatch,tmp_path);classification=run.lineage.census([],{});rows=[run.parent.placeholder(i,'missing')for i in range(64)]
    monkeypatch.setattr(run,'collect_updates',lambda *a:{});monkeypatch.setattr(run,'load_parent_rows',lambda :rows);monkeypatch.setattr(run.np.random,'default_rng',lambda *a:pytest.fail('No new parameter/noise/bootstrap on unresolved data'))
    p=dict(parent={},fine_control={});result=run.aggregate(p,classification,DIGEST)
    assert result['objects']==64 and result['expected_posteriors']==896 and result['unresolved']==896 and result['comparison_blocked']
    assert result['qualified_posteriors']==0 and len(result['object_summaries'])==64


def test_worker_release_before_model_atlas_or_arrays(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'validate',lambda _:({'both_axis_control_required':False},{}));monkeypatch.setattr(run,'LOCAL',tmp_path)
    monkeypatch.setattr(run.parent.oldstudy,'make_model',lambda *a,**k:pytest.fail('No model before release'))
    monkeypatch.setattr(run.np,'load',lambda *a,**k:pytest.fail('No archive before release'))
    with pytest.raises(FileNotFoundError):run.worker(0,'not-released')


def test_eight_generated_CPU_noGPU_worker_guards_no_data(tmp_path):
    r=resources(workers=8,cap=8);path=tmp_path/'resources.json';path.write_text(json.dumps(r))
    for slot in range(8):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        out=subprocess.run(run.worker_command(slot,'no-protocol',guard_only=True)+['--resources-json',str(path)],cwd=run.ROOT,env=env,capture_output=True,text=True,timeout=20)
        assert out.returncode==0,out.stderr


@pytest.mark.parametrize('kind',['linear','nonlinear'])
def test_successful_job_full_lean_archive_lineage_and_PIT_uses_selected_orders(monkeypatch,tmp_path,kind):
    local_records(monkeypatch,tmp_path);e=entry(kind=kind);unit=np.array([.4,.5]);e['original']['context']['true_unit_SHA256']=run.hashlib.sha256(np.ascontiguousarray(unit,dtype='<f8').tobytes()).hexdigest()
    original=run.parent.LOCAL/'object-05';original.mkdir(parents=True);np.savez(original/'draws-LOCAL.npz',unit=unit)
    for label in run.design.TARGETS:np.savez(original/(label+'-truth-LOCAL.npz'),response_m=np.zeros(24))
    final=dict(predictive_mean=np.zeros(96),predictive_SD=np.ones(96)*1e-6,unit_mean=np.ones(2)*.5,unit_covariance=np.eye(2),unit_CDF=np.tile(np.linspace(0,1,33),(2,1)),unit_quantiles=np.tile([.05,.5,.95],(2,1)),plugin_unit=np.ones(2)*.5,plugin_prediction=np.zeros(96),plugin_log_likelihood=0.)
    calls=[]
    class Fake:
        def __init__(self,*a,**kwargs):calls.append(kwargs)
        def qualify(self,*,plan):
            assert list(plan.inner)==e['plan']['inner']and list(plan.outer)==e['plan']['outer']
            return dict(qualified=True,local_final=final,diagnostics=dict(levels=e['levels']))
        def cdf(self,d,threshold,*,outer,inner,log_evidence):
            assert outer==512 and inner==1024
            return threshold
    final['log_evidence']=1.
    monkeypatch.setattr(run,'LinearAmplitudeExtendedReferenceV01',Fake);monkeypatch.setattr(run,'NonlinearConditionalReferenceV04',Fake);monkeypatch.setattr(run,'acquired_observations',lambda _:[]);monkeypatch.setattr(run.parent.oldstudy,'model_fingerprint',lambda _:'f'*64);monkeypatch.setattr(run.parent.cost,'engine_gates',lambda _:dict(passed=True));monkeypatch.setattr(run.parent.cost,'absolute_engine_witnesses',lambda _:dict(passed=True))
    p=dict(parent_configuration=dict(target_labels=list(run.design.TARGETS),atlas_fingerprint='toy'))
    packet=run.posterior_job(e,p,DIGEST,None,SimpleNamespace(predict=lambda *a:None),None);r=packet['record']
    assert r['qualified']and r['target_RMS_m']==0 and r['MAP_target_RMS_m']==0
    assert r['numerical_backend']==run.BACKENDS[kind]and r['context']['parent_record_SHA256']==e['original']['json']['sha256']
    assert r['context']['recipe_SHA256']==run.recipe_SHA(kind,e,list(run.design.TARGETS),'toy')
    raw=tmp_path/packet['binding']['raw']['path']
    with np.load(raw,allow_pickle=False)as z:np.testing.assert_array_equal(z['PIT'],unit);assert z['mean'].shape==(96,)and z['MAP_prediction'].shape==(96,)
    if kind=='linear':assert 'log_ratio_predict'in calls[0]
    else:assert 'residual_predict'in calls[0]
