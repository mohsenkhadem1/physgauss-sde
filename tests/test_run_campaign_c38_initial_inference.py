"""Manufactured metadata/point fields and owned tiny subprocesses only.

No actual candidate/prefix/target arrays, identity entropy, observation draws,
forward solves or inference integrations are consumed by these controls.
"""
import copy,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import pytest
from experiments import run_campaign_c38_initial_inference as m
from experiments.run_campaign_c38_static_prequalification import cases
from experiments.campaign_c38_table_validation import POINTS


def put(root,name,value=None,raw=None):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    if raw is not None:path.write_bytes(raw)
    else:path.write_text(json.dumps(value))
    return m.record(root,path)


def fixtures(root,monkeypatch):
    monkeypatch.setattr(m,'source_names',lambda:['source.py'])
    monkeypatch.setattr(m.obs,'load_frozen_design',lambda root:{'manufactured':True})
    src=put(root,'source.py',raw=b'no actual sources/data')
    cand=put(root,'candidate.npz',raw=b'fake candidate; no extraction allowed')
    prefix=[]
    for i in range(64):
        token=format(i+1,'032x');r=put(root,'PUBLIC/'+token+'.npz',raw=b'fake public prefix')
        prefix.append(dict(object_id=token,prefix=dict(path=Path(r['path']).name,sha256=r['sha256'])))
    manifest=dict(schema='C38-public-controlled-ideal-prefixes-v01',completed=True,information_role='PUBLIC_ACQUIRED_IDEAL_STATIC_POINTS_B',records=prefix,count=64,physical_configurations=4,noise_repeats_per_configuration=16,independent_population_objects=False,design_sha256=m.obs.DESIGN_SHA256,is_RGBD_perception=False,is_temporal_forecast=False)
    pub=put(root,'PUBLIC/manifest.json',manifest)
    groups=[];captures=[]
    for c in cases():
        prs=[]
        for n in ([32,16,16],[40,20,20]):
            r=put(root,'linear/'+c['id']+'-'+str(n[0])+'.npz',raw=b'fake linear capture')
            captures.append(r);prs.append(dict(case=c,cells=n,capture=r,finite=True))
        groups.append(dict(case=c,prediction_records=prs))
    seal=put(root,'seal.json',dict(completed=True,all28_predictions_sealed_before_nonlinear_target_access=True,nonlinear_target_access=False,rows=groups))
    summary=put(root,'summary.json',dict(completed=True,predictions_completed=28,all8_basis_original_residual_checks_passed=True,all8_independent_direction_checks_passed=True,nonlinear_target_reads=False,predictions_seal=seal))
    terminal=put(root,'terminal.json',dict(completed=True,all4_basis_and28_prediction_capture_hashes_match=True,all39_source_pins_unchanged=True,owned_PID_absence={'123':True},captures=captures,source_sha256={'source.py':src['sha256']}))
    return dict(candidate_archive=cand['path'],public_manifest=pub['path'],linear_summary=summary['path'],linear_terminal=terminal['path'])


def test_metadata_prepare_exclusive_write_boundary_and_no_arrays_or_noise(tmp_path,monkeypatch):
    args=fixtures(tmp_path,monkeypatch)
    monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('NPZ during prepare')))
    monkeypatch.setattr(m.obs,'mint_opaque_ids',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Identity draw')))
    monkeypatch.setattr(m.obs,'measurement_error',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Noise draw')))
    original=Path.open;writes=[]
    def guarded(self,mode='r',*a,**k):
        if any(x in mode for x in 'wax+'):
            assert self.resolve().is_relative_to(tmp_path);writes.append(self)
        return original(self,mode,*a,**k)
    monkeypatch.setattr(Path,'open',guarded)
    h=m.prepare('opdev/p.json','output',root=tmp_path,**args);p=m.validate('opdev/p.json',h,root=tmp_path)
    assert len(writes)==1 and not (tmp_path/'output').exists()
    assert len(p['objects'])==64 and len(p['model_inputs']['linear']['rows'])==28
    assert set(p['model_inputs'])=={'candidate_archive','public_manifest','linear'}
    assert 'PRIVATE' not in json.dumps(p['model_inputs'])
    with pytest.raises(FileExistsError):m.prepare('opdev/p.json','output',root=tmp_path,**args)
    bad=json.loads((tmp_path/'opdev/p.json').read_text());bad['source_sha256']={};(tmp_path/'opdev/p.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='Full source'):m.validate('opdev/p.json',m.sha(tmp_path/'opdev/p.json'),root=tmp_path)


def test_public_manifest_basename_and_symlink_escape_rejected_before_prefix_read(tmp_path,monkeypatch):
    args=fixtures(tmp_path,monkeypatch);p=tmp_path/args['public_manifest'];orig=json.loads(p.read_text())
    bad=copy.deepcopy(orig);bad['records'][0]['prefix']['path']='../PRIVATE/future.npz';p.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='basename'):m.public_plan(tmp_path,m.record(tmp_path,p))
    p.write_text(json.dumps(orig));r=orig['records'][0];prefix=p.parent/r['prefix']['path'];prefix.unlink();outside=tmp_path/'PRIVATE/future.npz';outside.parent.mkdir();outside.write_bytes(b'fake public prefix');prefix.symlink_to(outside)
    with pytest.raises(ValueError,match='escapes manifest parent'):m.public_plan(tmp_path,m.record(tmp_path,p))
    p.write_text(json.dumps(dict(orig,private_configuration_index=0)))
    with pytest.raises(ValueError,match='PUBLIC64'):m.public_plan(tmp_path,m.record(tmp_path,p))


def test_last_linear_capture_tamper_rejects_before_any_array_extract(tmp_path,monkeypatch):
    args=fixtures(tmp_path,monkeypatch);h=m.prepare('p.json','out',root=tmp_path,**args);p=m.validate('p.json',h,root=tmp_path)
    last=p['model_inputs']['linear']['rows'][-1]['capture'];(tmp_path/last['path']).write_bytes(b'corrupt last capture')
    monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Array before global hash barrier')))
    with pytest.raises(ValueError,match='Changed input'):m.build_linear_library(tmp_path,p,tmp_path/'derived.npz')
    with pytest.raises(ValueError,match='Changed input'):m.validate('p.json',h,root=tmp_path)
    assert not (tmp_path/'derived.npz').exists()


def test_linear_extraction_member_allowlist_and_compiled_scaling(tmp_path,monkeypatch):
    args=fixtures(tmp_path,monkeypatch);h=m.prepare('p.json','out',root=tmp_path,**args);p=m.validate('p.json',h,root=tmp_path)
    from test_campaign_c38_linear_candidate_forward import rows as manufactured
    data={(r['case']['id'],r['cells'][0]):r for r in manufactured()};byname={r['capture']['path']:data[(r['case']['id'],r['cells'][0])] for r in p['model_inputs']['linear']['rows']};reads=[];real=np.load
    class Spy:
        def __init__(self,name):self.row=byname[Path(name).relative_to(tmp_path).as_posix()]
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def __getitem__(self,key):
            assert key in ('material_probes_m','probe_positions_m');reads.append(key);return self.row[key]
    monkeypatch.setattr(np,'load',lambda path,**kw:Spy(path))
    r=m.build_linear_library(tmp_path,p,tmp_path/'derived.npz');assert len(reads)==56
    monkeypatch.setattr(np,'load',real);model=m.load_linear_library(tmp_path,r)
    np.testing.assert_allclose(model.predict_displacement('both_ends',1.,100000.,[11]),.75*(.02+.003)*POINTS[[11]],atol=1e-16)


def fake_controller(tmp,monkeypatch,*,fail_first=False,numeric_first=False,resource_first=False):
    p=dict(output='out',objects=[dict(object_id=format(i+1,'032x')) for i in range(64)],model_inputs=dict(candidate_archive={'path':'fake-candidate','sha256':'a'*64},public_manifest={'path':'fake-public','sha256':'b'*64}),taskset={'path':'fake-taskset'})
    monkeypatch.setattr(m,'validate',lambda *a,**k:p)
    monkeypatch.setattr(m,'guard',lambda *a:None)
    monkeypatch.setattr(m.time,'sleep',lambda *a:None)
    def build(root,p,dest):dest.write_bytes(b'fake derived');return m.record(root,dest)
    monkeypatch.setattr(m,'build_linear_library',build);calls=[];cleaned=[]
    class Fake:
        def __init__(self,cmd,**kwargs):
            assert kwargs['cwd']==tmp and kwargs['start_new_session'] is True
            assert not any('PRIVATE' in str(x) for x in cmd)
            self.pid=900000+len(calls);calls.append(self);self.cmd=cmd;self.code=None if resource_first and len(calls)==1 else (1 if fail_first and len(calls)==1 else 0)
            token=cmd[cmd.index('--object-id')+1];slot=int(cmd[cmd.index('--slot')+1]);self.token=token
            d=tmp/'out/objects'/token;d.mkdir(parents=True);models={}
            for name in m.MODELS:
                b=d/name;b.mkdir();files={}
                for n in ('posterior-arrays.npz','posterior-structure.json','numerics.json'):
                    (b/n).write_bytes(b'fake preserved result');files[n]=m.record(tmp,b/n)
                models[name]=dict(execution_completed=True,completed=not(numeric_first and len(calls)==1),numerical_controls_passed=not(numeric_first and len(calls)==1),files=files)
            m.write_new(d/'result.json',dict(completed=True,pid=self.pid,object_id=token,slot=slot,protocol_sha256='c'*64,models=models,private_truth_access=False,all_models_numerical_passed=not(numeric_first and len(calls)==1)))
        def poll(self):return self.code
        def wait(self,timeout=None):return self.code
    monkeypatch.setattr(m.subprocess,'Popen',Fake)
    def cleanup(proc,grace=3):cleaned.append(proc.pid);proc.code=-15;return -15
    monkeypatch.setattr(m,'cleanup_owned',cleanup)
    monkeypatch.setattr(m,'current_rss',lambda pid:m.RESOURCES['worker_RSS_bytes']+1 if resource_first and pid==900000 else 0)
    return p,calls,cleaned


def test_spy_queue_seals_all64_and_retains_numeric_failure_no_evaluation(tmp_path,monkeypatch):
    p,calls,_=fake_controller(tmp_path,monkeypatch,numeric_first=True)
    s=m.run('p.json','c'*64,root=tmp_path)
    assert s['completed'] and not s['technical_failure'] and len(calls)==64
    seal=json.loads((tmp_path/'out/prediction-seal.json').read_text())
    assert seal['count']==64 and seal['model_executions']==128 and not seal['all_numerical_controls_passed']
    assert seal['all_predictions_sealed_before_private_evaluation'] and not seal['private_truth_access']
    assert all(r['reaped'] for r in s['records']) and not s['live']
    with pytest.raises(FileExistsError):m.run('p.json','c'*64,root=tmp_path)


@pytest.mark.parametrize('resource_failure',[False,True])
def test_technical_or_memory_failure_stops_new_launches_drains_first8_keeps_queue(tmp_path,monkeypatch,resource_failure):
    p,calls,cleaned=fake_controller(tmp_path,monkeypatch,fail_first=not resource_failure,resource_first=resource_failure)
    s=m.run('p.json','c'*64,root=tmp_path)
    assert not s['completed'] and s['technical_failure'] and len(calls)==8 and len(s['records'])==8 and len(s['queued_object_ids'])==56
    assert s['all_owned_workers_reaped'] and not (tmp_path/'out/prediction-seal.json').exists()
    if resource_failure:assert cleaned==[900000]


def test_owned_timeout_cleanup_reaps_only_own_process_group(tmp_path):
    # An owned leader waits for its child after TERM, so both are demonstrably
    # reaped. A separate session is kept alive to detect broad/name-based kills.
    script="""import subprocess,sys,time,signal
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
open(sys.argv[1],'w').write(str(p.pid))
def term(*a):
 p.wait(timeout=2);sys.exit(0)
signal.signal(signal.SIGTERM,term)
while True:time.sleep(.1)
"""
    childfile=tmp_path/'child.pid';proc=subprocess.Popen([sys.executable,'-c',script,str(childfile)],start_new_session=True)
    other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    try:
        deadline=time.monotonic()+3
        while not childfile.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert childfile.exists();child=int(childfile.read_text())
        assert m.cleanup_owned(proc,grace=3)==0 and proc.poll()==0
        assert not Path('/proc',str(proc.pid)).exists() and not Path('/proc',str(child)).exists()
        assert other.poll() is None
    finally:
        m.cleanup_owned(proc);m.cleanup_owned(other)
